#!/usr/bin/env bash
# Watch the synced cloud inbox by polling (inotify doesn't fire on /mnt/c for
# files written by Windows). Paths come from core/config.py -- one source of
# truth, overridable with TRANSCRIBE_INBOX / TRANSCRIBE_CLOUD_ROOT.
#
# Safe to leave running while the realtime server is up: each transcribe run
# takes the GPU lease, so it waits its turn instead of colliding.
#
# A failed file never loops forever: batch/pipeline.py's exit code says whether
# retrying can help, and files that cannot be fixed by retrying are moved out of
# the inbox into the failed folder.
#
# Portable on purpose: it runs on Linux/WSL2 (GNU userland) and on macOS (bash 3.2,
# BSD userland). So no flock, no `stat -c`, no bash-4-only syntax in here.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

# Everything below also lands in logs/watcher.log, so a failure that scrolled
# past -- or happened in a terminal that is gone -- can still be read afterwards.
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/watcher.log") 2>&1

INBOX="$(python core/config.py INBOX)"
LOCAL_AUDIO="$(python core/config.py LOCAL_AUDIO)"
PROCESSED="$(python core/config.py PROCESSED_DIR)"
FAILED="$(python core/config.py FAILED_DIR)"
ATTEMPTS_DIR="$LOCAL_AUDIO/.attempts"   # per-file failure counts; survive a restart
POLL_INTERVAL=15          # seconds between inbox scans
STABLE_CHECKS=2           # consecutive equal-size reads before we call it synced
STABLE_WAIT=5             # seconds between size reads
MAX_STABLE_LOOPS=60       # give up waiting after ~5 min
STALL_WARN_AFTER=12       # warn once we've waited this many loops (~1 min)
MAX_ATTEMPTS=3            # retryable failures allowed before a file is set aside
# Exit codes from batch/pipeline.py -- keep in sync with EXIT_* there.
EXIT_PERMANENT=2          # the same input will fail the same way
EXIT_BUSY=75              # GPU held elsewhere; not the file's fault, not counted
# A directory, because mkdir is atomic everywhere. Overridable for tests.
LOCKDIR="${SUPERMEET_WATCHER_LOCK:-/tmp/transcriber-watcher.lock.d}"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# Size in bytes, 0 if the file is unreadable. `wc -c <` is the one spelling that
# GNU and BSD agree on (stat -c%s is GNU only, stat -f%z is BSD only); BSD pads
# the number with spaces, which the arithmetic expansion ignores.
file_size() {
    local n
    n="$(wc -c < "$1" 2>/dev/null)" || n=0
    echo $(( ${n:-0} ))
}

# True while the process in the lock is really a running watcher. Checking the
# command line, not just that the pid exists, so a pid reused after a crash or
# reboot does not keep the lock held forever.
lock_holder_alive() {
    local pid="$1" cmd
    case "$pid" in ''|*[!0-9]*) return 1 ;; esac
    cmd="$(ps -p "$pid" -o command= 2>/dev/null)" || return 1
    case "$cmd" in *watcher.sh*) return 0 ;; *) return 1 ;; esac
}

# One watcher at a time, or two runs fight over the same file. flock does not
# exist on macOS, so the lock is a directory created with mkdir (atomic, and it
# fails if it already exists) holding the owner's pid. A lock whose owner is gone
# (kill -9, power loss) is detected and taken over.
acquire_lock() {
    local tries=0 pid
    while ! mkdir "$LOCKDIR" 2>/dev/null; do
        pid="$(cat "$LOCKDIR/pid" 2>/dev/null || true)"
        if [ -z "$pid" ] && [ "$tries" -eq 0 ] && [ -d "$LOCKDIR" ]; then
            # Directory exists but no pid yet: its owner may be between mkdir and
            # writing the pid. Give it a moment before calling it stale.
            sleep 1; tries=1; continue
        fi
        if lock_holder_alive "$pid"; then
            log "Another watcher holds $LOCKDIR (pid $pid) -- exiting."
            return 1
        fi
        [ "$tries" -ge 3 ] && { log "Cannot take the lock $LOCKDIR -- exiting."; return 1; }
        tries=$((tries + 1))
        log "Removing stale lock ${pid:+(pid $pid is gone) }$LOCKDIR"
        rm -f "$LOCKDIR/pid"
        rmdir "$LOCKDIR" 2>/dev/null || true
    done
    echo "$$" > "$LOCKDIR/pid"
}

# sleep as a background job + wait: bash only runs a signal trap once the
# foreground command ends, so a plain `sleep 15` would delay `kill <pid>`
# (and the lock cleanup) by up to 15 seconds. `wait` returns at once.
nap() { sleep "$1" & wait $! 2>/dev/null; }

CHILD=""   # pid of the pipeline run in flight, so stopping the watcher stops it too

release_lock() {
    if [ -n "$CHILD" ]; then kill "$CHILD" 2>/dev/null; fi
    # Only ever remove a lock that is ours.
    if [ "$(cat "$LOCKDIR/pid" 2>/dev/null || true)" = "$$" ]; then
        rm -f "$LOCKDIR/pid"
        rmdir "$LOCKDIR" 2>/dev/null || true
    fi
}

# Move a file out of the inbox so the next scan stops picking it up. The local
# copy and any cached transcript stay, so dropping it back in after fixing the
# cause resumes cheaply.
set_aside() {
    local src="$1" name="$2"
    local dest="$FAILED/$name"
    [ -e "$dest" ] && dest="$FAILED/$(date +%Y%m%d-%H%M%S)_$name"
    if mv "$src" "$dest"; then
        rm -f "$ATTEMPTS_DIR/$name.count"
        log "  Moved to $dest"
        log "  Fix the cause, then drop it back in the inbox to retry."
    else
        log "  ERROR: could not move $name out of the inbox; it will be picked up again."
    fi
}

if ! acquire_lock; then
    exit 1
fi
# After acquiring, so a watcher that lost the race never deletes the winner's lock.
# INT/TERM/HUP exit explicitly so the EXIT trap (and the cleanup) always runs.
trap release_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

if [ ! -d "$INBOX" ]; then
    log "FATAL: inbox does not exist: $INBOX"
    log "  If you just switched cloud providers, create the folder or set"
    log "  TRANSCRIBE_INBOX / TRANSCRIBE_CLOUD_ROOT. Check: python config.py"
    exit 1
fi

mkdir -p "$PROCESSED" "$FAILED" "$LOCAL_AUDIO" "$ATTEMPTS_DIR"
log "Watcher started (polling every ${POLL_INTERVAL}s)"
log "  inbox:     $INBOX"
log "  archive:   $PROCESSED"
log "  failed:    $FAILED"
log "  log:       $LOG_DIR/watcher.log"

while true; do
    while IFS= read -r INBOX_PATH; do
        [ -n "$INBOX_PATH" ] || continue
        FILE="$(basename "$INBOX_PATH")"

        # Wait for the cloud client to finish writing: size must hold steady.
        log "Detected: $FILE - waiting for sync..."
        PREV=-1; STABLE=0; SYNCED=0
        for ((i = 1; i <= MAX_STABLE_LOOPS; i++)); do
            CURR="$(file_size "$INBOX_PATH")"
            if [ "$CURR" = "$PREV" ] && [ "$CURR" -gt 1024 ]; then
                STABLE=$((STABLE + 1))
                if [ "$STABLE" -ge "$STABLE_CHECKS" ]; then SYNCED=1; break; fi
            else
                STABLE=0
            fi
            # Say something when a file is clearly stuck, instead of failing mute.
            if [ "$i" -eq "$STALL_WARN_AFTER" ]; then
                log "  ...still waiting on $FILE (${CURR} bytes). Sync may be stalled."
            fi
            PREV="$CURR"
            nap "$STABLE_WAIT"
        done

        # Never transcribe a partially-synced file. Leave it; retry next scan.
        if [ "$SYNCED" -ne 1 ]; then
            log "SKIP: $FILE never stabilised after $((MAX_STABLE_LOOPS * STABLE_WAIT))s"
            log "  (currently $(file_size "$INBOX_PATH") bytes)"
            log "  Left in inbox; will retry. Check the cloud client if this repeats."
            continue
        fi

        # Copy to local disk (faster I/O) and run before archiving, so a crash
        # doesn't strand the only copy outside the inbox.
        if ! cp "$INBOX_PATH" "$LOCAL_AUDIO/$FILE"; then
            log "ERROR: copy failed for $FILE - leaving in inbox"
            continue
        fi

        log "Running pipeline: $FILE"
        # </dev/null: this loop's stdin is the find output, and any child that
        # reads stdin (ffmpeg does) would eat the paths still queued behind it.
        # Background + wait, so a signal to the watcher is handled while the
        # (long) pipeline run is still going, not after it.
        python supermeet.py transcribe "$LOCAL_AUDIO/$FILE" </dev/null &
        CHILD=$!
        wait "$CHILD"
        RC=$?
        CHILD=""
        COUNTER="$ATTEMPTS_DIR/$FILE.count"
        case "$RC" in
            0)
                log "OK: $FILE"
                rm -f "$COUNTER"
                DEST="$PROCESSED/$FILE"
                [ -e "$DEST" ] && DEST="$PROCESSED/$(date +%Y%m%d-%H%M%S)_$FILE"
                mv "$INBOX_PATH" "$DEST" && log "Archived: $DEST"
                ;;
            "$EXIT_BUSY")
                log "BUSY: $FILE - GPU is held elsewhere; retrying next poll (not counted)"
                ;;
            "$EXIT_PERMANENT")
                log "FAILED: $FILE (exit $RC) - retrying cannot fix this, setting it aside"
                set_aside "$INBOX_PATH" "$FILE"
                ;;
            *)
                N=$(( $(cat "$COUNTER" 2>/dev/null || echo 0) + 1 ))
                if [ "$N" -ge "$MAX_ATTEMPTS" ]; then
                    log "FAILED: $FILE (exit $RC) - attempt $N/$MAX_ATTEMPTS, giving up"
                    set_aside "$INBOX_PATH" "$FILE"
                else
                    echo "$N" > "$COUNTER"
                    log "FAILED: $FILE (exit $RC) - attempt $N/$MAX_ATTEMPTS, retrying next poll"
                fi
                ;;
        esac
    done < <(find "$INBOX" -maxdepth 1 -type f \
                \( -iname "*.mp3" -o -iname "*.m4a" -o -iname "*.wav" \
                   -o -iname "*.flac" -o -iname "*.ogg" \) \
                ! -name ".*" 2>/dev/null)

    nap "$POLL_INTERVAL"
done
