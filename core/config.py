#!/usr/bin/env python3
"""Shared paths and settings for the transcriber.

Cloud provider is a deployment detail, not code. Override any of these with
env vars (or the project's .env file, which is loaded here); otherwise the
first cloud root that actually exists on disk wins, so a half-finished
migration never silently writes to the wrong place. With no cloud folder at all
the root is ./whisper inside the repo, so a fresh clone works out of the box.
"""
import os
import sys
from pathlib import Path

# `python core/config.py` (batch/watcher.sh) puts core/, not the repo, on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# First, before anything reads os.environ: this module is imported by every
# entry point, and batch/watcher.sh runs it as a script, so it is the one place
# that makes .env work everywhere. Real environment variables still win.
from core import env
env.load()

# Repo root, two levels up -- this file lives in core/. Everything below is
# derived from it, so the wrong level silently splits the transcript cache.
PROJECT_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_DIR / "transcripts"
LOCAL_AUDIO = PROJECT_DIR / "audio"
# Archive processed originals on local disk, never in the synced folder --
# otherwise every transcribed recording sits in cloud storage forever.
PROCESSED_DIR = LOCAL_AUDIO / "processed"
# Where the watcher sets aside files that retrying cannot fix, so they stop
# looping through the inbox. Local for the same reason as PROCESSED_DIR.
FAILED_DIR = LOCAL_AUDIO / "failed"

# Name the summaries address the user by ("Lo asisten yang bantu <name> ...").
USER_NAME = os.environ.get("SUPERMEET_USER_NAME") or "user"

# Used when no cloud folder exists: everything stays inside the repo.
LOCAL_ROOT = PROJECT_DIR / "whisper"

# Windows cloud folders as WSL sees them, checked in order and sorted within a
# pattern so the pick is deterministic. Google Drive must be in "Mirror files"
# mode: the default "Stream files" mode is a virtual drive that WSL cannot read.
CLOUD_GLOBS = [
    "/mnt/c/Users/*/My Drive/whisper",      # Google Drive (mirror)
    "/mnt/c/Users/*/iCloudDrive/whisper",   # iCloud for Windows
]

# Cloud folders on a Mac, relative to the home directory. Checked after the WSL
# ones (which simply match nothing on a Mac). Google Drive for desktop keeps its
# folder under ~/Library/CloudStorage/GoogleDrive-<account>; the old client used
# ~/Google Drive. Use "Mirror files" mode there too, so the audio is really on disk.
HOME_CLOUD_GLOBS = [
    "Library/CloudStorage/GoogleDrive-*/My Drive/whisper",    # Google Drive for desktop
    "Google Drive/My Drive/whisper",                          # Google Drive, older client
    "Library/Mobile Documents/com~apple~CloudDocs/whisper",   # iCloud Drive
]


def _first_dir(root: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern.lstrip("/"))):
            if candidate.is_dir():
                return candidate
    return None


def detect_cloud_root(home: Path | None = None, wsl_globs: list[str] | None = None,
                      home_globs: list[str] | None = None) -> Path | None:
    """First existing cloud folder: WSL's /mnt/c globs, then the Mac ones under home.

    The arguments exist so tests can point the search at a fake tree; callers
    normally pass nothing.
    """
    found = _first_dir(Path("/"), CLOUD_GLOBS if wsl_globs is None else wsl_globs)
    if found:
        return found
    if home is None:
        try:
            home = Path.home()
        except RuntimeError:    # no resolvable home directory
            return None
    return _first_dir(home, HOME_CLOUD_GLOBS if home_globs is None else home_globs)


def cloud_root() -> Path:
    override = os.environ.get("TRANSCRIBE_CLOUD_ROOT")
    if override:
        return Path(override)
    return detect_cloud_root() or LOCAL_ROOT


CLOUD_ROOT = cloud_root()
INBOX = Path(os.environ.get("TRANSCRIBE_INBOX", CLOUD_ROOT / "transcribe-inbox"))
OUTPUT_DIR = Path(os.environ.get(
    "TRANSCRIBE_OUTPUT", CLOUD_ROOT / "transcribe-output" / "summaries"))
# Transcript PDFs sit next to summaries/, not inside it, so each folder can be
# browsed on its own -- and a transcribe-only run writes nothing to summaries/.
TRANSCRIPT_OUTPUT_DIR = Path(os.environ.get(
    "TRANSCRIBE_TRANSCRIPT_OUTPUT", CLOUD_ROOT / "transcribe-output" / "transcripts"))

# The local fallback is ours, so create it -- batch/watcher.sh refuses to start
# without an inbox. A real cloud root or an override is never created here: a
# missing one means the sync client is not running, and mkdir would hide that.
if CLOUD_ROOT == LOCAL_ROOT and not os.environ.get("TRANSCRIBE_CLOUD_ROOT"):
    for _d in (INBOX, OUTPUT_DIR, TRANSCRIPT_OUTPUT_DIR):
        if _d.is_relative_to(LOCAL_ROOT):
            _d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else None
    values = {"INBOX": INBOX, "OUTPUT_DIR": OUTPUT_DIR,
              "TRANSCRIPT_OUTPUT_DIR": TRANSCRIPT_OUTPUT_DIR, "CLOUD_ROOT": CLOUD_ROOT,
              "PROCESSED_DIR": PROCESSED_DIR, "FAILED_DIR": FAILED_DIR,
              "LOCAL_AUDIO": LOCAL_AUDIO}
    if name:
        print(values[name])
    else:
        for k, v in values.items():
            print(f"{k}={v}  {'[exists]' if Path(v).is_dir() else '[missing]'}")
