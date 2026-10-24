"""`supermeet.py kuliah <folder>`: transcribe every lecture in a folder.

    supermeet.py kuliah <folder> [--out DIR] [--dry-run] [--force] [--progress-json FILE]

Exit codes: 0 ok, 1 a meeting failed, 2 bad usage, 75 another run is active,
130 cancelled.
"""
import argparse
import fcntl
import json
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import matkul, output, plan
from .config import LOG_FILE, MIN_PART_SECONDS, OUTPUT_DIR, RUN_LOCK

EXIT_USAGE = 2
EXIT_BUSY = 75
EXIT_CANCELLED = 130

STATUS_LABEL = {
    plan.TODO: "akan diproses",
    plan.DONE: "sudah ada",
    plan.INCOMPLETE: "belum lengkap",
    plan.TOO_SHORT: "terlalu pendek",
    plan.CONFLICT: "konflik",
}
# Summary wording per skip reason, in display order.
SKIP_LABEL = [(plan.DONE, "sudah ada"), (plan.INCOMPLETE, "belum lengkap"),
              (plan.TOO_SHORT, "terlalu pendek"), (plan.CONFLICT, "konflik")]

# Share of a meeting's progress bar given to model loading and to writing; the
# rest is transcription, weighted by audio duration.
LOAD_SHARE, WRITE_SHARE = 0.03, 0.02


class MeetingError(RuntimeError):
    """A meeting that cannot be finished, with a message fit for the user."""


def to_wsl_path(raw: str) -> Path:
    """C:\\Users\\me\\x -> /mnt/c/Users/me/x; anything else is a normal path."""
    raw = raw.strip().strip('"')
    m = re.match(r"^([A-Za-z]):[\\/]*(.*)$", raw)
    if m:
        rest = m.group(2).replace("\\", "/")
        return Path(f"/mnt/{m.group(1).lower()}/{rest}")
    return Path(raw).expanduser()


def fmt_dur(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}"


class Reporter:
    """One place for the three outputs: terminal, log file, JSON event stream."""

    def __init__(self, json_path: Path | None, log_path: Path = LOG_FILE):
        self._json = None
        if json_path:
            self._json = open(json_path, "a", encoding="utf-8")
        self._log = None
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "a", encoding="utf-8")
        except OSError:
            pass
        self._tty = sys.stdout.isatty()
        self._bar_open = False

    def event(self, obj: dict) -> None:
        if self._json:
            self._json.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._json.flush()

    def log(self, line: str) -> None:
        if self._log:
            self._log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
            self._log.flush()

    def say(self, line: str = "") -> None:
        self.end_bar()
        print(line)

    def bar(self, mid: str, stage: str, fraction: float, n: int, count: int) -> None:
        if not self._tty:
            return
        width = 24
        filled = int(width * fraction)
        text = (f"  {mid:<9} {'█' * filled}{'░' * (width - filled)} "
                f"{fraction * 100:3.0f}%  {stage:<10} [{n}/{count}]")
        print("\r" + text, end="", flush=True)
        self._bar_open = True

    def end_bar(self) -> None:
        if self._bar_open:
            print()
            self._bar_open = False

    def close(self) -> None:
        self.end_bar()
        for f in (self._json, self._log):
            if f:
                f.close()


def print_plan(rep: Reporter, scan: plan.ScanResult, folder: Path) -> None:
    if not scan.meetings and not scan.unparsed:
        rep.say("  Nggak ada rekaman audio di folder ini.")
        return
    rep.say(f"  {'ID':<9} {'tanggal':<11} {'durasi':>8}  status")
    for m in scan.meetings:
        status = STATUS_LABEL[m.status]
        if m.note and m.status != plan.TODO:
            status += f" ({m.note})"
        shorts = sum(p.short for p in m.parts)
        if m.status == plan.TODO and shorts:
            status += f" ({shorts} bagian pendek dilewati)"
        rep.say(f"  {m.id:<9} {m.tanggal.isoformat():<11} {fmt_dur(m.duration):>8}  {status}")
    for p in scan.unparsed:
        rep.say(f"  ? {p.relative_to(folder).as_posix()}  (nama tidak dikenali)")
    rep.say()


def check_out_dir(out: Path) -> str | None:
    """Why `out` is unusable, or None.

    Up to two missing levels may be created on a first run, but never the
    folder those hang off: a mistyped --out should fail, not build a stray tree.
    """
    if out.exists():
        if not out.is_dir():
            return f"{out} bukan folder"
        return None if os.access(out, os.W_OK) else f"{out} nggak bisa ditulis"
    anchor = next((p for p in (out.parent, out.parent.parent) if p.is_dir()), None)
    if anchor is None:
        return f"folder output {out} nggak ada, dan {out.parent.parent} juga nggak ada"
    return None if os.access(anchor, os.W_OK) else f"{anchor} nggak bisa ditulis"


def run_meeting(m: plan.Meeting, open_transcriber: Callable, rep: Reporter,
                on_progress: Callable[[str, float], None],
                on_wait: Callable[[dict], None]) -> list[Path]:
    """Transcribe and write one meeting; returns the files written."""
    todo = m.transcribed_parts()
    total_dur = sum(p.duration for p in todo) or 1.0
    on_progress("load", 0.0)
    segments: dict[Path, list[dict]] = {}
    done_dur = 0.0
    with open_transcriber(m.kode, on_wait) as tr:
        for part in todo:
            def cb(stage: str, frac: float, base=done_dur, dur=part.duration):
                if stage == "load":
                    on_progress("load", LOAD_SHARE * frac)
                else:
                    on_progress("transcribe", LOAD_SHARE + (1 - LOAD_SHARE - WRITE_SHARE)
                                * (base + dur * min(max(frac, 0.0), 1.0)) / total_dur)
            segments[part.path] = tr.part(part.path, cb)
            done_dur += part.duration
    if not any(segments.values()):
        raise MeetingError("nggak ada ucapan yang kebaca (rekaman hening, atau "
                           "semuanya kebuang sebagai halusinasi)")
    on_progress("write", 1 - WRITE_SHARE)
    files = output.write_meeting(m, segments, matkul.get(m.kode).nama)
    on_progress("write", 1.0)
    return files


def process(scan: plan.ScanResult, out: Path, rep: Reporter, open_transcriber: Callable,
            force: bool = False) -> dict:
    """Run every todo meeting; one failing meeting never stops the rest."""
    if force:
        for m in scan.meetings:
            if m.status == plan.DONE:
                m.status, m.note = plan.TODO, ""
    todo = [m for m in scan.meetings if m.status == plan.TODO]
    result = {"processed": 0, "failed": [], "skipped": [m for m in scan.meetings
                                                       if m.status != plan.TODO]}
    for i, m in enumerate(todo, 1):
        rep.event({"type": "meeting_start", "id": m.id, "index": i, "count": len(todo)})
        rep.log(f"start {m.id} ({fmt_dur(m.duration)}, {len(m.parts)} file)")
        started = time.monotonic()

        last = {"stage": None, "fraction": -1.0}

        def on_progress(stage: str, fraction: float, m=m, i=i, last=last):
            # whisperx reports per chunk; hundreds of events add nothing over 0.5% steps.
            if stage == last["stage"] and abs(fraction - last["fraction"]) < 0.005 and fraction < 1:
                return
            last.update(stage=stage, fraction=fraction)
            rep.event({"type": "progress", "id": m.id,
                       "fraction": round(fraction, 4), "stage": stage})
            rep.bar(m.id, stage, fraction, i, len(todo))

        def on_wait(holder: dict | None, m=m):
            who = (holder or {}).get("name", "proses lain")
            rep.event({"type": "waiting_gpu", "id": m.id, "holder": who})
            rep.say(f"  {m.id}: GPU lagi dipakai {who}, nunggu...")
            rep.log(f"waiting_gpu {m.id} holder={who}")

        try:
            files = run_meeting(m, open_transcriber, rep, on_progress, on_wait)
        except Exception as e:  # noqa: BLE001 -- isolate per meeting, report below
            msg = str(e) or type(e).__name__
            if not isinstance(e, MeetingError):
                msg = f"{type(e).__name__}: {msg}"
            rep.say(f"  ✗ {m.id}: {msg}")
            rep.log(f"failed {m.id}: {msg}")
            rep.event({"type": "meeting_failed", "id": m.id, "error": msg})
            result["failed"].append((m.id, msg))
            continue
        secs = time.monotonic() - started
        rep.end_bar()
        rep.say(f"  ✓ {m.id}  {len(files)} file, {secs:.0f} detik")
        rep.log(f"done {m.id} in {secs:.1f}s -> " + ", ".join(map(str, files)))
        rep.event({"type": "meeting_done", "id": m.id,
                   "outputs": [str(p) for p in files], "seconds": round(secs, 1)})
        result["processed"] += 1
    return result


def print_summary(rep: Reporter, scan: plan.ScanResult, res: dict) -> None:
    rep.say()
    rep.say(f"  Selesai: {res['processed']} pertemuan diproses")
    skipped = []
    for status, label in SKIP_LABEL:
        n = sum(m.status == status for m in res["skipped"])
        if n:
            skipped.append(f"{n} {label}")
    if scan.unparsed:
        skipped.append(f"{len(scan.unparsed)} nama tidak dikenali")
    if skipped:
        rep.say("  Dilewati: " + ", ".join(skipped))
    if res["failed"]:
        rep.say(f"  Gagal: {len(res['failed'])}")
        for mid, err in res["failed"]:
            rep.say(f"    {mid}: {err}")
    rep.log(f"summary processed={res['processed']} skipped={len(res['skipped'])} "
            f"failed={len(res['failed'])}")


def main(argv: list[str], open_transcriber: Callable | None = None) -> int:
    ap = argparse.ArgumentParser(prog="supermeet.py kuliah", description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help=f"folder output (default {OUTPUT_DIR})")
    ap.add_argument("--dry-run", action="store_true", help="cuma tampilkan rencana")
    ap.add_argument("--force", action="store_true", help="proses ulang yang udah ada")
    ap.add_argument("--progress-json", default=None, metavar="FILE")
    try:
        args = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else 0

    def fail(msg: str, code: int) -> int:
        print(msg, file=sys.stderr)
        # The upload page only sees the exit code otherwise, which cannot say
        # *which* folder is missing.
        if args.progress_json:
            try:
                with open(args.progress_json, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "error", "message": msg},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
        return code

    folder = to_wsl_path(args.folder)
    if not folder.is_dir():
        return fail(f"Folder nggak ketemu: {folder}", EXIT_USAGE)
    out = to_wsl_path(args.out) if args.out else OUTPUT_DIR
    if not args.dry_run:
        problem = check_out_dir(out)
        if problem:
            return fail(problem, EXIT_USAGE)

    lock = None
    if not args.dry_run:
        RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock = open(RUN_LOCK, "a+")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return fail("Ada proses kuliah lain yang lagi jalan. Tunggu dulu.", EXIT_BUSY)

    try:
        rep = Reporter(Path(args.progress_json) if args.progress_json else None)
    except OSError as e:
        print(f"Nggak bisa nulis progress file: {e}", file=sys.stderr)
        return EXIT_USAGE

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    try:
        scan = plan.scan(folder, out)
        if args.force:
            for m in scan.meetings:
                if m.status == plan.DONE:
                    m.status, m.note = plan.TODO, ""
        rep.event({"type": "plan",
                   "meetings": [plan.meeting_to_dict(m, folder) for m in scan.meetings],
                   "unparsed": [p.relative_to(folder).as_posix() for p in scan.unparsed]})
        print_plan(rep, scan, folder)
        if args.dry_run:
            return 0

        rep.log(f"run folder={folder} out={out} force={args.force}")
        if any(m.status == plan.TODO for m in scan.meetings):
            out.mkdir(parents=True, exist_ok=True)   # depth bounded by check_out_dir
        if open_transcriber is None:
            from .transcribe import MeetingTranscriber as open_transcriber
        res = process(scan, out, rep, open_transcriber)
        print_summary(rep, scan, res)
        rep.event({"type": "summary", "processed": res["processed"],
                   "skipped": len(res["skipped"]), "failed": len(res["failed"])})
        return 1 if res["failed"] else 0
    except KeyboardInterrupt:
        rep.say("\n  Dibatalin.")
        rep.log("cancelled")
        rep.event({"type": "cancelled"})
        return EXIT_CANCELLED
    finally:
        rep.close()
        if lock:
            lock.close()
