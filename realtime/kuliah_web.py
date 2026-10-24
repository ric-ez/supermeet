"""Lecture upload page: drop recordings, press one button, download transcripts.

Uploads land in one shared pool (UPLOAD_DIR) that `supermeet.py kuliah` scans as
a whole. The run itself is that CLI in a subprocess; this module only spawns it
and reduces its --progress-json file into something the page can show.
"""
import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import signal
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from kuliah import config as kcfg
from kuliah import matkul, names
from kuliah.plan import meeting_to_dict, scan

from . import config, stt

log = logging.getLogger("rtsm.kuliah")
router = APIRouter()

MAX_ZIP_BYTES = 20 * 1024 ** 3
_RESERVED = (".run", ".incoming")

_proc: asyncio.subprocess.Process | None = None
_exit_code: int | None = None
_watch_task: asyncio.Task | None = None
_stopped_for_live = False

# Set by app.py: is a live meeting recording right now? The idle server keeps
# its streaming model warm, and with it the GPU lease that a lecture run needs.
live_active = lambda: False  # noqa: E731


def _upload_dir() -> Path:
    return kcfg.UPLOAD_DIR


def _running() -> bool:
    return _proc is not None and _proc.returncode is None


# --- placing files ----------------------------------------------------------
def _clean_name(raw: str) -> str:
    """Basename only; '' when nothing safe is left."""
    raw = raw.replace("\\", "/")
    base = PurePosixPath(raw).name
    base = "".join(c for c in base if c >= " " and c != "\x7f").strip()
    return "" if base in ("", ".", "..") else base


def _is_audio(name: str) -> bool:
    return Path(name).suffix.lower() in kcfg.AUDIO_EXTS


def _place(src: Path, name: str, out: dict) -> None:
    """Move `src` (a finished temp file) into the pool under the right name."""
    ext = Path(name).suffix
    parsed = names.parse(name)
    if parsed:
        target = _upload_dir() / parsed.kode / names.canonical(parsed, ext)
        bucket = "added"
    else:
        target = kcfg.UNNAMED_DIR / name
        bucket = "unnamed"
    target.parent.mkdir(parents=True, exist_ok=True)
    rel = target.relative_to(_upload_dir()).as_posix()
    if target.exists():
        if target.stat().st_size == src.stat().st_size:
            out["duplicates"].append(rel)
        else:
            out["conflicts"].append(rel)
        src.unlink(missing_ok=True)
        return
    shutil.move(str(src), str(target))
    out[bucket].append(rel)


def _empty_result() -> dict:
    return {"added": [], "unnamed": [], "duplicates": [], "conflicts": [], "ignored": []}


def _zip_entry_ok(info: zipfile.ZipInfo) -> str | None:
    """Basename to extract, or None to ignore the entry."""
    if info.is_dir():
        return None
    # Only regular files (reject symlinks etc. via unix mode bits).
    mode = info.external_attr >> 16
    if mode and (mode & 0o170000) not in (0, 0o100000):
        return None
    parts = [p for p in info.filename.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts or "__MACOSX" in parts:
        return None
    if any(p.startswith(("_", ".")) for p in parts):
        return None
    base = _clean_name(parts[-1])
    if not base or not _is_audio(base):
        return None
    return base


def _extract_zip(zpath: Path, tmpdir: Path, out: dict) -> None:
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        out["ignored"].append("zip rusak")
        return
    with zf:
        infos = zf.infolist()
        if sum(i.file_size for i in infos) > MAX_ZIP_BYTES:
            raise HTTPException(413, "Zip terlalu besar (maksimal 20 GB setelah diekstrak).")
        for info in infos:
            base = _zip_entry_ok(info)
            if base is None:
                if not info.is_dir():
                    out["ignored"].append(info.filename)
                continue
            fd, tmp = tempfile.mkstemp(dir=tmpdir, suffix=".part")
            tmp = Path(tmp)
            try:
                with os.fdopen(fd, "wb") as dst, zf.open(info) as srcf:
                    shutil.copyfileobj(srcf, dst, 1024 * 1024)
                _place(tmp, base, out)
            finally:
                tmp.unlink(missing_ok=True)


def _process_upload(tmp: Path, name: str) -> dict:
    out = _empty_result()
    ext = Path(name).suffix.lower()
    if ext == ".zip":
        try:
            _extract_zip(tmp, tmp.parent, out)
        finally:
            tmp.unlink(missing_ok=True)
    elif _is_audio(name) and not name.startswith("."):
        _place(tmp, name, out)
    else:
        tmp.unlink(missing_ok=True)
        out["ignored"].append(name)
    return out


@router.get("/kuliah")
async def page():
    return FileResponse(config.WEB_DIR / "kuliah.html")


@router.put("/kuliah/api/upload")
async def upload(request: Request, name: str = ""):
    base = _clean_name(name)
    if not base:
        raise HTTPException(400, "Nama file kosong.")
    incoming = _upload_dir() / ".incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    fd, tmp_s = tempfile.mkstemp(dir=incoming, suffix=".part")
    tmp = Path(tmp_s)
    try:
        with os.fdopen(fd, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        return await asyncio.to_thread(_process_upload, tmp, base)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# --- naming ------------------------------------------------------------------
@router.post("/kuliah/api/name")
async def give_name(request: Request):
    try:
        body = await request.json()
        file = _clean_name(str(body.get("file", "")))
        kode = str(body.get("kode", "")).strip()
        pertemuan = int(body.get("pertemuan"))
        tanggal = date.fromisoformat(str(body.get("tanggal", "")))
        bagian = body.get("bagian") or None
        total = body.get("total") or None
        bagian = int(bagian) if bagian is not None else None
        total = int(total) if total is not None else None
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Isian belum bener: cek kode, pertemuan, dan tanggal.")
    if (bagian is None) != (total is None):
        raise HTTPException(400, "Bagian dan total harus diisi dua-duanya, atau dikosongin dua-duanya.")
    src = kcfg.UNNAMED_DIR / file
    if not file or not src.is_file():
        raise HTTPException(400, "File-nya nggak ketemu.")
    ext = src.suffix
    try:
        parsed = names.Parsed(kode.upper(), pertemuan, tanggal, bagian, total)
        cname = names.canonical(parsed, ext)
    except Exception:
        raise HTTPException(400, "Isian belum bener.")
    back = names.parse(cname)
    if back != parsed or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,9}", kode):
        raise HTTPException(400, "Nama nggak valid: kode 1-10 huruf/angka, pertemuan 1-99, bagian harus 1..total.")
    target = _upload_dir() / back.kode / cname
    if target.exists():
        raise HTTPException(400, f"Sudah ada file bernama {cname}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(target))
    return {"file": target.relative_to(_upload_dir()).as_posix()}


# --- state & run -------------------------------------------------------------
def _progress_file() -> Path:
    return _upload_dir() / ".run" / "progress.jsonl"


def _message_for(code: int | None) -> str:
    return {
        2: "Gagal mulai: folder upload atau folder hasil bermasalah. Detailnya di logs/kuliah-web.log.",
        75: "Ada proses kuliah lain yang lagi jalan (mungkin dari terminal). Tunggu selesai dulu.",
        1: "Ada pertemuan yang gagal diproses. Lihat detailnya di bawah.",
        130: "Dibatalkan.",
        -signal.SIGTERM: "Dibatalkan.",
    }.get(code, "") if code is not None else ""


def _reduce(lines: list[str], active: bool, exit_code: int | None) -> dict:
    current = None
    waiting = None
    done, failed = [], []
    summary = None
    cancelled = False
    error = None
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue  # partially written last line
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "meeting_start":
            current = {"id": ev.get("id"), "index": ev.get("index"), "count": ev.get("count"),
                       "fraction": 0.0, "stage": None}
            waiting = None
        elif t == "waiting_gpu":
            waiting = ev.get("holder") or "?"
        elif t == "progress":
            if current is None or current["id"] != ev.get("id"):
                current = {"id": ev.get("id"), "index": None, "count": None,
                           "fraction": 0.0, "stage": None}
            current["fraction"] = ev.get("fraction", 0.0)
            current["stage"] = ev.get("stage")
            waiting = None
        elif t == "meeting_done":
            done.append({"id": ev.get("id"), "outputs": ev.get("outputs", []),
                         "seconds": ev.get("seconds")})
            current = waiting = None
        elif t == "meeting_failed":
            failed.append({"id": ev.get("id"), "error": ev.get("error", "")})
            current = waiting = None
        elif t == "summary":
            summary = {k: v for k, v in ev.items() if k != "type"}
        elif t == "cancelled":
            cancelled = True
        elif t == "error":
            error = ev.get("message")
    if not active:
        current = waiting = None
    if exit_code in (130, -signal.SIGTERM):
        cancelled = True
    message = "" if active else (error or _message_for(exit_code))
    if cancelled and not active and _stopped_for_live:
        message = ("Dihentikan karena ada rekaman live. Pertemuan yang udah selesai aman; "
                   "tekan Proses lagi setelah rapatnya beres buat lanjut.")
    return {"active": active, "exit_code": exit_code, "current": current,
            "waiting_gpu": waiting, "done": done, "failed": failed,
            "summary": summary, "cancelled": cancelled, "message": message}


def _read_run() -> dict:
    p = _progress_file()
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    return _reduce(lines, _running(), _exit_code)


@router.get("/kuliah/api/run")
async def run_status():
    return await asyncio.to_thread(_read_run)


def _unnamed_list() -> list[dict]:
    d = kcfg.UNNAMED_DIR
    if not d.is_dir():
        return []
    return [{"file": p.name, "size": p.stat().st_size}
            for p in sorted(d.iterdir()) if p.is_file() and not p.name.startswith(".")]


_TXT = re.compile(r"\[P(\d+)\] Transkrip\.txt")
_SRT = re.compile(r"\[P(\d+)\] Transkrip(?: \(bagian (\d+)\))?\.srt")


def _outputs() -> list[dict]:
    """Every finished transcript on disk, per course.

    Read from the output folder, not from the upload pool: the pool copies get
    cleaned up, and this page is the only way to fetch the results.
    """
    root = kcfg.OUTPUT_DIR
    if not root.is_dir():
        return []
    try:
        names_by_kode = {k: m.nama for k, m in matkul.load().items()}
    except Exception:
        names_by_kode = {}
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        files = {}
        for f in d.iterdir():
            if m := _TXT.fullmatch(f.name):
                files.setdefault(int(m[1]), {"txt": None, "srt": []})["txt"] = f
            elif m := _SRT.fullmatch(f.name):
                files.setdefault(int(m[1]), {"txt": None, "srt": []})["srt"].append(
                    (int(m[2] or 0), f))
        for p, fs in sorted(files.items()):
            if fs["txt"] is None:
                continue   # SRTs without their TXT: a run that did not finish
            out.append({
                "kode": d.name, "nama": names_by_kode.get(d.name, ""), "pertemuan": p,
                "txt": str(fs["txt"]), "srt": [str(f) for _, f in sorted(fs["srt"])],
                "mtime": fs["txt"].stat().st_mtime,
            })
    return out


def _state() -> dict:
    result = scan(_upload_dir(), kcfg.OUTPUT_DIR)
    try:
        mk = [{"kode": k, "nama": m.nama} for k, m in sorted(matkul.load().items())]
    except Exception:
        log.exception("matkul.load failed")
        mk = []
    return {
        "meetings": [meeting_to_dict(m, _upload_dir()) for m in result.meetings],
        "unnamed": _unnamed_list(),
        "outputs": _outputs(),
        "matkul": mk,
        "run": _read_run(),
        "output_dir": str(kcfg.OUTPUT_DIR),
        "upload_dir": str(_upload_dir()),
    }


@router.get("/kuliah/api/state")
async def state():
    return await asyncio.to_thread(_state)


async def _watch(proc: asyncio.subprocess.Process):
    global _exit_code
    code = await proc.wait()
    _exit_code = code
    log.info("kuliah run exited with %s", code)
    if not live_active():
        # Give Record its instant start back.
        try:
            await stt.warm_up()
        except Exception:
            log.exception("re-warming the streaming model failed")


def running() -> bool:
    return _running()


async def stop_for_live() -> None:
    """A live meeting outranks a lecture run: stop the run so the meeting gets
    the GPU now rather than after the lecture in progress. Finished parts are
    cached, so pressing Proses again picks up where it stopped."""
    global _stopped_for_live
    if not _running():
        return
    _stopped_for_live = True
    with contextlib.suppress(ProcessLookupError):
        _proc.send_signal(signal.SIGTERM)
    await _proc.wait()   # its GPU lease dies with it


@router.post("/kuliah/api/run")
async def start_run(request: Request):
    global _proc, _exit_code, _watch_task, _stopped_for_live
    if _running():
        raise HTTPException(409, "Masih ada proses yang jalan.")
    if live_active():
        raise HTTPException(409, "Lagi ada rekaman live. Proses kuliah setelah rapatnya selesai.")
    _stopped_for_live = False
    # The idle server holds the GPU lease along with its warm streaming model;
    # without this the run would wait on it forever. _watch reloads it after.
    await stt.release_model()
    force = False
    with contextlib.suppress(Exception):
        body = await request.json()
        force = bool(body.get("force"))
    pf = _progress_file()
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text("", encoding="utf-8")
    cmd = [sys.executable, str(config.CLI), "kuliah", str(_upload_dir()),
           "--progress-json", str(pf)]
    if force:
        cmd.append("--force")
    logdir = config.PROJECT_DIR / "logs"
    logdir.mkdir(exist_ok=True)
    with open(logdir / "kuliah-web.log", "ab") as logf:
        _exit_code = None
        _proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=str(config.PROJECT_DIR), stdin=asyncio.subprocess.DEVNULL,
            stdout=logf, stderr=logf)
    _watch_task = asyncio.create_task(_watch(_proc))
    return await asyncio.to_thread(_read_run)


@router.post("/kuliah/api/cancel")
async def cancel():
    if _running():
        with contextlib.suppress(ProcessLookupError):
            _proc.send_signal(signal.SIGTERM)
    return await asyncio.to_thread(_read_run)


# --- deleting ----------------------------------------------------------------
def _resolve_pool_file(rel: str) -> Path:
    root = _upload_dir().resolve()
    if not rel or rel.startswith("/"):
        raise HTTPException(400, "Path nggak valid.")
    p = (root / rel).resolve()
    if not p.is_relative_to(root) or p == root:
        raise HTTPException(400, "Path nggak valid.")
    if p.relative_to(root).parts[0] in _RESERVED:
        raise HTTPException(400, "Path nggak valid.")
    return p


def _rm(p: Path) -> bool:
    if not p.is_file():
        return False
    p.unlink()
    root = _upload_dir().resolve()
    with contextlib.suppress(OSError):
        if p.parent != root:
            p.parent.rmdir()  # only succeeds when empty
    return True


@router.delete("/kuliah/api/upload")
async def delete_upload(file: str = ""):
    if _running():
        raise HTTPException(409, "Lagi ada proses jalan, tunggu selesai dulu.")
    p = _resolve_pool_file(file)
    if not await asyncio.to_thread(_rm, p):
        raise HTTPException(404, "File nggak ketemu.")
    return {"deleted": file}


@router.post("/kuliah/api/cleanup")
async def cleanup():
    if _running():
        raise HTTPException(409, "Lagi ada proses jalan, tunggu selesai dulu.")

    def work():
        n = 0
        for m in meeting_dicts():
            if m["status"] != "done":
                continue
            for part in m["parts"]:
                try:
                    if _rm(_resolve_pool_file(part["file"])):
                        n += 1
                except HTTPException:
                    pass
        return n

    def meeting_dicts():
        r = scan(_upload_dir(), kcfg.OUTPUT_DIR)
        return [meeting_to_dict(m, _upload_dir()) for m in r.meetings]

    return {"deleted": await asyncio.to_thread(work)}


# --- download ----------------------------------------------------------------
@router.get("/kuliah/api/download")
async def download(path: str = ""):
    try:
        root = kcfg.OUTPUT_DIR.resolve()
        p = Path(path).resolve()
    except (OSError, ValueError):
        raise HTTPException(404)
    if not path or not p.is_relative_to(root) or not p.is_file():
        raise HTTPException(404)
    ext = p.suffix.lower()
    mt = {".txt": "text/plain; charset=utf-8",
          ".srt": "application/x-subrip; charset=utf-8",
          ".md": "text/markdown; charset=utf-8"}.get(ext)
    return FileResponse(p, media_type=mt, filename=p.name)


@router.get("/kuliah/api/download-zip")
async def download_zip(kode: str = ""):
    """Every transcript (or one course's) in one zip, laid out as KODE/file."""
    if kode and not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,9}", kode):
        raise HTTPException(404)

    def build() -> Path | None:
        items = [o for o in _outputs() if not kode or o["kode"] == kode.upper()]
        if not items:
            return None
        fd, tmp = tempfile.mkstemp(prefix="kuliah-", suffix=".zip")
        os.close(fd)
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for o in items:
                for f in (o["txt"], *o["srt"]):
                    zf.write(f, f"{o['kode']}/{Path(f).name}")
        return Path(tmp)

    path = await asyncio.to_thread(build)
    if path is None:
        raise HTTPException(404, "Belum ada transkrip.")
    name = f"Transkrip {kode.upper()}.zip" if kode else "Transkrip kuliah.zip"
    return FileResponse(path, media_type="application/zip", filename=name,
                        background=BackgroundTask(path.unlink, missing_ok=True))
