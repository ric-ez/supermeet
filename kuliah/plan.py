"""What is in a recordings folder, grouped into meetings, and what to do with each.

Cheap on purpose (no torch/whisperx, ffprobe results memoized): the upload page
calls scan() every few seconds, and the CLI uses the same result as its plan.
"""
import json
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import matkul, names
from .config import AUDIO_EXTS, MIN_PART_SECONDS, OUTPUT_DIR

TODO, DONE, INCOMPLETE, TOO_SHORT, CONFLICT = (
    "todo", "done", "incomplete", "too_short", "conflict")


@dataclass
class Part:
    path: Path
    bagian: int | None
    duration: float
    short: bool          # too short to transcribe; its duration still shifts later parts


@dataclass
class Meeting:
    kode: str
    pertemuan: int
    tanggal: date
    parts: list[Part]    # ordered by bagian
    status: str
    note: str
    total: int
    missing: list[int]
    out_dir: Path        # root of the output tree (the per-KODE folder is below it)

    @property
    def id(self) -> str:
        return f"{self.kode}-P{self.pertemuan:02d}"

    @property
    def multi(self) -> bool:
        return any(p.bagian is not None for p in self.parts)

    @property
    def duration(self) -> float:
        return sum(p.duration for p in self.parts)

    def txt_path(self) -> Path:
        return self.out_dir / self.kode / f"[P{self.pertemuan:02d}] Transkrip.txt"

    def srt_path(self, part: Part) -> Path:
        base = f"[P{self.pertemuan:02d}] Transkrip"
        if self.multi:
            base += f" (bagian {part.bagian})"
        return self.out_dir / self.kode / f"{base}.srt"

    def transcribed_parts(self) -> list[Part]:
        return [p for p in self.parts if not p.short]

    def planned_outputs(self) -> list[Path]:
        return [self.txt_path(), *(self.srt_path(p) for p in self.transcribed_parts())]


@dataclass
class ScanResult:
    meetings: list[Meeting] = field(default_factory=list)
    unparsed: list[Path] = field(default_factory=list)


_DURATIONS: dict[tuple, float] = {}


def probe_duration(path: Path) -> float:
    """Seconds of audio via ffprobe; 0.0 if unreadable (then treated as too short)."""
    try:
        st = path.stat()
    except OSError:
        return 0.0
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key in _DURATIONS:
        return _DURATIONS[key]
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        dur = float(json.loads(out.stdout)["format"]["duration"])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        # Not cached: a file still being copied may become readable a moment later.
        return 0.0
    _DURATIONS[key] = dur
    return dur


def _audio_files(folder: Path):
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if any(c.startswith(("_", ".")) for c in rel.parts):
            continue
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            yield p


def _judge(parsed: list[tuple[Path, names.Parsed]], durations: dict[Path, float],
           out_dir: Path) -> Meeting:
    first = parsed[0][1]
    kode, pertemuan = first.kode, first.pertemuan
    items = sorted(parsed, key=lambda t: (t[1].bagian or 0, str(t[0])))
    parts = [Part(p, n.bagian, durations[p], durations[p] < MIN_PART_SECONDS)
             for p, n in items]
    tanggal = min(n.tanggal for _, n in items)
    bagians = [n.bagian for _, n in items]
    totals = {n.total for _, n in items if n.total is not None}
    total = max(totals) if totals else 1
    m = Meeting(kode, pertemuan, tanggal, parts, TODO, "", total, [], out_dir)

    conflict = None
    if len({n.tanggal for _, n in items}) > 1:
        conflict = "tanggal di nama file beda-beda"
    elif any(b is None for b in bagians) and any(b is not None for b in bagians):
        conflict = "campur file utuh dan file 'bagian'"
    elif len({b for b in bagians if b is not None}) != len([b for b in bagians if b is not None]):
        conflict = "ada nomor bagian yang dobel"
    elif len(totals) > 1:
        conflict = "jumlah bagian di nama file nggak konsisten"
    if conflict:
        m.status, m.note = CONFLICT, conflict
        return m

    if m.txt_path().exists():
        m.status, m.note = DONE, "transkrip udah ada"
        return m

    present = {b for b in bagians if b is not None}
    if present:
        m.missing = [b for b in range(1, total + 1) if b not in present]
    if m.missing:
        nums = ", ".join(map(str, m.missing))
        m.status, m.note = INCOMPLETE, f"bagian {nums} dari {total} belum ada"
        return m

    if all(p.short for p in parts):
        m.status = TOO_SHORT
        m.note = f"semua rekaman < {MIN_PART_SECONDS:.0f} detik (kemungkinan kepencet)"
    return m


def scan(folder: Path, out_dir: Path = OUTPUT_DIR) -> ScanResult:
    folder = Path(folder)
    groups: dict[tuple[str, int], list[tuple[Path, names.Parsed]]] = {}
    unparsed: list[Path] = []
    for p in _audio_files(folder):
        parsed = names.parse(p.name)
        if parsed is None:
            unparsed.append(p)
        else:
            groups.setdefault((parsed.kode, parsed.pertemuan), []).append((p, parsed))

    durations: dict[Path, float] = {}
    for items in groups.values():
        for p, _ in items:
            durations[p] = probe_duration(p)
    meetings = [_judge(groups[k], durations, Path(out_dir)) for k in sorted(groups)]
    return ScanResult(meetings, unparsed)


def meeting_to_dict(m: Meeting, root: Path) -> dict:
    def rel(p: Path) -> str:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            return p.as_posix()

    if m.status == DONE:
        outputs = [p for p in m.planned_outputs() if p.exists()]
    elif m.status == TODO:
        outputs = m.planned_outputs()
    else:
        outputs = []
    return {
        "id": m.id,
        "kode": m.kode,
        "nama": matkul.get(m.kode).nama,
        "pertemuan": m.pertemuan,
        "tanggal": m.tanggal.isoformat(),
        "status": m.status,
        "note": m.note,
        "total": m.total,
        "missing": list(m.missing),
        "duration": round(m.duration, 1),
        "parts": [{"file": rel(p.path), "bagian": p.bagian,
                   "duration": round(p.duration, 1), "short": p.short}
                  for p in m.parts],
        "outputs": [str(p) for p in outputs],
    }
