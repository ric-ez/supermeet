"""Render a meeting's transcript as TXT (whole meeting) and SRT (per audio file).

Pure functions: segments in, text out. Nothing here touches the GPU, so a cached
transcript can be re-rendered any time the layout changes.
"""
import os
import tempfile
from datetime import date, datetime
from pathlib import Path

from .config import COMPUTE_TYPE, MODEL
from .plan import Meeting, Part

# Hardcoded rather than locale-based: the box's locale is not Indonesian and
# the names must not depend on it.
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
         "Agustus", "September", "Oktober", "November", "Desember"]

LONG_MEETING = 3600.0      # from here on, timestamps carry an hour field
PARAGRAPH_MAX = 60.0       # a paragraph never spans more than this before splitting
PAUSE_SPLIT = 3.0          # a pause this long ends a paragraph...
PAUSE_MIN_LEN = 20.0       # ...but only once it has this much text behind it
RULE = "=" * 64


def format_tanggal(d: date) -> str:
    return f"{HARI[d.weekday()]}, {d.day} {BULAN[d.month - 1]} {d.year}"


def fmt_clock(seconds: float, hours: bool) -> str:
    """0:05:12 / 05:12 (zero padded minutes) -- used inside [brackets]."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if hours else f"{h * 60 + m:02d}:{sec:02d}"


def fmt_duration(seconds: float, hours: bool) -> str:
    """1:32:10 or 45:10 for the Durasi line."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if hours else f"{h * 60 + m}:{sec:02d}"


def paragraphs(segments: list[dict]) -> list[tuple[float, str]]:
    """Group segments into (start, text) paragraphs, times as given."""
    out: list[tuple[float, str]] = []
    cur_start = prev_end = 0.0
    words: list[str] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if words:
            long_enough = prev_end - cur_start >= PAUSE_MIN_LEN
            if (seg["start"] - cur_start >= PARAGRAPH_MAX
                    or (long_enough and seg["start"] - prev_end >= PAUSE_SPLIT)):
                out.append((cur_start, " ".join(words)))
                words = []
        if not words:
            cur_start = seg["start"]
        words.append(text)
        prev_end = seg["end"]
    if words:
        out.append((cur_start, " ".join(words)))
    return out


def part_offsets(m: Meeting) -> list[float]:
    """Start of each part on the meeting's clock; short parts still take room."""
    offsets, acc = [], 0.0
    for p in m.parts:
        offsets.append(acc)
        acc += p.duration
    return offsets


def render_txt(m: Meeting, segments: dict[Path, list[dict]], nama: str = "",
               now: datetime | None = None) -> str:
    """`segments` maps each transcribed part's path to its own-clock segments."""
    from .matkul import Matkul
    now = now or datetime.now()
    hours = m.duration >= LONG_MEETING
    label = Matkul(m.kode, nama).label
    dur = fmt_duration(m.duration, hours)
    if m.multi:
        dur += f" ({len(m.parts)} bagian)"
    sources = [p.path.name for p in m.parts]

    lines = [
        f"Mata kuliah : {label}",
        f"Pertemuan   : P{m.pertemuan:02d}",
        f"Tanggal     : {format_tanggal(m.tanggal)}",
        f"Durasi      : {dur}",
        f"Sumber      : {sources[0]}",
        *(f"              {s}" for s in sources[1:]),
        f"Model       : WhisperX {MODEL} ({COMPUTE_TYPE}), dibuat {now:%Y-%m-%d %H:%M}",
        RULE,
        "",
    ]
    for part, offset in zip(m.parts, part_offsets(m)):
        if m.multi:
            head = f"bagian {part.bagian} dari {m.total}"
            if part.short:
                head += f" · dilewati ({int(round(part.duration))} detik)"
            elif offset > 0:
                head += f" · mulai [{fmt_clock(offset, hours)}]"
            lines += [f"── {head} ──", ""]
        if part.short:
            continue
        paras = paragraphs(segments.get(part.path, []))
        if not paras:
            lines += ["(nggak ada ucapan terdeteksi)", ""]
        for start, text in paras:
            lines += [f"[{fmt_clock(offset + start, hours)}] {text}", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render_srt(segments: list[dict]) -> str:
    """Times stay on the part's own clock so it lines up with that audio file."""
    blocks = []
    n = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        n += 1
        end = max(seg["end"], seg["start"] + 0.001)
        blocks.append(f"{n}\n{fmt_srt_time(seg['start'])} --> {fmt_srt_time(end)}\n{text}\n")
    return "\n".join(blocks)


def atomic_write(path: Path, text: str) -> None:
    """Temp file beside the target, then rename: a killed run never leaves a
    half-written file that a later scan would take for a finished transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_meeting(m: Meeting, segments: dict[Path, list[dict]], nama: str = "",
                  now: datetime | None = None) -> list[Path]:
    """Write the SRTs first and the TXT last: the TXT is the 'done' marker, so a
    crash in between leaves the meeting still looking like todo."""
    written = []
    for part in m.transcribed_parts():
        p = m.srt_path(part)
        atomic_write(p, render_srt(segments.get(part.path, [])))
        written.append(p)
    atomic_write(m.txt_path(), render_txt(m, segments, nama, now))
    return [m.txt_path(), *written]
