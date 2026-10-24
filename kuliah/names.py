"""Lecture recording names: "MK1 - P04 - 2026-09-22 (bagian 1 dari 3).m4a".

The name is the only metadata a recording carries -- course, meeting number,
date, and which piece of a meeting recorded in several. Strict on structure,
loose on spacing and case, because the names are typed by hand on a phone.
"""
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

_NAME = re.compile(
    r"^\s*(?P<kode>[A-Za-z][A-Za-z0-9]{0,9})\s*-\s*"
    r"P\s*(?P<pertemuan>\d{1,2})\s*-\s*"
    r"(?P<tanggal>\d{4}-\d{2}-\d{2})"
    r"(?:\s*\(\s*bagian\s+(?P<bagian>\d{1,2})\s+dari\s+(?P<total>\d{1,2})\s*\))?"
    r"\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Parsed:
    kode: str                  # upper case: "MK1", "MK2"
    pertemuan: int             # 4 for "P04"
    tanggal: date
    bagian: int | None = None  # None for a meeting recorded in one file
    total: int | None = None

    @property
    def meeting_id(self) -> str:
        return f"{self.kode}-P{self.pertemuan:02d}"


def parse(filename: str | Path) -> Parsed | None:
    """Parse a file name (directory and extension are ignored), or None."""
    m = _NAME.match(Path(filename).stem)
    if not m:
        return None
    try:
        tanggal = date.fromisoformat(m["tanggal"])
    except ValueError:
        return None
    pertemuan = int(m["pertemuan"])
    bagian = int(m["bagian"]) if m["bagian"] else None
    total = int(m["total"]) if m["total"] else None
    if pertemuan < 1 or (bagian is not None and not 1 <= bagian <= total):
        return None
    return Parsed(m["kode"].upper(), pertemuan, tanggal, bagian, total)


def canonical(p: Parsed, ext: str) -> str:
    """The tidy file name for `p`; parse(canonical(p, ext)) == p."""
    name = f"{p.kode} - P{p.pertemuan:02d} - {p.tanggal.isoformat()}"
    if p.bagian is not None:
        name += f" (bagian {p.bagian} dari {p.total})"
    return name + ext.lower()
