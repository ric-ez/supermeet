"""Per-course full names and vocabulary, read from matkul.toml.

The file is personal and git-ignored; matkul.example.toml is the template, and
without a matkul.toml every course is still processed, just without extras.

Read fresh on every call: the file is edited by hand between runs, and it is
tiny.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .config import MATKUL_FILE


@dataclass(frozen=True)
class Matkul:
    kode: str
    nama: str = ""
    istilah: str = ""

    @property
    def label(self) -> str:
        """"Manajemen Keuangan 1 (MK1)", or just "MK1" when no name is set."""
        return f"{self.nama} ({self.kode})" if self.nama else self.kode

    def initial_prompt(self) -> str | None:
        """Whisper's initial_prompt for this course, or None if nothing is set.

        Written as a sentence in the lecture's language -- Whisper treats the
        prompt as preceding transcript, so it biases spelling best when it reads
        like one.
        """
        parts = []
        if self.nama:
            parts.append(f"Kuliah {self.nama}.")
        if self.istilah:
            parts.append(f"Istilah: {self.istilah.rstrip('.')}.")
        return " ".join(parts) or None


def load(path: Path = MATKUL_FILE) -> dict[str, Matkul]:
    """Every configured course, keyed by upper-case code."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    return {
        kode.upper(): Matkul(kode.upper(),
                             str(v.get("nama", "")).strip(),
                             str(v.get("istilah", "")).strip())
        for kode, v in data.items() if isinstance(v, dict)
    }


def get(kode: str) -> Matkul:
    """The course for `kode`; an unknown code still works, just without extras."""
    kode = kode.upper()
    return load().get(kode, Matkul(kode))
