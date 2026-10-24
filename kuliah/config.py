"""Paths and knobs for `supermeet.py kuliah` and its upload page.

Shared by the CLI (kuliah/run.py) and the web page (realtime/kuliah_web.py), so
both agree on where recordings wait and where transcripts land. Paths can be
overridden with env vars.
"""
import os
from pathlib import Path

from core import device
from core.config import PROJECT_DIR

# Finished transcripts, one subfolder per course code:
#   <OUTPUT_DIR>/MK1/[P04] Transkrip.txt
# Local on purpose: results are fetched through the upload page, not synced.
OUTPUT_DIR = Path(os.environ.get("KULIAH_OUTPUT", PROJECT_DIR / "kuliah-output"))

# Uploaded recordings wait here until processed. Copies only -- the originals
# stay on the phone / PC they were uploaded from. Subfolders starting with "_"
# are skipped by the scanner; UNNAMED_DIR is one of them.
UPLOAD_DIR = Path(os.environ.get("KULIAH_UPLOADS", PROJECT_DIR / "kuliah-uploads"))
# Uploads whose name does not parse, held until a name is given on the page.
UNNAMED_DIR = UPLOAD_DIR / "_perlu-nama"

# Per-part transcript cache. An interrupted run resumes without redoing parts
# that finished, and the text can be re-rendered without the GPU.
CACHE_DIR = PROJECT_DIR / "transcripts" / "kuliah"
LOG_FILE = PROJECT_DIR / "logs" / "kuliah.log"

MATKUL_FILE = Path(__file__).resolve().parent / "matkul.toml"

AUDIO_EXTS = frozenset({".opus", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".webm"})

# Parts shorter than this are accidental taps on record, not lecture.
MIN_PART_SECONDS = 60.0

# Accuracy over speed on NVIDIA: large-v3, not turbo. int8 until a side-by-side
# on a real lecture shows float16 is worth the extra VRAM on the 6GB card. With
# no NVIDIA GPU it is large-v3-turbo on the CPU (core/device.py).
CHOICE = device.pick()
MODEL = CHOICE.model
COMPUTE_TYPE = os.environ.get("KULIAH_COMPUTE", CHOICE.compute_type)
LANGUAGE = "id"
BATCH_SIZE = 4
# Same reasoning as batch/pipeline.py: 10s chunks keep faint speech from being
# smeared into hallucination.
CHUNK_SIZE = 10

# One run at a time, whether started from the CLI or the page.
RUN_LOCK = Path(os.environ.get("KULIAH_LOCK", "/tmp/supermeet-kuliah.lock"))
