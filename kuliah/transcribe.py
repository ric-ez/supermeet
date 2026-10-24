"""WhisperX transcription of lecture parts, with a per-part cache.

One MeetingTranscriber per meeting: it holds the GPU lease and the model for
that meeting only, so the realtime server can claim the card between meetings.
torch/whisperx are imported inside functions -- importing this module (e.g. for
--dry-run) must stay cheap and must not touch CUDA.
"""
import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Callable

from core import gpu
from core.filters import clean_segments

from . import matkul
from .config import (BATCH_SIZE, CACHE_DIR, CHOICE, CHUNK_SIZE, COMPUTE_TYPE, LANGUAGE,
                     MODEL)

# on_progress(stage, fraction of this part, 0..1); stage is "load" or "transcribe".
PartProgress = Callable[[str, float], None]


class NoSpeech(RuntimeError):
    """Nothing usable came out of the audio."""


def cache_key(path: Path, initial_prompt: str | None) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "model": MODEL,
            "compute_type": COMPUTE_TYPE, "language": LANGUAGE,
            "chunk_size": CHUNK_SIZE, "initial_prompt": initial_prompt}


def cache_path(path: Path) -> Path:
    return CACHE_DIR / f"{path.stem}.json"


def load_cached(path: Path, initial_prompt: str | None) -> list[dict] | None:
    try:
        data = json.loads(cache_path(path).read_text(encoding="utf-8"))
        if data["key"] == cache_key(path, initial_prompt):
            return data["segments"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def save_cache(path: Path, initial_prompt: str | None, segments: list[dict],
               dropped: int) -> None:
    dest = cache_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {"key": cache_key(path, initial_prompt), "segments": segments,
               "dropped": dropped}
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class MeetingTranscriber:
    """Context manager: `with MeetingTranscriber(kode, on_wait) as t: t.part(...)`.

    The lease and model are taken lazily on the first part that is not cached,
    so re-rendering from cache never waits for (or blocks) the GPU.
    """

    def __init__(self, kode: str, on_wait: Callable[[dict], None] | None = None):
        self.prompt = matkul.get(kode).initial_prompt()
        self._on_wait = on_wait
        self._lease = None
        self._model = None

    def __enter__(self) -> "MeetingTranscriber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def part(self, path: Path, on_progress: PartProgress | None = None) -> list[dict]:
        """Cleaned segments of one file, on that file's own clock."""
        on_progress = on_progress or (lambda stage, frac: None)
        cached = load_cached(path, self.prompt)
        if cached is not None:
            on_progress("transcribe", 1.0)
            return cached

        on_progress("load", 0.0)
        model = self._ensure_model()
        import whisperx
        on_progress("transcribe", 0.0)
        audio = whisperx.load_audio(str(path))
        result = model.transcribe(
            audio, batch_size=BATCH_SIZE, language=LANGUAGE, chunk_size=CHUNK_SIZE,
            # whisperx reports 0..100 per decoded chunk.
            progress_callback=lambda pct: on_progress("transcribe", pct / 100.0))
        kept, dropped = clean_segments(result["segments"])
        segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()}
                    for s in kept]
        save_cache(path, self.prompt, segments, len(dropped))
        return segments

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        # Lease first: loading a second model on top of the realtime server's
        # hangs rather than failing (see core/gpu.py).
        self._lease = gpu.acquire("kuliah", wait=True, timeout=None,
                                  on_wait=self._on_wait)
        from core.quiet import silence_model_noise
        silence_model_noise()
        import whisperx
        print(f"\n  Whisper: {gpu.describe(CHOICE)}", flush=True)  # \n: a progress bar may be open
        self._model = whisperx.load_model(
            MODEL, device=CHOICE.device, compute_type=COMPUTE_TYPE, language=LANGUAGE,
            asr_options={"condition_on_previous_text": False,
                         "initial_prompt": self.prompt})
        return self._model

    def close(self) -> None:
        was_loaded = self._model is not None
        self._model = None
        try:
            if was_loaded:
                gc.collect()
                if CHOICE.is_cuda:
                    import torch
                    torch.cuda.empty_cache()
        finally:
            if self._lease is not None:
                self._lease.release()
                self._lease = None
