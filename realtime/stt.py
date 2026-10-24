"""Streaming speech-to-text: VAD-segmented rolling buffer over faster-whisper.

Whisper is much more accurate on a complete utterance than on an arbitrary time
slice, so the cut points come from Silero VAD (trailing silence), not a clock.
While an utterance is still open we re-transcribe it every few seconds and emit
that as an interim result, which is what makes the UI feel live.
"""
import asyncio
import functools
import gc
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, get_speech_timestamps

from core import gpu
from core.filters import hallucination_reason, norm

from . import config

log = logging.getLogger("rtsm.stt")

SR = config.SAMPLE_RATE
_VAD_OPTS = VadOptions(min_speech_duration_ms=200, min_silence_duration_ms=200)

# One model for the whole process: loading costs ~20s and ~1.5GB of VRAM, and
# sessions are sequential, so every session reuses the same instance.
_MODEL: WhisperModel | None = None

# Every CUDA touch — load, transcribe, release — goes through this one thread.
# ctranslate2 keeps per-thread CUDA state, so calling the model from whichever
# thread asyncio.to_thread happens to hand out corrupts the context after a few
# hundred calls and the process dies with "CUDA failed with error unknown error".
_GPU = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper-gpu")

# The cross-process GPU lease, held for exactly as long as the model is
# resident. The batch pass waits on this instead of relying on us to remember
# not to run both.
_LEASE: gpu.Lease | None = None


async def _on_gpu(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_GPU, functools.partial(fn, *args, **kwargs))


def get_model() -> WhisperModel:
    """Blocking. Call from a worker thread, not the event loop."""
    global _MODEL, _LEASE
    if _MODEL is None:
        if _LEASE is None or not _LEASE.held:
            _LEASE = gpu.acquire(
                "realtime", wait=True, timeout=config.GPU_WAIT_TIMEOUT,
                on_wait=lambda who: log.info(
                    "GPU held by %s — waiting", (who or {}).get("name", "?")),
            )
        log.info("loading faster-whisper %s (%s, %s)",
                 config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE)
        try:
            _MODEL = WhisperModel(
                config.WHISPER_MODEL,
                device=config.WHISPER_DEVICE,
                compute_type=config.WHISPER_COMPUTE,
            )
        except BaseException:
            # A failed load (offline, OOM) must not keep the lease: with no
            # model, release_model() returns early and the batch pass and
            # lecture runs would wait on it forever.
            _LEASE.release()
            raise
        log.info("model ready")
    return _MODEL


async def warm_up():
    """Load the model ahead of the first session so Record is instant."""
    await _on_gpu(get_model)


async def release_model():
    await _on_gpu(_release_model_sync)


def _release_model_sync():
    """Drop the model and give the VRAM back.

    The batch pass wants large-v3 plus pyannote, which does not fit alongside
    turbo on a 6GB card — so the handoff has to run with this unloaded. This
    also drops the GPU lease, which is what lets the batch pass start. Next
    session reloads from the local cache (a few seconds).
    """
    global _MODEL, _LEASE
    if _MODEL is None:
        return
    _MODEL = None
    gc.collect()
    if _LEASE is not None and _LEASE.held:
        _LEASE.release()
    log.info("model released")


class Utterance:
    __slots__ = ("start", "end", "text")

    def __init__(self, start: float, end: float, text: str):
        self.start, self.end, self.text = start, end, text

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


class StreamingSTT:
    """Owns the model and the rolling audio buffer for one session."""

    def __init__(self, on_partial, on_final):
        """on_partial(text) and on_final(Utterance) are async callables."""
        self.on_partial = on_partial
        self.on_final = on_final

        self._model: WhisperModel | None = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._buf_start = 0.0          # session-clock seconds of _buf[0]
        self._lock = asyncio.Lock()    # guards _buf against feed/loop interleaving
        self._infer = asyncio.Lock()   # one whisper call at a time (6GB card)
        self._last_partial = 0.0
        self._last_partial_text = ""
        self._recent_norms: deque[str] = deque(maxlen=12)
        self._context = ""             # tail of committed text, fed back as prompt
        self._running = False
        self._task: asyncio.Task | None = None
        self.dropped = 0

    # --- lifecycle -------------------------------------------------------
    async def start(self):
        if self._model is None:
            self._model = await _on_gpu(get_model)
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        """Flush whatever is still buffered, then shut the loop down."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            tail, start = self._buf, self._buf_start
            self._buf = np.zeros(0, dtype=np.float32)
        if len(tail) >= int(SR * config.MIN_UTTERANCE):
            await self._commit(tail, start)

    # --- ingest ----------------------------------------------------------
    async def feed(self, pcm: bytes):
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        async with self._lock:
            self._buf = np.concatenate((self._buf, samples))

    @property
    def buffered_seconds(self) -> float:
        return len(self._buf) / SR

    # --- main loop -------------------------------------------------------
    async def _loop(self):
        while self._running:
            await asyncio.sleep(config.STT_TICK)
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("stt tick failed")

    async def _tick(self):
        async with self._lock:
            buf, start = self._buf, self._buf_start
        if len(buf) < int(SR * 0.5):
            return

        speech = await asyncio.to_thread(get_speech_timestamps, buf, _VAD_OPTS, SR)

        if not speech:
            # Pure silence. Keep a short tail so a word starting right now isn't
            # clipped, and slide the clock forward.
            if len(buf) > int(SR * 2.0):
                keep = int(SR * 0.3)
                await self._trim(len(buf) - keep)
                self._last_partial_text = ""
            return

        last_end = speech[-1]["end"]
        trailing_silence = (len(buf) - last_end) / SR
        duration = len(buf) / SR

        if trailing_silence >= config.SILENCE_HOLD:
            cut = min(len(buf), last_end + int(SR * 0.2))
            await self._commit(buf[:cut], start)
            await self._trim(cut)
        elif duration >= config.MAX_UTTERANCE:
            # Nobody paused. Cut at the last VAD boundary we have, or mid-word
            # at the buffer end if the whole window is one unbroken run.
            cut = last_end if last_end > int(SR * 1.0) else len(buf)
            await self._commit(buf[:cut], start)
            await self._trim(cut)
        elif time.monotonic() - self._last_partial >= config.PARTIAL_EVERY:
            self._last_partial = time.monotonic()
            window = buf[-int(SR * config.PARTIAL_WINDOW):]
            text = await self._transcribe(window, beam_size=1)
            if text and text != self._last_partial_text:
                self._last_partial_text = text
                await self.on_partial(text)

    async def _trim(self, cut: int):
        async with self._lock:
            self._buf = self._buf[cut:]
            self._buf_start += cut / SR
        self._last_partial = 0.0
        self._last_partial_text = ""

    async def _commit(self, audio: np.ndarray, start: float):
        duration = len(audio) / SR
        if duration < config.MIN_UTTERANCE:
            return
        text = await self._transcribe(audio, beam_size=5)
        if not text:
            return

        reason = hallucination_reason(text, duration, list(self._recent_norms))
        if reason:
            self.dropped += 1
            log.info("dropped %.1fs utterance (%s): %s", duration, reason, text[:60])
            return

        self._recent_norms.append(norm(text))
        self._context = (self._context + " " + text)[-400:]
        await self.on_final(Utterance(start, start + duration, text))

    async def _transcribe(self, audio: np.ndarray, beam_size: int) -> str:
        async with self._infer:
            segments, _ = await _on_gpu(
                self._model.transcribe,
                audio,
                language=config.LANGUAGE,
                beam_size=beam_size,
                vad_filter=False,               # we already segmented on VAD
                condition_on_previous_text=False,  # the top hallucination source
                initial_prompt=self._context or None,
            )
            return " ".join(s.text.strip() for s in segments).strip()
