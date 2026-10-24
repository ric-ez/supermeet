"""Knobs for the realtime server. Override any of these with an env var of the
same name plus an RTSM_ prefix — see README.

Shared paths (cloud folders, transcripts) live in core/config.py; these are the
server's own tuning, which the batch pass has no opinion about."""
import os
from pathlib import Path

from core import device, env

# Before the first os.environ read below, so RTSM_* and OPENROUTER_API_KEY in
# .env count. (core.config does the same for the shared settings.)
env.load()

def _env(name: str, default, cast=str):
    raw = os.environ.get(f"RTSM_{name}")
    return cast(raw) if raw is not None else default

PROJECT_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_DIR / "sessions"
WEB_DIR = Path(__file__).resolve().parent / "web"
# The archival pass runs as a separate process so it gets the GPU to itself.
CLI = PROJECT_DIR / "supermeet.py"

# === Server ===
HOST = _env("HOST", "0.0.0.0")
PORT = _env("PORT", 8848, int)

# === Audio ===
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16

# === STT ===
# large-v3-turbo is ~1.5GB in int8 — fits alongside nothing else on a 6GB card,
# so the batch pipeline and this server must not run at the same time.
WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
# auto: CUDA when there is an NVIDIA GPU, the CPU otherwise (a MacBook).
# RTSM_WHISPER_DEVICE=cuda or cpu forces it.
WHISPER_DEVICE = _env("WHISPER_DEVICE", "auto")
if WHISPER_DEVICE == "auto":
    WHISPER_DEVICE = device.pick().device
WHISPER_COMPUTE = _env("WHISPER_COMPUTE", "int8")
LANGUAGE = _env("LANGUAGE", "id")

# Utterance segmentation. Whisper is far more accurate on a complete utterance
# than on an arbitrary time slice, so we cut on silence rather than on a clock.
SILENCE_HOLD = 0.60      # seconds of trailing silence that closes an utterance
MIN_UTTERANCE = 0.40     # ignore speech runs shorter than this (coughs, clicks)
MAX_UTTERANCE = 22.0     # force a cut if someone never pauses
PARTIAL_EVERY = 2.5      # seconds between interim (uncommitted) transcriptions
# Interims only ever re-transcribe this much of the open utterance. Without the
# cap, a speaker who never pauses makes every interim more expensive than the
# last, until they cost more GPU than the interval between them.
PARTIAL_WINDOW = 12.0
STT_TICK = 0.35          # how often the STT loop inspects the buffer

# === Brain (Claude) ===
# Through OpenRouter when OPENROUTER_API_KEY is set, straight to Anthropic
# otherwise. Same SDK and same request either way -- OpenRouter serves the
# Messages API at /api/v1/messages -- but its model ids are slugs
# ("anthropic/claude-haiku-5.5"), not Anthropic's ("claude-haiku-5-5").
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
# Haiku 5.5: an estimated ~$0.14 for a 2-hour meeting, against ~$2.8 on
# Sonnet 5, the previous default. If the host questions turn generic, try
# RTSM_CLAUDE_EFFORT=medium before going back to a bigger model.
CLAUDE_MODEL = _env("CLAUDE_MODEL",
                    "anthropic/claude-haiku-5.5" if OPENROUTER_API_KEY else "claude-haiku-5-5")
CLAUDE_EFFORT = _env("CLAUDE_EFFORT", "low")  # these are short, latency-sensitive calls
SUMMARY_INTERVAL = _env("SUMMARY_INTERVAL", 60.0, float)
QUESTIONS_INTERVAL = _env("QUESTIONS_INTERVAL", 45.0, float)
# Don't spend a call on nothing new — both workers gate on fresh transcript text.
MIN_NEW_CHARS = _env("MIN_NEW_CHARS", 350, int)
MAX_SUMMARY_TOKENS = 4000
MAX_QUESTIONS_TOKENS = 1500

# How long the server waits for the batch pass to release the GPU before
# giving up. Long, because a queued batch run finishes on its own.
GPU_WAIT_TIMEOUT = _env("GPU_WAIT_TIMEOUT", 1800.0, float)
# A model load that never returns must not wedge the server. Hit on a 6GB card
# when the batch pipeline is still holding VRAM.
MODEL_LOAD_TIMEOUT = _env("MODEL_LOAD_TIMEOUT", 180.0, float)
# How long a new session waits for an in-flight archival handoff to release the
# GPU before starting anyway. A meeting matters more than the archive pass.
HANDOFF_WAIT_TIMEOUT = _env("HANDOFF_WAIT_TIMEOUT", 600.0, float)
