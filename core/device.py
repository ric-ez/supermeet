"""Which device and Whisper model to use, decided once and shared.

NVIDIA GPU (CUDA) keeps the original, accuracy-first setup. Anything else --
a MacBook, a laptop without an NVIDIA card -- runs on the CPU with the smaller,
much faster `large-v3-turbo`, because large-v3 on a CPU takes hours per
recording. (ctranslate2, which WhisperX and faster-whisper sit on, has no
Apple GPU backend, so a Mac is a CPU machine here.)

Overrides, both optional:
  SUPERMEET_DEVICE         auto (default) | cuda | cpu
  SUPERMEET_WHISPER_MODEL  any faster-whisper model name
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    device: str          # "cuda" | "cpu"
    compute_type: str    # passed to WhisperX / faster-whisper
    model: str           # Whisper model name

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"


def cuda_available() -> bool:
    """True when torch can see an NVIDIA GPU. False if torch is missing."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pick() -> Choice:
    wanted = os.environ.get("SUPERMEET_DEVICE", "auto").strip().lower() or "auto"
    if wanted not in ("auto", "cuda", "cpu"):
        raise SystemExit(f"SUPERMEET_DEVICE must be auto, cuda or cpu, not {wanted!r}")
    device = ("cuda" if cuda_available() else "cpu") if wanted == "auto" else wanted
    default_model = "large-v3" if device == "cuda" else "large-v3-turbo"
    model = os.environ.get("SUPERMEET_WHISPER_MODEL", "").strip() or default_model
    return Choice(device=device, compute_type="int8", model=model)
