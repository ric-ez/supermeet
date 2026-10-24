#!/usr/bin/env python3
"""Mute the model stack's startup chatter so the step bars stay readable.

whisperx, lightning, pyannote and huggingface_hub each narrate model loading
on stderr, which interleaves with the progress UI in batch/pipeline.py and
makes a healthy run look like it is failing. Everything muted here is either
progress we already print ourselves or a notice we have looked at and decided
to accept -- each one is matched on its exact text, so a *new* warning from
the same library still reaches the terminal.
"""
import logging
import warnings

# Matched against the warning text as a regex, so escape the punctuation.
_MESSAGES = (
    # pyannote turns TF32 off when it moves to CUDA. That is the accuracy /
    # reproducibility tradeoff we want; turning it back on is not an option we
    # are offering, so the instructions are just noise.
    r"TensorFloat-32 \(TF32\) has been disabled",
    # Raised per pooled window that covers a single frame -- a speaker turn too
    # short to have any variance. Pooling yields 0 std and moves on.
    r"std\(\): degrees of freedom is <= 0",
)

# Loggers whose INFO/WARNING lines duplicate our own step reporting.
_LOGGERS = {
    # "Performing voice activity detection using Pyannote..." -- step 1 already
    # says it is transcribing.
    "whisperx.vads.pyannote": logging.WARNING,
    # The checkpoint-upgrade notice for pyannote's bundled VAD weights. The
    # upgrade happens in memory every load and changes nothing for us.
    "lightning.pytorch.utilities.migration.utils": logging.WARNING,
    "pytorch_lightning.utilities.migration.utils": logging.WARNING,
    # Hub timeouts that huggingface_hub retries on its own. A download that
    # genuinely fails still raises, and the pipeline reports that.
    "huggingface_hub.utils._http": logging.ERROR,
}


def silence_model_noise() -> None:
    """Install the filters. Safe to call more than once."""
    for pattern in _MESSAGES:
        warnings.filterwarnings("ignore", message=pattern)
    for name, level in _LOGGERS.items():
        logging.getLogger(name).setLevel(level)
