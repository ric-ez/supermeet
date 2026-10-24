#!/usr/bin/env python3
"""Whisper hallucination filters — the single source for both pipelines.

The batch pipeline and the realtime server both read this module, so a
threshold tuned here takes effect in both. They used to live in separate
projects with separate copies, and drifted apart.

Two entry points over one decision function:
  hallucination_reason(text, duration, recent_norms) -> reason | None
  clean_segments(segments)                           -> (kept, dropped)
plus collapse_loops(text), which clean_segments applies to what it keeps.
"""
import re
import unicodedata

# Whisper invents canned phrases over silence; measured on real audio those run
# 0.4-2.2 chars/sec while genuine speech runs 6.5-15.0. 3.0 sits in the gap.
MIN_CHARS_PER_SEC = 3.0
# Measured: at 5.0 a hallucination sitting just under the bound slips through.
RATE_MIN_DURATION = 3.0

# A short utterance that exactly repeats one of the last few is almost always
# Whisper filling a pause. Streaming hits this constantly because every pause
# becomes its own inference call, but batch sees it too.
ECHO_WINDOW = 6
ECHO_MAX_DURATION = 3.0

SUBTITLE_BOILERPLATE = re.compile(
    r"terima kasih telah menonton"
    r"|sampai jumpa di video (berikutnya|selanjutnya)"
    r"|jangan lupa (like|subscribe)"
    r"|subtitle by|sub by|transcri(bed|pt) by",
    re.IGNORECASE,
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\s]", "", text).strip()


def is_repetitive(text: str, min_repeats: int = 3) -> bool:
    """True if one short phrase repeats from the start and covers most of it."""
    words = norm(text).split()
    if len(words) < min_repeats:
        return False
    for size in range(1, len(words) // min_repeats + 1):
        phrase = words[:size]
        reps = 0
        while words[reps * size:(reps + 1) * size] == phrase:
            reps += 1
        if reps >= min_repeats and reps * size >= len(words) * 0.8:
            return True
    return False


def is_degenerate(text: str) -> bool:
    """True for stutter-loops that is_repetitive misses.

    is_repetitive only anchors at the first word, so a single leading token
    hides the loop: "di BFC, BFC, BFC, BFC." walks straight past it. Both
    checks below are position-independent.
    """
    words = norm(text).split()
    if len(words) < 4:
        return False

    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    if best >= 3:
        return True

    # One word dominating the utterance is a loop, not speech.
    return len(words) >= 5 and max(words.count(w) for w in set(words)) / len(words) >= 0.6


def hallucination_reason(
    text: str, duration: float, recent_norms: "list[str] | None" = None
) -> str | None:
    """Why this utterance should be dropped, or None to keep it.

    recent_norms: normalised text of the last few kept utterances, newest last.
    """
    recent = recent_norms or []
    n = norm(text)
    if not n:
        return "empty"
    if SUBTITLE_BOILERPLATE.search(text):
        return "boilerplate"
    if is_repetitive(text) or is_degenerate(text):
        return "repetition"
    if duration >= RATE_MIN_DURATION and len(text) / max(duration, 1e-6) < MIN_CHARS_PER_SEC:
        return "low-rate"
    if recent and n == recent[-1]:
        return "duplicate"
    if duration <= ECHO_MAX_DURATION and n in recent[-ECHO_WINDOW:]:
        return "echo"
    return None


# A phrase of up to this many words repeated back-to-back this many times is a
# decoding loop, even with real speech around it. Seen in batch: "Ataupun
# ngomongin jangkang. 10 juta, aku bagi 10, 10 juta, 10 juta, ..." ran nine
# copies to the end of the chunk. The segment checks above miss it -- the loop
# does not start the utterance and no single word dominates -- and dropping the
# segment would lose the real opening anyway, so the loop is cut instead.
LOOP_MAX_UNIT = 4
LOOP_MIN_REPEATS = 5


def collapse_loops(text: str) -> str:
    """Cut any embedded phrase loop down to one copy; other text is untouched."""
    tokens = text.split()
    keys = [norm(t) for t in tokens]
    out, i, changed = [], 0, False
    while i < len(tokens):
        advance = 1
        for size in range(1, LOOP_MAX_UNIT + 1):
            unit = keys[i:i + size]
            if len(unit) < size:
                break
            if not all(unit):
                continue
            reps = 1
            while keys[i + reps * size:i + (reps + 1) * size] == unit:
                reps += 1
            if reps >= LOOP_MIN_REPEATS:
                out.extend(tokens[i:i + size])
                advance, changed = reps * size, True
                break
        if advance == 1:
            out.append(tokens[i])
        i += advance
    return " ".join(out) if changed else text


def clean_segments(segments: list) -> tuple:
    """Drop Whisper hallucinations from a finished transcript.

    Segments that survive have embedded phrase loops collapsed.
    Returns (kept, dropped), where dropped entries are
    (reason, start, duration, text).
    """
    kept, dropped, recent = [], [], []
    for seg in segments:
        text = seg["text"].strip()
        duration = max(seg["end"] - seg["start"], 1e-6)

        # Judged on the raw text first, so a segment that is nothing but a loop
        # is still dropped rather than shrunk to one plausible-looking copy.
        reason = hallucination_reason(text, duration, recent)
        if reason:
            dropped.append((reason, seg["start"], duration, text))
        else:
            collapsed = collapse_loops(text)
            if collapsed != text:
                seg = {**seg, "text": collapsed}
            kept.append(seg)
            recent.append(norm(collapsed))
            del recent[:-ECHO_WINDOW]
    return kept, dropped
