"""Timestamp formatting shared by both pipelines' transcripts."""


def fmt_ts(seconds: float) -> str:
    """Seconds -> HH:MM:SS, the anchor format used in transcripts and summaries."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
