"""Load the project's .env into os.environ, so setup is "copy .env.example, fill it in".

No dependency: the format is KEY=value lines, `#` comments, optional `export `
prefix and optional single or double quotes. A variable already in the real
environment always wins over the file, so `GEMINI_API_KEY=x python supermeet.py`
and shell exports keep working. Safe to call more than once.
"""
import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def parse(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[key] = value
    return values


def load(path: Path = ENV_FILE) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for key, value in parse(text).items():
        # An empty value in the file ("KEY=") means "not set", not "set to blank".
        if value and key not in os.environ:
            os.environ[key] = value
