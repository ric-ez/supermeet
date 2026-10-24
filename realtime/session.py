"""One live meeting: its audio on disk, its transcript, and the handoff to the
batch pipeline once it ends."""
import asyncio
import json
import logging
import subprocess
import sys
import time
import wave
from datetime import datetime
from pathlib import Path

from core.timestamps import fmt_ts

from . import config

log = logging.getLogger("rtsm.session")


class Session:
    def __init__(self, title: str | None = None, ai: bool = True):
        # ai=False is the transcribe-only mode: no Claude workers, so no
        # summary/questions panels and no API cost.
        self.ai = ai
        self.started_at = datetime.now()
        self.id = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.title = title or f"Meeting {self.started_at.strftime('%d %b %Y %H:%M')}"
        self.dir = config.SESSIONS_DIR / self.id
        self.dir.mkdir(parents=True, exist_ok=True)

        self.utterances: list[dict] = []
        self.summary: dict | None = None
        self.questions: list[dict] = []
        self._t0 = time.monotonic()
        # Named after the session, not "audio": the batch pass derives its
        # output filenames from the audio stem, so a fixed name makes every
        # session overwrite the previous one's PDFs in the cloud folder.
        self.audio_path = self.dir / f"meeting_{self.id}.wav"
        self._wav = wave.open(str(self.audio_path), "wb")
        self._wav.setnchannels(config.CHANNELS)
        self._wav.setsampwidth(config.SAMPLE_WIDTH)
        self._wav.setframerate(config.SAMPLE_RATE)
        self._closed = False
        log.info("session %s started -> %s", self.id, self.dir)

    # --- clock -----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    # --- writes ----------------------------------------------------------
    def write_audio(self, pcm: bytes):
        if not self._closed:
            self._wav.writeframes(pcm)

    def add_utterance(self, start: float, end: float, text: str):
        self.utterances.append({"start": start, "end": end, "text": text})

    def transcript_text(self) -> str:
        """Timestamped plain text — what the Claude workers read."""
        return "\n".join(f"[{fmt_ts(u['start'])}] {u['text']}" for u in self.utterances)

    def state(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "ai": self.ai,
            "elapsed": self.elapsed,
            "utterances": self.utterances,
            "summary": self.summary,
            "questions": self.questions,
        }

    # --- teardown --------------------------------------------------------
    def close(self) -> dict:
        """Flush everything to disk. Returns the paths written."""
        if self._closed:
            return {}
        self._closed = True
        self._wav.close()

        md = self.dir / "transcript.md"
        md.write_text(
            f"# {self.title}\n\n_Transkrip realtime — {self.started_at:%d %b %Y %H:%M}_\n\n"
            + "\n\n".join(f"**[{fmt_ts(u['start'])}]** {u['text']}" for u in self.utterances)
            + "\n",
            encoding="utf-8",
        )
        (self.dir / "session.json").write_text(
            json.dumps(self.state(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        notes = self.dir / "notes.md"
        notes.write_text(self._render_notes(), encoding="utf-8")
        log.info("session %s closed (%d utterances)", self.id, len(self.utterances))
        return {"transcript": str(md), "notes": str(notes), "audio": str(self.audio_path)}

    def _render_notes(self) -> str:
        s = self.summary or {}
        lines = [f"# {self.title}", "",
                 f"_Catatan realtime — {self.started_at:%d %b %Y %H:%M}, durasi {fmt_ts(self.elapsed)}_", ""]
        if s.get("summary"):
            lines += ["## Ringkasan", "", s["summary"], ""]
        if s.get("key_points"):
            lines += ["## Key Points", ""] + [f"- {p}" for p in s["key_points"]] + [""]
        if s.get("action_items"):
            lines += ["## ⚡ Action Items", ""]
            for a in s["action_items"]:
                meta = " · ".join(x for x in (a.get("owner"), a.get("due")) if x)
                lines.append(f"- {a['text']}" + (f" _({meta})_" if meta else ""))
            lines.append("")
        if s.get("open_threads"):
            lines += ["## Masih Menggantung", ""] + [f"- {t}" for t in s["open_threads"]] + [""]
        if self.questions:
            lines += ["## Pertanyaan yang Sempat Disiapkan", ""]
            lines += [f"- {q['text']}" for q in self.questions] + [""]
        return "\n".join(lines)

    # --- handoff ---------------------------------------------------------
    async def handoff(self) -> str:
        """Re-run the audio through the batch pass for the archival transcript.

        The realtime transcript is good enough to act on during the meeting;
        the batch pipeline (large-v3 + diarization) is what goes in the
        archive. It runs as a separate process so it gets the whole GPU — the
        caller releases the streaming model first, and the batch pass takes
        the GPU lease itself.
        """
        if not config.CLI.exists():
            return f"skipped: {config.CLI} not found"

        log.info("handoff: %s transcribe %s", config.CLI, self.audio_path)
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(config.CLI), "transcribe", str(self.audio_path),
            cwd=str(config.PROJECT_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        tail = out.decode("utf-8", "replace").strip().splitlines()[-15:]
        (self.dir / "handoff.log").write_text("\n".join(tail), encoding="utf-8")
        return "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
