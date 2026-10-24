"""The two Claude workers that run alongside a live meeting.

Both send the whole transcript-so-far rather than a delta: prompt caching makes
the repeated prefix cheap, and full context beats incremental summarisation for
catching a callback to something said forty minutes ago. Each worker gates on
MIN_NEW_CHARS so a quiet stretch costs nothing.
"""
import asyncio
import json
import logging
import time

from anthropic import AsyncAnthropic, APIStatusError, APIConnectionError, RateLimitError

from core.config import USER_NAME

from . import config

log = logging.getLogger("rtsm.brain")

SUMMARY_SYSTEM = """Lo asisten yang nemenin {user} di tengah meeting yang MASIH BERJALAN. Tugas lo: jaga ringkasan berjalan yang selalu up to date, plus action items.

Yang lo terima adalah transkrip live — hasil speech-to-text realtime, jadi wajar ada typo, kata kepotong, atau nama orang salah tulis. Baca maksudnya, jangan kepaku sama teks mentahnya.

Aturan:
- Bahasa Indonesia natural, padat. Ini dibaca sambil meeting jalan, bukan dibaca santai.
- `summary`: 3-6 kalimat. Kondisi terkini pembahasan, bukan sejarah lengkap.
- `key_points`: 3-7 poin paling penting sejauh ini. Satu baris per poin.
- `action_items`: SEMUA hal yang perlu dilakukan/diperhatikan — tugas, deadline, komitmen, keputusan yang harus diambil, follow-up ke orang tertentu, hal yang perlu dipelajari. Sertakan PIC & deadline kalau disebut. Kalau ragu actionable atau bukan, MASUKIN. Ini bagian terpenting.
- `open_threads`: hal yang digantung, belum dijawab, atau belum kelar dibahas.
- Jangan kasih caveat, disclaimer, atau bilang "transkrip belum lengkap". {user} udah tau ini live.""".replace("{user}", USER_NAME)

QUESTIONS_SYSTEM = """Lo asisten yang nemenin {user} di tengah meeting yang MASIH BERJALAN. Tugas lo cuma satu: siapin pertanyaan tajam yang bisa {user} lempar ke pembicara/host SEKARANG.

Yang lo terima transkrip live dari speech-to-text, jadi wajar berantakan. Baca maksudnya.

Kriteria pertanyaan yang bagus:
- Nyambung sama yang BARU AJA dibahas (menit-menit terakhir), bukan topik 1 jam lalu
- Ngebongkar hal yang pembicara lewatin, asumsikan diem-diem, atau jawab setengah
- Spesifik dan bisa langsung dibacain — bukan "bisa dijelaskan lebih lanjut?"
- Sopan tapi nggak basa-basi
- Bahasa Indonesia, nada kolega yang beneran nyimak

Tiap pertanyaan kasih `why`: satu kalimat pendek kenapa ini layak ditanya, biar {user} bisa nilai cepat mana yang mau dipakai.

Urutkan dari yang paling worth it. Maksimal 5. Kalau memang belum ada bahan yang cukup, balikin list kosong — jangan ngarang pertanyaan generik.""".replace("{user}", USER_NAME)

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "owner": {"type": "string"},
                    "due": {"type": "string"},
                },
                "required": ["text", "owner", "due"],
                "additionalProperties": False,
            },
        },
        "open_threads": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "key_points", "action_items", "open_threads"],
    "additionalProperties": False,
}

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "why": {"type": "string"},
                },
                "required": ["text", "why"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


class Brain:
    """Runs the summary and question workers against a live transcript."""

    def __init__(self, get_transcript, publish):
        """get_transcript() -> str, publish(event: dict) -> awaitable."""
        self.get_transcript = get_transcript
        self.publish = publish
        if config.OPENROUTER_API_KEY:
            # auth_token, not api_key: OpenRouter takes a Bearer token. Any
            # explicit credential also stops the SDK reading ANTHROPIC_API_KEY
            # from the env, so the batch pipeline's key never goes to OpenRouter.
            self.client = AsyncAnthropic(base_url=config.OPENROUTER_BASE_URL,
                                         auth_token=config.OPENROUTER_API_KEY)
        else:
            self.client = AsyncAnthropic()
        self.cost = 0.0
        self.calls = 0
        # Per worker: transcript lengths at each earlier call -- see _transcript_blocks.
        self._cuts: dict[str, list[int]] = {}
        self._force: set[str] = set()
        self._tasks: list[asyncio.Task] = []

    async def start(self):
        self._tasks = [
            asyncio.create_task(self._worker(
                "summary", config.SUMMARY_INTERVAL, SUMMARY_SYSTEM,
                SUMMARY_SCHEMA, config.MAX_SUMMARY_TOKENS,
                "Update ringkasan berjalan berdasarkan transkrip di atas.",
            )),
            asyncio.create_task(self._worker(
                "questions", config.QUESTIONS_INTERVAL, QUESTIONS_SYSTEM,
                QUESTIONS_SCHEMA, config.MAX_QUESTIONS_TOKENS,
                "Kasih pertanyaan yang paling layak dilempar ke pembicara sekarang.",
            )),
        ]

    async def stop(self):
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

    async def request_now(self, kind: str):
        """Force one worker to run on the next tick (the UI's refresh button)."""
        self._force.add(kind)

    async def _worker(self, kind, interval, system, schema, max_tokens, instruction):
        last_len = 0
        last_run = 0.0
        while True:
            await asyncio.sleep(2.0)
            transcript = self.get_transcript()
            forced = kind in self._force
            due = time.monotonic() - last_run >= interval
            enough = len(transcript) - last_len >= config.MIN_NEW_CHARS
            if not (forced or (due and enough)):
                continue
            if not transcript.strip():
                continue

            self._force.discard(kind)
            last_run = time.monotonic()
            last_len = len(transcript)
            try:
                data = await self._call(kind, system, schema, max_tokens, transcript, instruction)
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                log.warning("%s worker: %s", kind, e)
                await self.publish({"type": "brain_error", "kind": kind, "detail": str(e)[:200]})
                continue
            except Exception:
                log.exception("%s worker crashed", kind)
                continue

            await self.publish({"type": kind, "data": data, "cost": round(self.cost, 4)})

    def _transcript_blocks(self, kind, transcript) -> list[dict]:
        """The transcript as one block per call this worker has made, breakpoint on the last.

        A cache read only lands on a block boundary where an earlier request put
        a breakpoint, so a single block that grows never hits: every call would
        rewrite the whole transcript at the write price. Cutting where this
        worker's earlier calls ended keeps those blocks byte-identical, so the
        breakpoint's lookback finds the previous call's entry one block back and
        only the new lines are written. That relies on the transcript being
        append-only, which Session.transcript_text is.
        """
        cuts = self._cuts.setdefault(kind, [])
        if cuts and cuts[-1] > len(transcript):  # not a continuation; start over
            cuts.clear()
        if not cuts or cuts[-1] < len(transcript):
            cuts.append(len(transcript))
        bounds = [0, *cuts]
        blocks = [{"type": "text", "text": transcript[a:b]} for a, b in zip(bounds, bounds[1:])]
        blocks[0]["text"] = "<transkrip_live>\n" + blocks[0]["text"]
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    async def _call(self, kind, system, schema, max_tokens, transcript, instruction) -> dict:
        response = await self.client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{
                "role": "user",
                "content": [
                    *self._transcript_blocks(kind, transcript),
                    # The closing tag sits past the breakpoint: inside it, it
                    # would make the last block differ from the next call's.
                    {"type": "text", "text": f"\n</transkrip_live>\n\n{instruction}"},
                ],
            }],
            output_config={
                "effort": config.CLAUDE_EFFORT,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            raise RuntimeError(f"refused: {getattr(response.stop_details, 'category', None)}")

        self._track(response.usage)
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    def _track(self, usage):
        self.calls += 1
        # Keyed by Anthropic id; OpenRouter bills Claude at the same rates, so
        # "anthropic/claude-haiku-5.5" looks up "claude-haiku-5-5".
        model = config.CLAUDE_MODEL.removeprefix("anthropic/").replace(".", "-")
        rates = {"claude-opus-5": (5.0, 25.0), "claude-sonnet-5": (2.0, 10.0),
                 "claude-haiku-5-5": (0.10, 0.50), "claude-haiku-4-5": (1.0, 5.0)}
        rin, rout = rates.get(model, (5.0, 25.0))
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Haiku 5.5 bills the whole request at 5x once the prompt passes 100K
        # tokens -- roughly six hours of transcript.
        if model == "claude-haiku-5-5" and usage.input_tokens + cached + written > 100_000:
            rin, rout = rin * 5, rout * 5
        self.cost += (
            usage.input_tokens * rin
            + cached * rin * 0.1
            + written * rin * 1.25
            + usage.output_tokens * rout
        ) / 1_000_000
