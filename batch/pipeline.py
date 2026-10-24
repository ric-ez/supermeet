#!/usr/bin/env python3
"""All-in-one: transcribe + diarize → verbatim transcript PDF + summary PDF → synced cloud folder."""
import os
import shutil
import sys
import time
import threading
import subprocess
import tempfile
from pathlib import Path
from contextlib import contextmanager

# Allow running this file directly, not only through supermeet.py, and load
# .env before the settings below read the environment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import device, env
env.load()

import torch
import whisperx
from google import genai
from google.genai import errors as genai_errors, types

# === Config ===
# NVIDIA: large-v3. No NVIDIA GPU: the CPU path with large-v3-turbo (core/device.py).
CHOICE = device.pick()
MODEL = CHOICE.model
LANGUAGE = "id"
DEVICE = CHOICE.device
COMPUTE_TYPE = CHOICE.compute_type
BATCH_SIZE = 4
# whisperx merges VAD segments up to this length. The default of 30 is too coarse
# for speaker attribution (turns are 1-5s) and smears faint speech into
# hallucination; 10 tracks pyannote's split far more closely and recovers real
# utterances that 30 loses entirely.
CHUNK_SIZE = 10

# Hallucination thresholds live in core/filters.py, which the realtime server
# reads too — tune them there and both pipelines follow.
# How long a batch run waits for the GPU before giving up. The watcher retries
# on its next poll, so giving up is cheap.
GPU_WAIT_TIMEOUT = float(os.environ.get("SUPERMEET_GPU_WAIT", 1800))
# Emit a timestamp at least this often, even mid-speaker, so the summary has
# navigation anchors in single-speaker recordings.
TIMESTAMP_INTERVAL = 60.0
# pyannote emits zero-length phantom turns; they inflate the speaker count and
# tell us nothing about who spoke.
MIN_TURN_DURATION = 0.1

from core import gpu
from core.config import PROJECT_DIR, TRANSCRIPTS_DIR, OUTPUT_DIR, TRANSCRIPT_OUTPUT_DIR, USER_NAME
from core.filters import clean_segments
from core.pdf import render_pdf
from core.quiet import silence_model_noise
from core.timestamps import fmt_ts

# The model is chosen per run by `supermeet.py watch`, which exports it; the
# default here is what a bare `transcribe` uses. A free-tier Flash model, so the
# summary costs nothing (within the free tier's rate limits).
GEMINI_MODEL = os.environ.get("SUPERMEET_GEMINI_MODEL", "gemini-3.8-flash")
# Extra context for this run's summaries, asked once by `watch`.
EXTRA_PROMPT = os.environ.get("SUPERMEET_EXTRA_PROMPT", "").strip()
# Set by `watch` when its "Transcribe only" row is picked; --transcribe-only does
# the same for one run. No Gemini call at all, so no API key is needed either.
TRANSCRIBE_ONLY = os.environ.get("SUPERMEET_TRANSCRIBE_ONLY", "") not in ("", "0")
# Shared budget for the model's thinking + the visible summary, not the summary
# alone: Gemini thinking tokens count against the output cap, so a tight cap can
# be eaten by thinking and truncate the answer. 65536 is gemini-3.8-flash's own
# output limit, and the call streams, so the headroom costs nothing until it is used.
MAX_SUMMARY_TOKENS = 65536
# A summary pass is one long request, so give it room. Seconds here (the
# environment variable is), though the SDK itself takes milliseconds. It is a
# per-connection limit, not a cap on how long the model may think.
API_TIMEOUT = float(os.environ.get("SUPERMEET_API_TIMEOUT", 1800))
# Rate limits (429) and overload (5xx) are normal on the free tier. Retry a few
# times here, waiting what the API asks for when it says; if it still fails the
# run exits 1 and the watcher tries again on a later poll.
API_RETRY_WAITS = (30, 60, 120)
MAX_RETRY_WAIT = 300
# Exit codes, read by batch/watcher.sh to decide what happens to the file. A
# failure that retrying cannot fix -- a truncated summary, a blocked prompt, a bad
# key or model name -- used to be retried every poll forever. Anything uncaught
# still exits 1 and is retried a few times.
EXIT_PERMANENT = 2    # the same input will fail the same way: set it aside
EXIT_BUSY = 75        # GPU held by another process: retry, not the file's fault


class PermanentFailure(RuntimeError):
    """Retrying the same file cannot succeed; the run exits EXIT_PERMANENT."""


PROMPT_INTRO = (f"Lo asisten yang bantu {USER_NAME} meringkas konten audio (lecture, rapat, "
                "podcast, wawancara, diskusi, atau apapun).")

SECTIONS = {
    "ringkasan": """## Ringkasan Singkat
2-3 kalimat inti dari konten.""",

    "takeaways": """## Key Takeaways
3-7 poin terpenting. Bullet list.""",

    "detail": """## Poin-Poin Penting (Detail)
Detail per topik. Pakai timestamp [HH:MM:SS] untuk navigasi. Kalau ada multi-speaker, attribute insight ke speaker yang ngomong.""",

    "action_items": f"""## ⚡ Action Items
**WAJIB DIPRIORITASKAN — JANGAN MISS APA PUN.** Ekstrak SEMUA hal yang perlu {USER_NAME} atau orang lain lakukan/perhatikan. Termasuk:
- PR / tugas / homework yang disebut
- Kisi-kisi ujian / quiz / topik yang akan diuji
- Deadline / tenggat waktu (cantumkan tanggal/waktu kalau ada)
- Janji / komitmen / commitment yang dibuat
- Hal yang perlu di-research / dipelajari lebih lanjut
- Keputusan yang perlu diambil
- Follow-up dengan orang tertentu
- Action concrete yang implied tapi nggak eksplisit ("kalian harus paham X" → action: pelajari X)

Format bullet list dengan deadline & PIC kalau ada. Kalau ragu actionable atau bukan, **masukin aja** — better safe than sorry. Action items adalah bagian terpenting dari ringkasan ini.""",

    "konteks": """## Konteks Tambahan
Catatan lain yang penting (mood, dynamics speaker, hal aneh yang patut diingat, references yang disebut).""",
}

PROMPT_RULES = """Aturan:
- Format markdown clean, ready for Obsidian
- Bahasa Indonesia akademik tapi natural
- Jangan kasih caveat atau disclaimer
- Kalau transkrip pakai label `[SPEAKER_00]`, dst., attribute insight & action ke speaker yang tepat"""


def build_system_prompt() -> str:
    head = f"{PROMPT_INTRO} Output ringkasan struktur ini, bahasa Indonesia natural:"
    return "\n\n".join([head, *SECTIONS.values(), PROMPT_RULES])


SYSTEM_PROMPT = build_system_prompt()


# === Progress UI ===
class ProgressTracker:
    """Visual progress for a pipeline whose steps are decided at runtime."""

    BAR_WIDTH = 30

    def __init__(self, steps: list):
        """steps: list of (name, emoji), in execution order."""
        self.steps = steps
        self.total = len(steps)
        self.current = 0
        self.start_time = time.time()
        self.step_times = []
        self.step_start = None

    def _bar(self, done: int) -> str:
        filled = int(self.BAR_WIDTH * done / self.total)
        return "\u2588" * filled + "\u2591" * (self.BAR_WIDTH - filled)

    def _fmt_duration(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m{secs}s"

    def start_step(self):
        self.current += 1
        self.step_start = time.time()
        name, emoji = self.steps[self.current - 1]
        bar = self._bar(self.current - 1)
        elapsed = self._fmt_duration(time.time() - self.start_time)
        print(f"\n{bar}  Step {self.current}/{self.total} \u2014 {emoji}  {name}  (elapsed: {elapsed})")

    def end_step(self, note: str = ""):
        duration = time.time() - self.step_start
        self.step_times.append(duration)
        bar = self._bar(self.current)
        msg = f"{bar}  \u2713 Step {self.current}/{self.total} done in {self._fmt_duration(duration)}"
        if note:
            msg += f"  ({note})"
        print(msg)

    def finish(self):
        total = time.time() - self.start_time
        bar = "\u2588" * self.BAR_WIDTH
        print(f"\n{bar}  \u2713 Pipeline complete in {self._fmt_duration(total)}")
        breakdown = " | ".join(
            f"{self.steps[i][0]}: {self._fmt_duration(t)}"
            for i, t in enumerate(self.step_times)
        )
        print(f"   {breakdown}\n")


@contextmanager
def step(progress: ProgressTracker):
    progress.start_step()
    note_holder = {"note": ""}

    def set_note(text: str):
        note_holder["note"] = text

    yield set_note
    progress.end_step(note_holder["note"])


def transcribe(audio_path: Path, set_note):
    print(f"   Whisper: {gpu.describe(CHOICE)}")
    print(f"   Loading WhisperX...")
    model = whisperx.load_model(
        MODEL, device=DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE,
        asr_options={"condition_on_previous_text": False},
    )
    print(f"   Transcribing {audio_path.name}...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE,
                              chunk_size=CHUNK_SIZE)

    del model
    if CHOICE.is_cuda:
        torch.cuda.empty_cache()

    segments, dropped = clean_segments(result["segments"])
    if dropped:
        secs = sum(d for _, _, d, _ in dropped)
        by_reason = {}
        for reason, _, _, _ in dropped:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        detail = ", ".join(f"{n} {r}" for r, n in sorted(by_reason.items()))
        print(f"   Filtered {len(dropped)} hallucinated segments "
              f"({secs / 60:.1f} min): {detail}")

    if not segments:
        raise PermanentFailure(
            f"Every segment in {audio_path.name} was filtered as a hallucination "
            f"({len(dropped)} dropped) \u2014 the audio is likely silence, music, or "
            f"too quiet to transcribe. Nothing to summarize."
        )

    set_note(f"{len(segments)} segments, {len(dropped)} filtered")
    return segments


def diarize(audio_path: Path, segments: list, hf_token: str, set_note,
            speaker_hints: dict = None) -> list:
    from pyannote.audio import Pipeline

    print(f"   Converting audio to clean WAV...")
    clean_wav = Path(tempfile.gettempdir()) / f"{audio_path.stem}_pyannote.wav"
    try:
        subprocess.run([
            # -nostdin: under watcher.sh our stdin is the inbox file list, and
            # ffmpeg reads keypresses from it, eating bytes off the next path.
            "ffmpeg", "-nostdin", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(clean_wav),
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "replace").strip().splitlines()
        tail = "\n".join(stderr[-8:])
        raise RuntimeError(f"ffmpeg failed ({e.returncode}):\n{tail}") from None

    print(f"   Loading pyannote pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device(DEVICE))
    
    hints = speaker_hints or {}
    if hints:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(hints.items()))
        print(f"   Running diarization ({detail})...")
    else:
        print(f"   Running diarization...")
    diarization = pipeline(str(clean_wav), **hints)
    clean_wav.unlink(missing_ok=True)

    annotation = getattr(diarization, "speaker_diarization", diarization)
    speaker_turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
        if turn.end - turn.start > MIN_TURN_DURATION
    ]
    n_speakers = len(set(s for _, _, s in speaker_turns))
    if n_speakers < 2 and not hints:
        print(f"   Note: only 1 speaker detected. If this recording has more, "
              f"re-run with --speakers N (pyannote under-clusters similar or "
              f"quiet voices).")

    for seg in segments:
        s_start, s_end = seg["start"], seg["end"]
        max_overlap = 0
        assigned = "UNKNOWN"
        for sp_start, sp_end, sp in speaker_turns:
            overlap = max(0, min(s_end, sp_end) - max(s_start, sp_start))
            if overlap > max_overlap:
                max_overlap = overlap
                assigned = sp
        seg["speaker"] = assigned

    del pipeline
    if CHOICE.is_cuda:
        torch.cuda.empty_cache()

    set_note(f"{n_speakers} speakers")
    return segments


def write_transcript(audio_path: Path, segments: list) -> Path:
    clean_stem = audio_path.stem.replace(".", "-")
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out = TRANSCRIPTS_DIR / f"{clean_stem}.md"

    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {audio_path.stem}\n\n")
        prev_speaker = None
        last_header_at = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg["start"]
            text = seg["text"].strip()

            speaker_changed = speaker != prev_speaker
            stale = (
                last_header_at is not None
                and start - last_header_at >= TIMESTAMP_INTERVAL
            )
            if speaker_changed or stale:
                f.write(f"\n**[{fmt_ts(start)}] [{speaker}]**\n")
                prev_speaker = speaker
                last_header_at = start
            f.write(f"{text} ")

    return out


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _user_content(transcript_text: str) -> str:
    # Long document first, instructions last -- adherence is better that way,
    # and the run-specific request should be the last thing read.
    content = f"Berikut transkrip, ringkas sesuai instruksi:\n\n{transcript_text}"
    if EXTRA_PROMPT:
        content += (f"\n\nKonteks & permintaan tambahan dari {USER_NAME} untuk ringkasan "
                    f"ini:\n{EXTRA_PROMPT}")
    return content


def _retry_delay(error) -> float | None:
    """The wait a 429 asks for (a RetryInfo detail like "retryDelay": "34s"), if any."""
    try:
        for detail in error.details["error"]["details"]:
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None


def _generate(client, content: str):
    """One streamed summary call. Returns (text, usage_metadata)."""
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=MAX_SUMMARY_TOKENS,
    )
    started = time.time()
    chars = [0]
    done = threading.Event()

    def tick():
        # The model thinks before the first chunk and sends no text meanwhile, so
        # the counter sits at 0 for a while -- the elapsed clock is what says it
        # is alive.
        while not done.wait(1.0):
            print(f"\r   Streaming: {chars[0]} chars, "
                  f"{time.time() - started:.0f}s elapsed...", end="", flush=True)

    ticker = threading.Thread(target=tick, daemon=True)
    ticker.start()
    parts, usage, finish, block = [], None, None, None
    try:
        for chunk in client.models.generate_content_stream(
                model=GEMINI_MODEL, contents=content, config=config):
            if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                block = chunk.prompt_feedback.block_reason
            if chunk.usage_metadata:
                usage = chunk.usage_metadata
            if chunk.candidates and chunk.candidates[0].finish_reason:
                finish = chunk.candidates[0].finish_reason
            if chunk.text:
                parts.append(chunk.text)
                chars[0] += len(chunk.text)
    finally:
        done.set()
        ticker.join()
    print(f"\r   Received {chars[0]} chars in {time.time() - started:.0f}s."
          f"{' ' * 20}")

    if block:
        raise PermanentFailure(
            f"{GEMINI_MODEL} blocked this transcript as a prompt "
            f"(block_reason={block.name}).")
    if finish == types.FinishReason.MAX_TOKENS:
        raise PermanentFailure(
            f"{GEMINI_MODEL} hit the {MAX_SUMMARY_TOKENS}-token cap (thinking + "
            f"summary share it) and was truncated. Try a different "
            f"SUPERMEET_GEMINI_MODEL, then drop the file back in the inbox -- the "
            f"transcript is cached, so it skips straight to summarizing.")
    if finish is not None and finish != types.FinishReason.STOP:
        # SAFETY, PROHIBITED_CONTENT, RECITATION, ...: whatever text came back is
        # cut short or withheld, and would land in the archived PDF as if whole.
        raise PermanentFailure(
            f"{GEMINI_MODEL} stopped early and would not finish the summary "
            f"(finish_reason={finish.name}).")
    text = "".join(parts)
    if not text.strip():
        raise PermanentFailure(f"{GEMINI_MODEL} returned no text.")
    return text, usage


def summarize(transcript_path: Path, set_note) -> str:
    transcript_text = transcript_path.read_text(encoding="utf-8")

    print(f"   Calling {GEMINI_MODEL}...")
    if EXTRA_PROMPT:
        print(f"   Konteks tambahan: {EXTRA_PROMPT[:70]}")

    # Streamed, not a single blocking POST. A whole-transcript summary with
    # thinking runs for many minutes, and an idle HTTPS connection that long
    # gets dropped somewhere between here and the API after the work is
    # already done. Streaming keeps bytes moving, so the connection stays alive
    # and we can show that it is progressing.
    client = genai.Client(
        api_key=_api_key(),
        http_options=types.HttpOptions(timeout=int(API_TIMEOUT * 1000)),
    )
    content = _user_content(transcript_text)

    for attempt in range(len(API_RETRY_WAITS) + 1):
        try:
            text, usage = _generate(client, content)
            break
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            quota = e.code == 429
            if not (quota or e.code >= 500) or attempt == len(API_RETRY_WAITS):
                raise
            wait_s = min(_retry_delay(e) or API_RETRY_WAITS[attempt], MAX_RETRY_WAIT)
            why = ("kuota free tier kena limit (429)" if quota
                   else f"Gemini lagi overload ({e.code})")
            print(f"\n   {why}; nunggu {wait_s:.0f}s lalu coba lagi "
                  f"({attempt + 1}/{len(API_RETRY_WAITS)})...", flush=True)
            time.sleep(wait_s)

    tokens_in = (usage.prompt_token_count or 0) if usage else 0
    # Thinking tokens count as output (and against the cap) but are reported apart.
    tokens_out = ((usage.candidates_token_count or 0)
                  + (usage.thoughts_token_count or 0)) if usage else 0
    set_note(f"{tokens_in} in + {tokens_out} out, free tier")
    return text


def make_summary_pdf(summary_text: str, base_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / f"{base_name}_summary.md"
    pdf_path = OUTPUT_DIR / f"{base_name}_summary.pdf"

    nice_title = base_name.replace("_", " ").replace("-", " ").title()
    md_path.write_text(f"# {nice_title}\n\n{summary_text}", encoding="utf-8")

    return render_pdf(md_path, pdf_path, toc=True)


def make_transcript_pdf(transcript_md: Path, base_name: str) -> Path:
    TRANSCRIPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # The markdown goes next to the PDF: it is what gets handed to Claude (Cowork)
    # for summarising by hand -- plain text, nothing lost to PDF extraction.
    shutil.copyfile(transcript_md, TRANSCRIPT_OUTPUT_DIR / f"{base_name}_transcript.md")
    pdf_path = TRANSCRIPT_OUTPUT_DIR / f"{base_name}_transcript.pdf"
    return render_pdf(transcript_md, pdf_path, toc=False)


def main(argv: list[str] | None = None):
    # Before any model loads, so their startup chatter never lands in the
    # middle of a progress bar.
    silence_model_noise()
    argv = sys.argv[1:] if argv is None else argv
    args, flags, speaker_hints = [], set(), {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--speakers", "--min-speakers", "--max-speakers"):
            key = {"--speakers": "num_speakers",
                   "--min-speakers": "min_speakers",
                   "--max-speakers": "max_speakers"}[a]
            if i + 1 >= len(argv) or not argv[i + 1].isdigit():
                print(f"ERROR: {a} needs a number", file=sys.stderr)
                sys.exit(EXIT_PERMANENT)
            speaker_hints[key] = int(argv[i + 1])
            i += 2
            continue
        if a.startswith("--"):
            flags.add(a)
        else:
            args.append(a)
        i += 1
    force = "--force" in flags
    transcribe_only = TRANSCRIBE_ONLY or "--transcribe-only" in flags

    if not args:
        print("Usage: supermeet.py transcribe <audio_file> [options]", file=sys.stderr)
        print("  --force            re-transcribe even if a cached transcript exists",
              file=sys.stderr)
        print("  --speakers N       tell pyannote exactly how many speakers to expect",
              file=sys.stderr)
        print("  --min-speakers N   lower bound on speaker count", file=sys.stderr)
        print("  --max-speakers N   upper bound on speaker count", file=sys.stderr)
        print("  --transcribe-only  skip the Gemini summary; transcript PDF only",
              file=sys.stderr)
        sys.exit(EXIT_PERMANENT)
    if "num_speakers" in speaker_hints and len(speaker_hints) > 1:
        print("ERROR: --speakers cannot be combined with --min/--max-speakers",
              file=sys.stderr)
        sys.exit(EXIT_PERMANENT)
    if not transcribe_only and not _api_key():
        print("ERROR: GEMINI_API_KEY not set", file=sys.stderr)
        sys.exit(EXIT_PERMANENT)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN not set \u2014 diarization akan di-skip", file=sys.stderr)

    audio = Path(args[0]).expanduser().resolve()
    if not audio.exists():
        print(f"File not found: {audio}", file=sys.stderr)
        sys.exit(EXIT_PERMANENT)

    # Resume: transcription is the expensive step, so reuse it when it already ran.
    clean_stem = audio.stem.replace(".", "-")
    cached_transcript = TRANSCRIPTS_DIR / f"{clean_stem}.md"
    resuming = cached_transcript.exists() and not force and not speaker_hints
    if speaker_hints and cached_transcript.exists() and not force:
        print("  Speaker hints given \u2014 re-running transcription and diarization.")

    print(f"\n{'\u2550' * 50}")
    print(f"  Processing: {audio.name}")
    if transcribe_only:
        print(f"  Transcribe only — tanpa ringkasan")
    if resuming:
        print(f"  Resuming from cached transcript ({cached_transcript.name})")
        print(f"  Pass --force to re-transcribe from audio.")
    print(f"{'\u2550' * 50}")

    steps = []
    if not resuming:
        steps.append(("Transcribe (WhisperX)", "\U0001f3a4"))
        if hf_token:
            steps.append(("Diarize (pyannote)", "\U0001f465"))
    if transcribe_only:
        steps.append(("Generate PDF", "\U0001f4c4"))
    else:
        steps.append((f"Summarize ({GEMINI_MODEL})", "\U0001f9e0"))
        steps.append(("Generate PDFs", "\U0001f4c4"))
    progress = ProgressTracker(steps)

    if resuming:
        transcript_path = cached_transcript
    else:
        # One 6GB card, two pipelines. Held only across the model-loading steps;
        # summarising and rendering need no GPU, so the realtime server can have
        # the card back while this run finishes.
        def announce(who):
            name = (who or {}).get("name", "another process")
            print(f"   GPU dipakai {name} — nunggu sampai bebas...", flush=True)

        try:
            with gpu.lease("batch", wait=True, timeout=GPU_WAIT_TIMEOUT,
                           on_wait=announce):
                with step(progress) as set_note:
                    segments = transcribe(audio, set_note)

                if hf_token:
                    with step(progress) as set_note:
                        segments = diarize(audio, segments, hf_token, set_note,
                                           speaker_hints)
        except gpu.GPUBusy as e:
            print(f"\nERROR: {e}. Coba lagi nanti, atau matiin yang megang.",
                  file=sys.stderr)
            return EXIT_BUSY
        except PermanentFailure as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return EXIT_PERMANENT

        transcript_path = write_transcript(audio, segments)

    if not transcribe_only:
        try:
            with step(progress) as set_note:
                summary = summarize(transcript_path, set_note)
        except PermanentFailure as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            return EXIT_PERMANENT
        except (genai_errors.ClientError, genai_errors.ServerError) as e:
            detail = f"{type(e).__name__} {e.code}: {e.message or e}"
            if e.code == 429 or e.code >= 500:
                # summarize() already retried; a quota that is still exhausted
                # (the free tier's per-minute or per-day limit) or an overload
                # that outlasts it clears with time, so the watcher gets another go.
                print(f"\nERROR: {detail}\n   Kuota free tier Gemini habis atau "
                      f"server lagi sibuk -- nanti dicoba lagi.", file=sys.stderr)
                return 1
            # The request itself is wrong -- a bad key, an unknown model, an
            # invalid parameter. Sending it again cannot succeed.
            print(f"\nERROR: {detail}", file=sys.stderr)
            return EXIT_PERMANENT

    base_name = audio.stem.replace(".", "-")
    with step(progress) as set_note:
        print(f"   Rendering transcript PDF...")
        transcript_pdf = make_transcript_pdf(transcript_path, base_name)
        note = f"transcript {transcript_pdf.stat().st_size / 1024:.0f}KB"
        if not transcribe_only:
            print(f"   Rendering summary PDF...")
            summary_pdf = make_summary_pdf(summary, base_name)
            note += f" \u00b7 summary {summary_pdf.stat().st_size / 1024:.0f}KB"
        set_note(note)

    progress.finish()
    print(f"  \U0001f4c2 Transcript: {transcript_pdf}")
    if not transcribe_only:
        print(f"  \U0001f4c2 Summary:    {summary_pdf}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
