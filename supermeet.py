#!/usr/bin/env python3
"""supermeet — one entry point for both halves of the meeting pipeline.

    supermeet.py serve [--https] [--port N]   realtime copilot, during a meeting
    supermeet.py transcribe <audio> [...]     batch archival pass, after one
    supermeet.py watch                        watch the cloud inbox, transcribe
    supermeet.py pdf <file.md> [out_dir]      render a markdown file to PDF
    supermeet.py kuliah <folder> [--dry-run]  transcribe lecture recordings, grouped per meeting
    supermeet.py gpu                          who holds the GPU lease
    supermeet.py doctor [--online]            check setup: tools, packages, keys, folders

The two pipelines share core/: paths, hallucination filters, PDF rendering, and
the GPU lease that keeps them from loading models on top of each other.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _dyld_fallback() -> "str | None":
    """DYLD_FALLBACK_LIBRARY_PATH to hand to the process on macOS, or None.

    weasyprint loads pango/cairo/glib through dlopen, and on Apple Silicon
    Homebrew keeps those in /opt/homebrew/lib, which dyld does not search
    (Intel Homebrew uses /usr/local/lib, which it does). Without this the PDF
    step dies with "cannot load library 'libpango-1.0-0'". dyld reads the
    variable when a process starts, so it cannot be set from inside the process
    that needs it -- hence the re-exec below. The dyld defaults are kept in the
    value, because setting the variable replaces them.
    """
    if sys.platform != "darwin" or os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"):
        return None
    prefix = os.environ.get("HOMEBREW_PREFIX") or next(
        (p for p in ("/opt/homebrew", "/usr/local")
         if os.path.exists(f"{p}/bin/brew")), "")
    if not prefix or prefix == "/usr/local" or not os.path.isdir(f"{prefix}/lib"):
        return None
    home = os.path.expanduser("~")
    return f"{prefix}/lib:{home}/lib:/usr/local/lib:/usr/lib"


# Re-exec under the project venv if we were started by some other interpreter.
# The shebang finds the system python, which has none of the dependencies, and
# a single entry point should not require remembering to spell out .venv/bin.
# Resolved relative to this file, so it survives the directory being moved.
# The same hop is used to start macOS with the library path above: SIP strips
# DYLD_* from the environment whenever a process passes through /bin/bash or
# /usr/bin/env, so batch/watcher.sh cannot hand it down; each python entry
# re-adds it (it is a no-op once set, so this cannot loop).
_VENV_PY = ROOT / ".venv" / "bin" / "python"
_use_venv = _VENV_PY.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve()
_dyld = _dyld_fallback()
if _dyld:
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = _dyld
if _use_venv or _dyld:
    _py = str(_VENV_PY) if _use_venv else sys.executable
    os.execve(_py, [_py, str(Path(__file__).resolve()), *sys.argv[1:]], dict(os.environ))

sys.path.insert(0, str(ROOT))

# Load .env before anything reads os.environ (cmd_watch copies it, the pipelines
# read their keys from it). Real environment variables still win over the file.
from core import env  # noqa: E402
env.load()

USAGE = __doc__.split("\n\n")[1]

# Extra summarising context, asked by `watch` and read by batch/pipeline.py.
# Also honoured if exported by hand, which is how `transcribe` gets one.
EXTRA_PROMPT_VAR = "SUPERMEET_EXTRA_PROMPT"

# Summariser options offered by `watch`, default first. Gemini free tier: no
# dollar rate, but it has per-minute and per-day request limits (a 429 makes the
# watcher retry later). Model ids are from ai.google.dev/gemini-api/docs/models;
# the first row must match GEMINI_MODEL's default in batch/pipeline.py.
SUMMARISERS = [
    ("gemini-3.8-flash",      "Gemini 3.8 Flash",      "$0  free tier, ada limit harian"),
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", "$0  free tier, lebih ringan dan cepat"),
]

# Offered after the summarisers, so Enter still means the first Gemini row. Picking
# it exports SUPERMEET_TRANSCRIBE_ONLY: batch/pipeline.py then skips Gemini entirely.
TRANSCRIBE_ONLY_VAR = "SUPERMEET_TRANSCRIBE_ONLY"
TRANSCRIBE_ONLY_ROW = ("transcribe-only", "Transcribe only",
                       "$0  tanpa ringkasan, cuma PDF transkrip")

# Per-recording-type emphasis, appended after the transcript as the last thing
# the model reads. These do not replace the structure in batch/pipeline.py's system
# prompt -- they say what to dig harder for inside it. Add a case by adding a row.
SEMINAR_PROMPT = """Ini rekaman seminar/webinar. Selain struktur standar:
- Pisahin jelas mana materi dari pembicara dan mana yang muncul dari sesi tanya-jawab.
- Tangkap framework, model, atau kerangka berpikir yang diajarin, lengkap sama \
contoh kasus yang dipakai buat ngejelasinnya.
- Catat angka, data, dan studi kasus konkret yang disebut — itu yang paling gampang ilang.
- Kumpulin semua referensi yang disebut (buku, tools, orang, report, link) di satu tempat.
- Catat identitas & kredensial pembicara, plus info acara lanjutan, komunitas, atau \
kontak yang dibagiin.
- Di sesi QnA, tulis pertanyaannya sekalian jawabannya — bukan cuma jawabannya."""

KULIAH_PROMPT = """Ini rekaman kuliah. Selain struktur standar:
- Prioritas nomor satu: apa pun yang nyerempet penilaian — kisi-kisi, 'ini bakal keluar \
di ujian', tugas, PR, deadline, format pengumpulan. Jangan sampai kelewat satu pun.
- Bikin daftar istilah: tiap istilah teknis, definisi, atau rumus yang dijelasin, pakai \
bahasa dosennya sendiri.
- Tandain hal yang dosen ulang, tekanin, atau bilang 'penting' — itu sinyal materi ujian.
- Contoh soal atau hitungan yang dikerjain di kelas: tulis langkahnya, bukan cuma hasilnya.
- Catat bacaan, bab, atau sumber yang disuruh baca.
- Kalau ada yang digantung ('nanti dibahas minggu depan'), masukin ke open threads."""

SAHAM_PROMPT = """Ini diskusi saham/market. Selain struktur standar:
- Tiap emiten atau instrumen yang disebut: tulis kodenya, tesisnya \
(bullish/bearish/netral), plus alasan yang dikasih.
- Angka itu intinya: harga, valuasi (PER, PBV, EV/EBITDA), target price, \
support/resistance, dividend yield, proyeksi growth. Kutip persis, jangan dibuletin.
- Pisahin fakta (laporan keuangan, berita, data) dari opini dan tebakan pembicara.
- Catat katalis sekalian timeframe-nya — kapan sesuatu diharapin kejadian.
- Risiko dan bear case yang disebut, sependek apa pun, tetap tulis.
- Kalau ada yang nyebut posisi atau rencana aksi (beli, jual, nunggu, cut loss), \
catat siapa yang bilang dan syaratnya apa.
- Sebutin asumsi yang dipakai buat nyampe ke kesimpulan, bukan cuma kesimpulannya."""

RAPAT_PROMPT = """Ini rekaman rapat kerja. Selain struktur standar:
- Keputusan yang diambil: apa keputusannya, siapa yang mutusin, alasannya — pisahin \
dari yang masih wacana.
- Tiap action item wajib ada PIC dan deadline kalau kesebut; kalau nggak kesebut, \
tulis '(PIC belum ditentuin)' biar kelihatan bolongnya.
- Catat ketidaksepakatan dan trade-off yang dibahas, termasuk opsi yang ditolak dan kenapa.
- Blocker dan dependency: siapa nunggu apa dari siapa.
- Hal yang ditunda atau nggak sempat dibahas, masukin ke open threads.
- Kalau ada angka (budget, timeline, target, kapasitas), kutip persis."""

CONTEXTS = [
    ("Tanpa penekanan khusus", ""),
    ("Seminar / webinar (online atau offline)", SEMINAR_PROMPT),
    ("Kuliah", KULIAH_PROMPT),
    ("Diskusi saham / market", SAHAM_PROMPT),
    ("Rapat kerja / meeting umum", RAPAT_PROMPT),
    ("Tulis sendiri...", None),
]


def _run_out(*cmd: str) -> str:
    """stdout of a command, or "" if it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lan_ips() -> list[str]:
    """IPv4 addresses other devices can reach this machine on, best guess first.

    `hostname -I` on Linux/WSL; on macOS (BSD hostname has no -I) the Wi-Fi/
    Ethernet interfaces via ipconfig, then ifconfig, then the address the OS
    would use to reach the internet.
    """
    def ipv4(words: list[str]) -> list[str]:
        out = []
        for w in words:
            parts = w.split(".")
            if (len(parts) == 4 and all(x.isdigit() and int(x) < 256 for x in parts)
                    and not w.startswith(("0.", "127.", "169.254.")) and w not in out):
                out.append(w)
        return out

    found = ipv4(_run_out("hostname", "-I").split())
    if not found and sys.platform == "darwin":
        found = ipv4([_run_out("ipconfig", "getifaddr", f"en{i}").strip() for i in range(10)])
    if not found:
        words = [line.split()[1].removeprefix("addr:")
                 for line in _run_out("ifconfig").splitlines()
                 if line.strip().startswith("inet ") and len(line.split()) > 1]
        found = ipv4(words)
    if not found:
        import socket
        try:    # a UDP "connect" sends nothing; it only makes the OS pick a route
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("10.255.255.255", 1))
                found = ipv4([sock.getsockname()[0]])
        except OSError:
            pass
    return found


def cmd_serve(argv: list[str]) -> int:
    from realtime import config

    port = config.PORT
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    https = "--https" in argv

    args = [
        sys.executable, "-m", "uvicorn", "realtime.app:app",
        "--host", config.HOST, "--port", str(port), "--log-level", "warning",
        # Loading the STT model starves the event loop for ~15s, which outlasts
        # uvicorn's default 20s ping timeout. Be generous or viewers get dropped
        # mid-start.
        "--ws-ping-interval", "30", "--ws-ping-timeout", "90",
    ]
    scheme = "http"
    if https:
        subprocess.run([str(ROOT / "make-cert.sh")], check=True)
        args += ["--ssl-keyfile", str(ROOT / "certs/key.pem"),
                 "--ssl-certfile", str(ROOT / "certs/cert.pem")]
        scheme = "https"

    lan = lan_ips()
    print(f"  local : {scheme}://localhost:{port}")
    if lan:
        print(f"  phone : {scheme}://{lan[0]}:{port}")
        if https:
            print(f"  iPad  : {scheme}://{lan[0]}:{port}/ca.crt  sekali dulu, lihat README")
    print()
    return subprocess.call(args, cwd=str(ROOT))


def cmd_transcribe(argv: list[str]) -> int:
    from batch import pipeline
    return pipeline.main(argv) or 0


def cmd_kuliah(argv: list[str]) -> int:
    from kuliah import run
    return run.main(argv)


def cmd_watch(argv: list[str]) -> int:
    """Ask once for extra summarising context, then watch the inbox with it set.

    The watcher is long-lived and every file it picks up goes through the same
    Gemini call, so the answer applies to the whole run -- say so when asking.
    Passed down as an env var: watcher.sh re-invokes this script per file, and
    the env survives both hops without threading an argument through bash.
    """
    env = os.environ.copy()
    # Skip both questions when there is nobody to answer them (nohup, systemd,
    # cron) or when the caller already set the variables deliberately.
    if sys.stdin.isatty():
        transcribe_only = env.get(TRANSCRIBE_ONLY_VAR, "") not in ("", "0")
        if not env.get("SUPERMEET_GEMINI_MODEL") and not transcribe_only:
            print("Model buat ringkasan run ini:")
            options = [*SUMMARISERS, TRANSCRIBE_ONLY_ROW]
            for i, (_, label, rate) in enumerate(options, 1):
                print(f"  {i}. {label}   {rate}")
            model = _ask_choice(options)
            if model == TRANSCRIBE_ONLY_ROW[0]:
                env[TRANSCRIBE_ONLY_VAR], transcribe_only = "1", True
            else:
                env["SUPERMEET_GEMINI_MODEL"] = model
            print()
        # Without a summary there is nothing for a recording-type emphasis to shape.
        if not env.get(EXTRA_PROMPT_VAR) and not transcribe_only:
            print("Jenis rekaman (nentuin penekanan ringkasan):")
            for i, (label, _) in enumerate(CONTEXTS, 1):
                print(f"  {i}. {label}")
            extra = _ask_context()
            if extra:
                env[EXTRA_PROMPT_VAR] = extra
            print()
    return subprocess.call([str(ROOT / "batch/watcher.sh"), *argv], env=env)


def _ask_line() -> str:
    """One line of input, treating Ctrl-D / Ctrl-C as an empty answer."""
    try:
        return input("  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _ask_context() -> str:
    """Pick a preset, or type a one-off note. Empty answer means no emphasis."""
    while True:
        raw = _ask_line()
        if not raw:
            return ""
        if raw.isdigit() and 1 <= int(raw) <= len(CONTEXTS):
            _, preset = CONTEXTS[int(raw) - 1]
            if preset is not None:
                return preset
            print("  Tulis konteksnya (contoh: 'ini rapat vendor, catat komitmen harga'):")
            return _ask_line()
        # Anything else is taken as the note itself -- typing it straight is
        # faster than picking the last option first.
        return raw


def _ask_choice(options: list) -> str:
    """Re-ask until the answer is in range. Empty answer takes the first option."""
    while True:
        raw = _ask_line()
        if not raw:
            return options[0][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print(f"  Pilih 1-{len(options)}, atau Enter buat yang pertama.")


def cmd_pdf(argv: list[str]) -> int:
    """Standalone markdown -> PDF, for notes that never went through a pipeline."""
    from core.config import OUTPUT_DIR
    from core.pdf import render_pdf

    if not argv:
        print("Usage: supermeet.py pdf <file.md> [output_dir]", file=sys.stderr)
        return 1
    md = Path(argv[0]).expanduser().resolve()
    if not md.exists():
        print(f"File not found: {md}", file=sys.stderr)
        return 1
    out_dir = Path(argv[1]).expanduser().resolve() if len(argv) > 1 else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = render_pdf(md, out_dir / f"{md.stem}.pdf", toc=True, title=md.stem)
    print(f"  {pdf}  ({pdf.stat().st_size / 1024:.0f}KB)")
    return 0


def cmd_gpu(argv: list[str]) -> int:
    from core import gpu

    who = gpu.holder()
    if not who:
        print("  GPU lease: bebas")
    else:
        import time
        held = time.time() - who.get("since", time.time())
        print(f"  GPU lease: {who.get('name')} (pid {who.get('pid')}), {held:.0f}s")
    print(f"  lockfile : {gpu.LOCK_PATH}")
    return 0


def cmd_doctor(argv: list[str]) -> int:
    from core import doctor
    return doctor.main(argv)


COMMANDS = {
    "serve": cmd_serve,
    "transcribe": cmd_transcribe,
    "watch": cmd_watch,
    "pdf": cmd_pdf,
    "kuliah": cmd_kuliah,
    "gpu": cmd_gpu,
    "doctor": cmd_doctor,
}


def main() -> int:
    # Progress output is usually being watched live through a redirect -- the
    # watcher's log, or a handoff subprocess. Block buffering makes a run that
    # is waiting on the GPU lease look like a hang.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(USAGE)
        return 0 if len(sys.argv) > 1 else 1
    name, *rest = sys.argv[1:]
    handler = COMMANDS.get(name)
    if handler is None:
        print(f"Unknown command: {name}\n", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 1
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())
