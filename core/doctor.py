"""`supermeet.py doctor` — a checklist of what is installed, configured and reachable.

Nothing heavy is imported: packages are probed with find_spec, and the one check
that needs torch (is CUDA usable) runs in a subprocess with a timeout, so a broken
GPU driver cannot hang or crash the doctor. Which device and Whisper model the
batch and kuliah runs will use comes from core.device, so the doctor and the
pipelines cannot disagree. On a Mac it also checks Homebrew, pango and the
library path weasyprint needs. With --online it also makes one tiny
real Gemini call, to prove the key and model work. Key values are never printed.
Exit status is 1 if any line is a red cross, so it can gate scripts.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OK, WARN, FAIL, INFO = "✅", "⚠️ ", "❌", "ℹ️ "

MACOS = sys.platform == "darwin"
NO_CUDA = "torch nggak lihat GPU CUDA"   # the one not-a-problem reason in cuda_status()

# (import name, pip name, who needs it)
PACKAGES = [
    ("whisperx", "whisperx", "batch"),
    ("torch", "torch", "batch + realtime"),
    ("pyannote.audio", "pyannote.audio", "batch, diarization"),
    ("weasyprint", "weasyprint", "batch, PDF"),
    ("google.genai", "google-genai", "batch, summary"),
    ("anthropic", "anthropic", "realtime"),
    ("faster_whisper", "faster-whisper", "realtime"),
    ("fastapi", "fastapi", "realtime"),
]


def mask(secret: str) -> str:
    return f"…{secret[-4:]}" if len(secret) > 8 else "…"


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # find_spec imports parent packages ("google" for "google.genai").
        return False


def default_model() -> str:
    """The summariser model, without importing batch/pipeline.py (whisperx, torch...)."""
    if os.environ.get("SUPERMEET_GEMINI_MODEL"):
        return os.environ["SUPERMEET_GEMINI_MODEL"]
    try:
        src = (ROOT / "batch" / "pipeline.py").read_text(encoding="utf-8")
        m = re.search(r'^GEMINI_MODEL\s*=.*?"SUPERMEET_GEMINI_MODEL",\s*"([^"]+)"', src, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return "gemini-3.8-flash"


def gemini_key() -> "tuple[str, str] | None":
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(var):
            return var, os.environ[var]
    return None


def cuda_status() -> "tuple[bool, str] | None":
    """(usable, detail) from a throwaway interpreter, or None if torch is absent."""
    if not has_module("torch"):
        return None
    code = ("import torch\n"
            "ok = torch.cuda.is_available()\n"
            "print(int(ok))\n"
            "if ok:\n"
            "    p = torch.cuda.get_device_properties(0)\n"
            "    print(p.name, round(p.total_memory / 2**30, 1))\n")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=90)
    except subprocess.TimeoutExpired:
        return False, "torch nggak jawab dalam 90 detik"
    lines = out.stdout.split("\n")
    if out.returncode != 0 or not lines or lines[0] not in ("0", "1"):
        return False, "torch gagal di-import"
    if lines[0] == "0":
        return False, NO_CUDA
    name, _, gib = lines[1].rpartition(" ")
    return True, f"{name}, {gib} GB VRAM"


def check_gemini_online(key: str, model: str) -> "tuple[str, str]":
    """One 5-token call. Returns (icon, message)."""
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        return FAIL, "google-genai belum ke-install"
    try:
        client = genai.Client(api_key=key)
        client.models.generate_content(
            model=model, contents="Balas satu kata: ok",
            config=types.GenerateContentConfig(max_output_tokens=5))
    except errors.APIError as e:
        code = getattr(e, "code", None)
        if code in (400, 403):
            return FAIL, f"key salah atau nggak punya akses ({code})"
        if code == 404:
            return FAIL, f"model nggak ada: {model} (404)"
        if code == 429:
            return WARN, "kuota habis / kena limit (429), key-nya sendiri valid"
        return FAIL, f"Gemini error {code}"
    except Exception as e:  # network down, DNS, proxy...
        return FAIL, f"nggak bisa nyambung ke Gemini ({type(e).__name__})"
    return OK, f"{model} jawab, key valid"


def install_hint(tool: str) -> str:
    return f"brew install {tool}" if MACOS else f"sudo apt install {tool}   (macOS: brew install {tool})"


def device_choice(cuda_usable: bool):
    """core.device.pick() as if torch had reported `cuda_usable`.

    pick() asks torch in-process; the doctor already asked in a subprocess (safe
    against a broken driver), so feed it that answer instead of importing torch here.
    """
    from core import device
    real = device.cuda_available
    device.cuda_available = lambda: cuda_usable
    try:
        return device.pick()
    finally:
        device.cuda_available = real


def brew_prefix() -> "str | None":
    """Homebrew's prefix (/opt/homebrew on Apple Silicon, /usr/local on Intel)."""
    prefix = os.environ.get("HOMEBREW_PREFIX")
    if prefix and Path(prefix).is_dir():
        return prefix
    found = shutil.which("brew")
    if found:
        return str(Path(found).parent.parent)   # not resolve(): /usr/local/bin/brew is a symlink
    for candidate in ("/opt/homebrew", "/usr/local"):
        if (Path(candidate) / "bin" / "brew").exists():
            return candidate
    return None


def pango_installed(prefix: str) -> bool:
    return any((Path(prefix) / "lib").glob("libpango-1.0*.dylib"))


def dyld_status(prefix: str, value: "str | None") -> "tuple[str, str, str]":
    """(icon, detail, hint) for DYLD_FALLBACK_LIBRARY_PATH on macOS.

    weasyprint finds pango with dlopen. Intel Homebrew's /usr/local/lib is in
    dyld's default search; Apple Silicon's /opt/homebrew/lib is not, so the
    variable has to include it. supermeet.py sets it on start (dyld only reads
    it at process start, so it cannot be set later from inside python).
    """
    lib = f"{prefix}/lib"
    manual = f'export DYLD_FALLBACK_LIBRARY_PATH="{lib}"'
    if prefix == "/usr/local":
        return OK, "nggak perlu (Homebrew di /usr/local)", ""
    if value and lib in value.split(":"):
        return OK, f"ke-set ({lib})", ""
    if value:
        return WARN, f"ke-set tapi nggak ada {lib}", f"tambahin {lib} ke isinya, atau: {manual}"
    return (WARN, "belum di-set",
            f"jalan lewat ./supermeet.py nge-set sendiri; kalau pakai python langsung: {manual}")


def main(argv: list[str]) -> int:
    online = "--online" in argv
    results = []

    def line(icon: str, label: str, detail: str = "", hint: str = "") -> None:
        results.append(icon)
        print(f"{icon} {label:<22}{detail}")
        if hint:
            print(f"   {'':<22}{hint}")

    print("supermeet doctor\n")

    v = sys.version_info
    ver = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 10):
        line(OK, "python", ver)
    else:
        line(FAIL, "python", f"{ver}, butuh 3.10 ke atas")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if not in_venv:
        line(WARN, "venv", "nggak lagi di venv", "jalanin ./setup.sh dulu (bikin .venv)")

    if (ROOT / ".env").exists():
        line(OK, ".env", "ada")
    else:
        line(WARN, ".env", "belum ada", "cp .env.example .env, lalu isi key-nya")

    for tool in ("ffmpeg", "pandoc"):
        if shutil.which(tool):
            line(OK, tool, "ada di PATH")
        else:
            line(FAIL, tool, "nggak ketemu", install_hint(tool))

    if MACOS:
        import platform
        prefix = brew_prefix()
        if platform.machine() == "x86_64":
            line(WARN, "arsitektur", "Mac Intel (x86_64)",
                 "WhisperX butuh PyTorch 2.8 dan ctranslate2 4.5+, yang nggak punya paket Intel Mac")
        if prefix is None:
            line(WARN, "Homebrew", "nggak ketemu",
                 "https://brew.sh, lalu ./setup.sh lagi")
        else:
            line(OK, "Homebrew", prefix)
            if pango_installed(prefix):
                line(OK, "pango", "ada (buat PDF)")
            else:
                line(WARN, "pango", "belum ke-install, PDF bakal gagal", "brew install pango")
            icon, detail, hint = dyld_status(prefix, os.environ.get("DYLD_FALLBACK_LIBRARY_PATH"))
            line(icon, "DYLD_FALLBACK_LIB…", detail, hint)

    for mod, pip_name, who in PACKAGES:
        if has_module(mod):
            line(OK, pip_name, "ke-install")
        else:
            line(FAIL, pip_name, f"belum ke-install ({who})",
                 f"pip install {pip_name}   atau ulang ./setup.sh")

    cuda = cuda_status()
    if cuda is None:
        line(WARN, "device", "dilewati, torch belum ada")
    elif not cuda[0] and cuda[1] != NO_CUDA:
        line(WARN, "device", cuda[1], "torch-nya bermasalah, jadi GPU nggak bisa dicek")
    else:
        try:
            choice = device_choice(cuda[0])
        except SystemExit as e:     # a bad SUPERMEET_DEVICE
            line(FAIL, "device", str(e))
        else:
            if choice.is_cuda and cuda[0]:
                line(OK, "device", f"GPU CUDA: {cuda[1]} ({choice.model})")
            elif choice.is_cuda:
                line(FAIL, "device", "SUPERMEET_DEVICE=cuda tapi torch nggak lihat GPU CUDA",
                     "pasang driver NVIDIA, atau hapus SUPERMEET_DEVICE biar jalan di CPU")
            else:
                forced = os.environ.get("SUPERMEET_DEVICE", "").strip().lower() == "cpu"
                line(INFO, "device",
                     f"CPU ({choice.model})" + (", dipaksa SUPERMEET_DEVICE=cpu" if forced else ""),
                     "lebih lambat dari GPU NVIDIA; pertama kali jalan, model di-download dulu")

    key = gemini_key()
    if key:
        line(OK, key[0], f"ke-set ({mask(key[1])})")
    else:
        line(FAIL, "GEMINI_API_KEY", "belum di-set",
             "tanpa ini ringkasan nggak jalan; transcribe-only tetap bisa")
    if os.environ.get("HF_TOKEN"):
        line(OK, "HF_TOKEN", f"ke-set ({mask(os.environ['HF_TOKEN'])})")
    else:
        line(WARN, "HF_TOKEN", "belum di-set", "diarization (label pembicara) bakal di-skip")
    if os.environ.get("ANTHROPIC_API_KEY"):
        line(OK, "ANTHROPIC_API_KEY", f"ke-set ({mask(os.environ['ANTHROPIC_API_KEY'])})")
    else:
        line(WARN, "ANTHROPIC_API_KEY", "belum di-set", "cuma dibutuhin mode realtime (serve)")

    try:
        from core import config
    except Exception as e:  # a bad path env var should be reported, not crash the doctor
        line(FAIL, "folder", f"core.config gagal di-load: {e}")
    else:
        cloud = getattr(config, "CLOUD_ROOT", None)
        local = cloud is not None and Path(cloud) == ROOT / "whisper"
        if cloud is not None:
            line(OK, "root folder",
                 f"{cloud}  ({'lokal, nggak ada cloud folder' if local else 'cloud'})")
        for label, name in (("inbox", "INBOX"), ("output ringkasan", "OUTPUT_DIR"),
                            ("output transkrip", "TRANSCRIPT_OUTPUT_DIR")):
            path = getattr(config, name, None)
            if path is None:
                continue
            if Path(path).is_dir():
                line(OK, label, str(path))
            else:
                line(WARN, label, f"belum ada: {path}", "bakal dibikin pas dipakai, atau mkdir sendiri")

    if (ROOT / "certs" / "cert.pem").exists():
        line(OK, "certs/", "ada")
    else:
        line(WARN, "certs/", "belum ada", "cuma buat HTTPS realtime di HP: ./make-cert.sh")

    if online:
        if key:
            icon, msg = check_gemini_online(key[1], default_model())
            line(icon, "Gemini online", msg)
        else:
            line(WARN, "Gemini online", "dilewati, key belum di-set")

    bad, warn = results.count(FAIL), results.count(WARN)
    print(f"\n{bad} masalah, {warn} peringatan." if bad or warn else "\nSemua beres.")
    if not online:
        print("Tambah --online buat tes panggilan Gemini beneran.")
    return 1 if bad else 0
