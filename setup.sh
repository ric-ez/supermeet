#!/usr/bin/env bash
# One-shot setup: check system tools, make .venv, install packages, create .env,
# then run the doctor. Safe to re-run. Never uses sudo and never installs system
# packages -- missing ones are only reported, with the command to install them.
#
# Works on Linux/WSL2 (apt) and macOS (Homebrew, Apple Silicon or Intel).
# Keep it bash 3.2 clean (macOS ships 3.2): no arrays, no declare -A, no
# ${var,,}, no mapfile, no [[ -v ]]; lists are plain strings.
#
# SETUP_SKIP_PIP=1 skips the pip steps (for testing the rest of the script).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

missing=""   # hard requirements that are absent (space separated)
hints=""     # install commands to print at the end (newline separated)

ok()   { printf '  ✅ %s\n' "$*"; }
info() { printf '  ℹ️  %s\n' "$*"; }
warn() { printf '  ⚠️  %s\n' "$*"; }
bad()  { printf '  ❌ %s\n' "$*"; }
add_missing() { missing="$missing $1"; }
add_hint()    { hints="$hints$1
"; }

macos=0; if [[ "$(uname -s)" == "Darwin" ]]; then macos=1; fi
arch="$(uname -m)"

echo "== Cek sistem =="

PYTHON="python3"
BREW_PREFIX=""
pkgs=""      # system packages to put in the install hint (space separated)

if (( macos )); then
  if [[ "$arch" == "arm64" ]]; then
    ok "macOS, Apple Silicon (arm64)"
  else
    ok "macOS, $arch"
  fi

  # Homebrew provides everything below. Right after installing it, brew is often
  # not on PATH yet (Apple Silicon: /opt/homebrew/bin), so look in the usual places.
  if ! command -v brew >/dev/null 2>&1; then
    for b in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [[ -x "$b" ]]; then
        eval "$("$b" shellenv)"
        warn "brew ada di $b tapi belum masuk PATH terminal ini."
        warn "Biar permanen: echo 'eval \"\$($b shellenv)\"' >> ~/.zprofile"
        break
      fi
    done
  fi
  if ! command -v brew >/dev/null 2>&1; then
    bad "Homebrew belum ke-install"
    echo
    echo "Install Homebrew dulu (ini perintah resmi dari https://brew.sh, nggak dijalanin otomatis):"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo "Habis itu ikuti petunjuk 'Next steps' di akhir installernya, buka terminal baru,"
    echo "lalu jalankan ./setup.sh lagi."
    exit 1
  fi
  BREW_PREFIX="$(brew --prefix)"
  ok "Homebrew ($BREW_PREFIX)"

  if [[ "$arch" == "x86_64" ]]; then
    warn "Mac Intel (x86_64): WhisperX butuh PyTorch 2.8 dan ctranslate2 4.5+, dan keduanya"
    warn "nggak punya paket buat Intel Mac. Langkah pip kemungkinan besar gagal."
    warn "(Kalau ini sebenarnya Mac Apple Silicon, terminalmu jalan lewat Rosetta; pakai terminal biasa.)"
  fi

  # System python is often 3.9 (Xcode tools), too old. WhisperX supports 3.10-3.13,
  # so look for a Homebrew python in that range, newest-safest first.
  PYTHON=""
  for c in "$BREW_PREFIX/bin/python3.12" "$BREW_PREFIX/bin/python3.13" \
           "$BREW_PREFIX/bin/python3.11" "$BREW_PREFIX/bin/python3.10" \
           python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null && \
       "$c" -c 'import venv, ensurepip' 2>/dev/null; then
      PYTHON="$(command -v "$c")"
      break
    fi
  done
  if [[ -n "$PYTHON" ]]; then
    ok "python $("$PYTHON" -c 'import platform; print(platform.python_version())') ($PYTHON)"
  else
    bad "python 3.10 sampai 3.13 nggak ketemu (python bawaan macOS biasanya 3.9, terlalu lama)"
    add_missing python
    pkgs="$pkgs python@3.12"
    PYTHON="python3"
  fi
else
  # python >= 3.10
  if command -v python3 >/dev/null && \
     python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
  else
    bad "python3 3.10 ke atas nggak ketemu"
    add_missing python3
    add_hint "sudo apt install python3 python3-venv"
  fi

  # python3-venv is a separate apt package on Debian/Ubuntu.
  if command -v python3 >/dev/null && ! python3 -c 'import venv, ensurepip' 2>/dev/null; then
    bad "modul venv Python nggak ada"
    add_missing python3-venv
    add_hint "sudo apt install python3-venv"
  fi
fi

for tool in ffmpeg pandoc; do
  if command -v "$tool" >/dev/null; then
    ok "$tool"
  else
    bad "$tool belum ke-install"
    add_missing "$tool"
    pkgs="$pkgs $tool"
  fi
done

# weasyprint (PDF) needs pango at runtime; the pip wheel does not bundle it.
if (( macos )); then
  if brew list --versions pango >/dev/null 2>&1; then
    ok "pango (buat PDF)"
  else
    warn "pango belum ke-install, PDF bakal gagal. Install: brew install pango"
    # Joins the hint below only if something else is missing; pango alone is a warning.
    pkgs="$pkgs pango"
  fi
else
  # Captured first: grep -q exiting early would SIGPIPE ldconfig and trip pipefail.
  libs="$(ldconfig -p 2>/dev/null || true)"
  if grep -qi pango <<<"$libs"; then
    ok "pango (buat PDF)"
  else
    warn "pango nggak ketemu, PDF bisa gagal. Install: sudo apt install libpango-1.0-0 libpangoft2-1.0-0"
  fi
fi

if [[ -n "$pkgs" ]]; then
  if (( macos )); then add_hint "brew install$pkgs"
  else add_hint "sudo apt install$pkgs"; fi
fi

if (( macos )) && [[ "$BREW_PREFIX" != "/usr/local" ]]; then
  # Apple Silicon Homebrew keeps libpango (and ffmpeg's libs, for torchcodec) in
  # /opt/homebrew/lib, which macOS does not search by default. supermeet.py sets
  # DYLD_FALLBACK_LIBRARY_PATH for itself on start; nothing to do here.
  info "supermeet.py otomatis set DYLD_FALLBACK_LIBRARY_PATH=$BREW_PREFIX/lib supaya pango ketemu."
  info "Kalau kamu jalanin python langsung (bukan lewat supermeet.py) dan PDF gagal:"
  info "  export DYLD_FALLBACK_LIBRARY_PATH=\"$BREW_PREFIX/lib\""
fi

if command -v openssl >/dev/null; then
  ok "openssl"
else
  warn "openssl nggak ada, cuma dibutuhin ./make-cert.sh (HTTPS realtime di HP)"
fi

if (( ! macos )) && command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  ok "GPU NVIDIA: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
elif (( macos )); then
  info "Mac: transkripsi jalan di CPU (model large-v3-turbo), lebih lambat dari GPU NVIDIA"
  info "Pertama kali jalan, model di-download dulu (beberapa GB), jadi run pertama lebih lama."
else
  info "GPU NVIDIA nggak terdeteksi: transkripsi jalan di CPU (model large-v3-turbo), lebih lambat dari GPU NVIDIA"
  info "Pertama kali jalan, model di-download dulu (beberapa GB), jadi run pertama lebih lama."
fi

if [[ -n "$missing" ]]; then
  echo
  echo "Yang wajib dan belum ada:$missing"
  echo "Install dulu, lalu jalankan ./setup.sh lagi:"
  printf '%s' "$hints" | sed 's/^/  /'
  if (( macos )) && [[ "$PYTHON" == "python3" ]]; then
    echo "  (setelah install python, ./setup.sh otomatis pakai $BREW_PREFIX/bin/python3.12)"
  fi
  exit 1
fi

echo
echo "== Python environment =="
if [[ -x .venv/bin/python ]]; then
  if (( macos )) && ! .venv/bin/python -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 14) else 1)' 2>/dev/null; then
    bad ".venv dibuat dengan Python yang nggak cocok ($(.venv/bin/python -c 'import platform; print(platform.python_version())'))"
    echo "  Hapus dan bikin ulang: rm -rf .venv && ./setup.sh"
    exit 1
  fi
  ok ".venv sudah ada"
else
  "$PYTHON" -m venv .venv
  ok ".venv dibuat"
fi

if [[ "${SETUP_SKIP_PIP:-0}" == "1" ]]; then
  warn "SETUP_SKIP_PIP=1, install package dilewati"
else
  if (( macos )); then
    echo "  Install pertama kali beberapa GB dan butuh waktu, sabar ya. (torch di Mac itu versi biasa, tanpa CUDA.)"
  else
    echo "  Install pertama kali beberapa GB (torch + model GPU) dan butuh waktu, sabar ya."
  fi
  .venv/bin/python -m pip install -q -U pip
  if ! .venv/bin/python -m pip install -r requirements.txt; then
    bad "pip install gagal, lihat pesan error di atas"
    if (( macos )); then
      echo "  Kalau ada error compile: xcode-select --install, lalu ulangi ./setup.sh"
    fi
    exit 1
  fi
  ok "package ke-install"
fi

echo
echo "== Konfigurasi =="
if [[ -f .env ]]; then
  ok ".env sudah ada, nggak diubah"
elif [[ -f .env.example ]]; then
  cp .env.example .env
  chmod 600 .env
  ok ".env dibuat dari .env.example"
else
  warn ".env.example nggak ketemu, bikin .env sendiri"
fi

# Istilah per matkul untuk mode kuliah; pribadi, jadi ikut .gitignore.
if [[ ! -f kuliah/matkul.toml && -f kuliah/matkul.example.toml ]]; then
  cp kuliah/matkul.example.toml kuliah/matkul.toml
  ok "kuliah/matkul.toml dibuat dari contoh"
fi

echo
echo "== Doctor =="
# A red line here is information, not a reason to abort the setup summary.
.venv/bin/python supermeet.py doctor || true

cat <<'EOF'

== Langkah berikutnya ==
  1. Isi GEMINI_API_KEY di .env (gratis di https://aistudio.google.com/apikey).
  2. Buat label pembicara: setujui syarat model pyannote di Hugging Face
     (pyannote/speaker-diarization-3.1 dan pyannote/segmentation-3.0), lalu
     isi HF_TOKEN di .env.
  3. Cek lagi:    ./supermeet.py doctor --online
  4. Jalankan:    ./supermeet.py watch
EOF
