# supermeet

Transkrip dan ringkasan rekaman audio (meeting, kuliah, seminar, podcast) yang jalan di mesin sendiri. Audio diproses lokal dengan Whisper; teks transkripnya diringkas Gemini jadi PDF yang rapi. Ada juga panel live di browser untuk dipakai selama meeting, dan mode khusus untuk rekaman kuliah. Prompt dan bahasa default-nya Indonesia.

| Alur | Perintah | Untuk apa | Hasil |
|---|---|---|---|
| **Batch** (sesudah meeting) | `watch` / `transcribe` | Arsip akurat: WhisperX large-v3 + diarization (label speaker) + ringkasan Gemini | PDF ringkasan dan PDF transkrip |
| **Realtime copilot** (selama meeting) | `serve` | Transkrip live, ringkasan bergulir, dan pertanyaan untuk host, ditenagai Claude | Panel di browser; audionya diserahkan ke alur batch saat meeting selesai |
| **Kuliah** | `kuliah` atau halaman `/kuliah` | Transkrip mentah per pertemuan, tanpa ringkasan, biaya $0 | `.txt` + `.srt` per pertemuan |

---

## Mulai cepat

**Syarat**

- **Linux, WSL2 (Windows), atau macOS dengan chip Apple Silicon (M1 dan seterusnya).** Mac Intel belum bisa: WhisperX membutuhkan torch 2.8, sedangkan wheel torch untuk Mac Intel berhenti di 2.2
- **GPU NVIDIA sangat disarankan.** Dengan GPU, transkripsi memakai `large-v3` (sekitar 8x realtime). Tanpa GPU, termasuk di Mac, transkripsi jalan di **CPU** dengan model `large-v3-turbo`: hasilnya baik tapi jauh lebih lambat, dan model di-download saat pertama jalan
- Python 3.10 sampai 3.13 (batas atas dari WhisperX), `ffmpeg`, `pandoc` (`setup.sh` mengecek semuanya dan mencetak perintah `apt` atau `brew` kalau ada yang kurang; script ini tidak memasang paket sistem sendiri). Di Mac, `setup.sh` juga membutuhkan [Homebrew](https://brew.sh)
- API key Gemini gratis dari https://aistudio.google.com/apikey

**Langkah**

```bash
git clone <url-repo-ini> supermeet && cd supermeet
./setup.sh                      # bikin .venv, pasang dependensi, salin .env.example -> .env
```

Buka `.env`, isi minimal `GEMINI_API_KEY`. Lalu cek semuanya:

```bash
./supermeet.py doctor          # checklist: dependensi, device, key, folder
./supermeet.py doctor --online # plus satu panggilan Gemini kecil untuk memvalidasi key
```

Jalankan watcher dan biarkan hidup:

```bash
./supermeet.py watch
```

Jawab dua pertanyaan di layar (model ringkasan dan jenis rekaman; Enter dua kali sudah cukup), lalu **taruh file audio** (`.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`) ke `whisper/transcribe-inbox/`. Beberapa menit kemudian PDF-nya muncul di `whisper/transcribe-output/`:

```
whisper/transcribe-output/summaries/<nama>_summary.pdf       ringkasan
whisper/transcribe-output/transcripts/<nama>_transcript.pdf  transkrip lengkap (+ .md)
```

Tidak perlu setup cloud apa pun. Kalau mau menjatuhkan rekaman dari HP/tablet, lihat [Folder cloud](#folder-cloud-opsional).

Mau coba satu file tanpa watcher:

```bash
./supermeet.py transcribe rekaman.mp3
```

> `./supermeet.py ...` dan `./supermeet.py ...` sama saja; script ini otomatis pindah ke `.venv` kalau ada.

### Khusus MacBook

```bash
brew install ffmpeg pandoc python@3.12 pango   # sekali saja, kalau belum ada
git clone <url-repo-ini> supermeet && cd supermeet
./setup.sh
```

- `setup.sh` memilih Python dari Homebrew (3.10 sampai 3.13) untuk `.venv`. Python bawaan macOS biasanya terlalu tua.
- Mac tidak punya perintah `python`; pakai `./supermeet.py ...` (atau aktifkan `.venv`).
- Tanpa GPU NVIDIA, `doctor` menampilkan `CPU (large-v3-turbo)`. Itu normal. Rekaman panjang butuh waktu; untuk kecepatan, pakai model lebih kecil: `SUPERMEET_WHISPER_MODEL=small` di `.env`.
- Google Drive (mode **Mirror files**) dan iCloud Drive dideteksi otomatis, lihat [Folder cloud](#folder-cloud-opsional). Beri Terminal izin ke folder itu di System Settings > Privacy & Security.
- Realtime copilot di Mac memakai tombol mic di browser. Klien capture audio sistem hanya untuk Windows.
- Status: dukungan Mac **belum diuji di perangkat Mac asli** (dikembangkan dan diuji di Linux). Kalau ada yang gagal, jalankan `./supermeet.py doctor` dan buka issue.

---

## Perintah

```
supermeet.py doctor [--online]             cek setup: tool, paket, key, folder
supermeet.py watch                         pantau inbox terus-menerus (alur batch)
supermeet.py transcribe <audio> [opsi]     satu file, sekali jalan
supermeet.py serve [--https] [--port N]    server realtime + halaman kuliah
supermeet.py kuliah <folder> [opsi]        rekaman kuliah, digabung per pertemuan
supermeet.py pdf <file.md> [out_dir]       markdown -> PDF bergaya
supermeet.py gpu                           siapa yang memegang GPU sekarang
```

Opsi `transcribe`:

| Opsi | Efek |
|---|---|
| `--force` | Abaikan transkrip cache, transkripsi ulang dari audio |
| `--speakers N` | Beri tahu pyannote jumlah speaker yang pasti |
| `--min-speakers N` / `--max-speakers N` | Batas bawah / atas jumlah speaker (tidak bisa digabung dengan `--speakers`) |
| `--transcribe-only` | Tanpa ringkasan Gemini, hanya PDF transkrip |

---

## Alur batch (`watch` / `transcribe`)

```
rekaman (file audio apa pun)
   └─► taruh di  <folder whisper>/transcribe-inbox/
          └─► watcher.sh polling tiap 15 detik, menunggu ukuran file stabil
                 └─► pipeline.py: WhisperX -> pyannote -> filter halusinasi -> Gemini
                        ├─► transcribe-output/summaries/*_summary.pdf (+ .md)
                        └─► transcribe-output/transcripts/*_transcript.pdf (+ .md)
```

Ekstensi yang dipantau watcher: `.mp3 .m4a .wav .flac .ogg`. Setelah sukses, file asli dipindah ke `audio/processed/`; yang gagal permanen ke `audio/failed/`. Semua output watcher juga masuk `logs/watcher.log`.

### Menu `watch`

Tiap kali `watch` dijalankan, ia bertanya dua hal untuk run itu (berlaku ke semua file yang masuk sampai watcher dimatikan):

```
Model buat ringkasan run ini:
  1. Gemini 3.8 Flash        $0  free tier, ada limit harian
  2. Gemini 3.5 Flash-Lite   $0  free tier, lebih ringan dan cepat
  3. Transcribe only         $0  tanpa ringkasan, cuma PDF transkrip
  > 1

Jenis rekaman (nentuin penekanan ringkasan):
  1. Tanpa penekanan khusus
  2. Seminar / webinar (online atau offline)
  3. Kuliah
  4. Diskusi saham / market
  5. Rapat kerja / meeting umum
  6. Tulis sendiri...
  > 3
```

Preset tidak mengganti struktur output di `SYSTEM_PROMPT`; ia menambahkan apa yang perlu digali lebih dalam di dalam struktur itu. Kuliah mencari kisi-kisi, istilah, dan hal yang diulang dosen. Seminar memisahkan materi dari tanya-jawab dan mengumpulkan referensi. Saham memaksa angka dan asumsi ditulis persis. Rapat memaksa tiap action item punya PIC. Mengetik kalimat langsung (bukan angka) juga boleh dan dianggap konteks sekali pakai. Preset baru = tambah satu baris di `CONTEXTS` (`supermeet.py`).

Enter di kedua pertanyaan = Gemini 3.8 Flash tanpa penekanan khusus.

Kalau tidak ada TTY (nohup, systemd, cron), kedua pertanyaan dilewati dan nilai default dipakai. Override manual lewat `.env` atau env var, dan ini juga cara `transcribe` mendapat konteks:

```bash
export SUPERMEET_GEMINI_MODEL=gemini-3.5-flash-lite
export SUPERMEET_EXTRA_PROMPT="ini rapat vendor, catat komitmen harga"
./supermeet.py transcribe rekaman.mp3
```

### Transcribe only

Pilihan 3 di menu (atau `--transcribe-only`, atau `SUPERMEET_TRANSCRIBE_ONLY=1`) tidak memanggil Gemini sama sekali: hanya transkrip + diarization, keluarnya PDF transkrip dan `.md`. Tidak butuh `GEMINI_API_KEY`, dan pertanyaan jenis rekaman dilewati. File `.md` bisa dilampirkan ke asisten AI lain kalau mau diringkas manual; teks polos lebih hemat token daripada PDF.

Kalau belakangan butuh ringkasannya, ambil file-nya dari `audio/processed/` dan taruh lagi di inbox saat `watch` jalan dengan model. Transkripnya sudah di-cache (`transcripts/` di root repo), jadi langsung ke Gemini tanpa GPU.

### Ringkasan pakai Gemini

Ringkasan batch memakai Gemini, bukan Claude. Ambil key gratis di https://aistudio.google.com/apikey, isi `GEMINI_API_KEY` di `.env` (`GOOGLE_API_KEY` juga terbaca kalau yang pertama tidak di-set).

- Free tier punya limit per menit dan per hari. Limit per menit biasanya lewat sendiri: `pipeline.py` menunggu lalu mencoba lagi sampai 3 kali, dan watcher menambah putaran lagi di poll berikutnya.
- Limit harian tidak pulih dalam hitungan menit, jadi file-nya berakhir di `audio/failed/`. Taruh balik ke inbox besok; transkripnya sudah di-cache.
- **Privasi di free tier:** Google boleh memakai isi yang dikirim untuk mengembangkan produknya, jadi teks transkrip rekamanmu terkirim ke Google. Kalau isinya sensitif, pakai **Transcribe only**.

### Kalau gagal

Exit code `pipeline.py` menentukan nasib file:

| Exit | Artinya | Watcher |
|---|---|---|
| `0` | Beres | Pindah ke `audio/processed/` |
| `2` | Tidak akan sembuh kalau diulang: ringkasan terpotong `max_tokens`, model menolak, API key / model / parameter salah, audio hening semua | Langsung pindah ke `audio/failed/` |
| `75` | GPU sedang dipegang proses lain | Coba lagi di poll berikutnya, tidak dihitung sebagai gagal |
| lainnya | Jaringan putus, API overload, kuota Gemini habis, crash | Coba lagi, maksimal 3 kali, setelah itu ke `audio/failed/` |

Beresin penyebabnya, lalu taruh balik file-nya ke inbox. Transkrip sudah di-cache, jadi yang diulang hanya langkah Gemini.

---

## Folder cloud (opsional)

Tanpa konfigurasi, semuanya ada di `<repo>/whisper/`:

```
whisper/
  transcribe-inbox/                 taruh audio di sini
  transcribe-output/summaries/      *_summary.pdf
  transcribe-output/transcripts/    *_transcript.pdf + .md
```

Kalau mau menjatuhkan rekaman dari HP atau tablet, arahkan `whisper` ke folder yang disinkronkan layanan cloud. Di WSL, folder berikut dideteksi otomatis dan dipakai menggantikan `<repo>/whisper/`:

| Layanan | Folder Windows | Di WSL |
|---|---|---|
| Google Drive | `C:\Users\<user>\My Drive\whisper` | `/mnt/c/Users/<user>/My Drive/whisper` |
| iCloud Drive | `C:\Users\<user>\iCloudDrive\whisper` | `/mnt/c/Users/<user>/iCloudDrive/whisper` |

Di macOS, urutan deteksinya setelah folder WSL di atas:

| Layanan | Folder |
|---|---|
| Google Drive (mode Mirror files) | `~/Library/CloudStorage/GoogleDrive-<akun>/My Drive/whisper` (klien lama: `~/Google Drive/My Drive/whisper`) |
| iCloud Drive | `~/Library/Mobile Documents/com~apple~CloudDocs/whisper` |

Syarat Google Drive: aplikasinya harus dalam mode **Mirror files** (Preferences > Google Drive > My Drive syncing options). Mode default "Stream files" adalah drive virtual yang tidak bisa dibaca dari WSL.

Alur dari HP/tablet: upload rekaman lewat aplikasi Drive/iCloud ke `whisper/transcribe-inbox/`, PDF muncul di `whisper/transcribe-output/`. Syaratnya `watch` sedang jalan di komputer.

Lokasi lain atau override manual lewat `.env`:

| Var | Efek |
|---|---|
| `TRANSCRIBE_CLOUD_ROOT` | Folder akar (berisi `transcribe-inbox/` dan `transcribe-output/`) |
| `TRANSCRIBE_INBOX` | Folder inbox saja |
| `TRANSCRIBE_OUTPUT` | Folder ringkasan |
| `TRANSCRIBE_TRANSCRIPT_OUTPUT` | Folder transkrip |

Cek hasil resolusi folder: `python core/config.py`.

---

## Realtime copilot (`serve`)

Panel live selama meeting: transkrip, pertanyaan untuk host, dan ringkasan + action items yang diperbarui berkala. Bagian ini memakai **Claude** (bukan Gemini) dan butuh `ANTHROPIC_API_KEY`, atau `OPENROUTER_API_KEY` kalau mau lewat OpenRouter. Model STT-nya faster-whisper `large-v3-turbo`, tanpa diarization.

```
Zoom / Meet / Teams / ruangan
   └─► klien capture  ──►  ws://host:8848/ws/ingest   (PCM 16 kHz mono)
          └─► VAD -> whisper turbo -> filter halusinasi
                 └─► 2 worker Claude: ringkasan+action, pertanyaan
                        └─► browser: transkrip live | pertanyaan host | ringkasan
                               └─► tekan Selesai -> handoff ke alur batch -> PDF arsip
```

```bash
./supermeet.py serve              # http://localhost:8848
./supermeet.py serve --port 9000
```

Saat **Selesai** ditekan, audio sesi (`sessions/<waktu>/meeting_*.wav`) diserahkan ke `transcribe`, jadi tiap meeting tetap menghasilkan PDF arsip tanpa drop file manual.

### Sumber audio

Server tidak mendengar apa pun sampai ada klien capture yang tersambung. Indikator koneksinya ada di header UI.

**Meeting online di laptop Windows** (bisa mendengar suara lawan bicara). Jalankan klien capture memakai Python **Windows**, bukan Python WSL, karena WASAPI loopback hanya ada di sisi Windows:

```powershell
py -m pip install soundcard numpy websockets
py "\\wsl$\<Distro>\home\<user>\supermeet\realtime\ingest\win_capture.py"
```

Ganti `<Distro>` dan `<user>` sesuai instalasi WSL-mu (atau salin `win_capture.py` ke folder Windows dan jalankan dari sana). Python Windows bisa dipasang dengan `winget install Python.Python.3.12`.

Opsi: `--source both` (default, speaker + mic), `loopback` (speaker saja), `mic` (mic saja, untuk meeting tatap muka), `--list` (daftar device), `--server ws://host:8848/ws/ingest`, `--mic-gain`, `--speaker-gain`.

**Meeting tatap muka / HP di meja:** buka UI, tekan **Mic device ini**. Tombol **Audio sistem** memakai screen-share Chrome; di Windows hanya menawarkan audio kalau memilih **Entire Screen**.

Perekam yang menyimpan dulu baru sinkron (mis. alat perekam saku) tidak bisa realtime. Pakai alur batch.

### Akses dari HP / tablet

Di Windows + WSL2, aktifkan mirrored networking. Buat `C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
firewall=true
```

Lalu `wsl --shutdown`, buka WSL lagi, dan izinkan port-nya sekali (PowerShell sebagai admin):

```powershell
New-NetFirewallRule -DisplayName "supermeet" -Direction Inbound -LocalPort 8848 -Protocol TCP -Action Allow
```

Buka `http://<ip-komputer>:8848` di perangkat lain di jaringan yang sama (alamat LAN dicetak saat `serve` mulai).

**`--https` wajib untuk capture mic dari HP.** Browser hanya memberi akses mic di secure context; `localhost` dianggap aman, `http://192.168.x.x` tidak. `./supermeet.py serve --https` menjalankan `make-cert.sh`, yang membuat CA lokal dan sertifikat server di `certs/`. HP menampilkan peringatan sertifikat satu kali; terima, lalu jalan.

**iPad / iPhone (Safari)** butuh langkah tambahan. Safari mengizinkan peringatan sertifikat dilewati untuk halamannya tapi tidak untuk websocket-nya ([WebKit bug 41419](https://lists.webkit.org/pipermail/webkit-dev/2011-June/017335.html)); halaman terbuka tapi panel tidak pernah tersambung. Jadi perangkat harus benar-benar mempercayai sertifikatnya. Cukup sekali:

1. `./supermeet.py serve --https` (pertama kali, ini membuat CA di `certs/ca.pem`)
2. Di perangkat iOS, buka `https://<ip-komputer>:8848/ca.crt` dengan **Safari** (Chrome iOS tidak bisa memasang profil), lewati peringatan, lalu **Allow**
3. Settings > **Profile Downloaded** > Install
4. Settings > General > About > **Certificate Trust Settings** > aktifkan *supermeet local CA*
5. Buka `https://<ip-komputer>:8848`; gembok normal dan tombol mic jalan

Kalau IP komputer berubah (Wi-Fi lain), `serve --https` menerbitkan ulang sertifikat server otomatis; CA tetap sama, jadi tidak perlu pasang ulang di perangkat iOS. iOS menolak sertifikat server yang berumur lebih dari 825 hari atau tanpa EKU serverAuth, walaupun CA-nya sudah dipercaya ([aturan Apple](https://support.apple.com/103769)); `make-cert.sh` sudah mengikutinya.

Catatan khusus iPad/iPhone:

- **Audio sistem tidak tersedia.** Halaman web di iOS tidak bisa mendengar aplikasi lain (Zoom/Meet/Teams), jadi tombolnya disembunyikan. Kalau meeting jalan di laptop, pakai `win_capture.py` di laptop dan biarkan iPad hanya untuk melihat panel.
- **Satu sumber audio saja.** Mic perangkat dan `win_capture.py` jangan menyala bersamaan; server tidak men-mix, frame dua stream tercampur dan transkrip rusak. `--source both` sudah menangkap lawan bicara + mic laptop sekaligus.
- **Safari harus tetap di depan.** iOS memutus mic begitu Safari ke background.
- **`certs/ca-key.pem` jangan disebar.** Siapa pun yang memegang file itu bisa membuat sertifikat yang dipercaya perangkatmu. Kalau tidak dipakai lagi, hapus profilnya di Settings > General > VPN & Device Management.

### Model dan biaya

Default-nya Claude Haiku 5.5 ($0.10 / $0.50 per 1 juta token): meeting 2 jam diperkirakan sekitar **$0.14** (perkiraan, belum diukur). Transkrip di-cache sebagai prefix, jadi biaya naik jauh lebih landai daripada linear terhadap durasi. Angka berjalannya tampil di header UI. Haiku 5.5 mengenakan tarif 5x begitu prompt melewati 100 ribu token (kira-kira transkrip 6 jam); header UI ikut menghitungnya kalau terjadi.

Kalau pertanyaan untuk host mulai generik, naikkan `RTSM_CLAUDE_EFFORT=medium` sebelum pindah ke model yang lebih besar. Ganti model lewat `RTSM_CLAUDE_MODEL` (format ID beda: OpenRouter `anthropic/claude-sonnet-5`, Anthropic langsung `claude-sonnet-5`).

---

## Mode kuliah (`kuliah`)

Transkrip mentah per pertemuan untuk bahan belajar. Bukan ringkasan, tidak memakai model bahasa, biaya $0. Tidak butuh `GEMINI_API_KEY` maupun `HF_TOKEN`.

```
upload (satu file / .zip)  ──►  kuliah-uploads/<KODE>/
   └─► tombol Proses ─► WhisperX large-v3 + VAD + filter halusinasi
          └─► kuliah-output/<KODE>/[P04] Transkrip.txt  (+ .srt)
                 └─► download dari halaman (per file, per matkul, atau semua sekaligus .zip)
```

**Dari halaman:** `./supermeet.py serve`, buka `http://localhost:8848/kuliah` (atau dari HP/tablet, lihat bagian realtime). Tarik file atau `.zip`, tekan **Proses N pertemuan**, lalu ambil hasilnya di kartu **Hasil**. Hasil tidak disinkron ke mana-mana; halaman ini satu-satunya pintu keluarnya. Kartu itu membaca langsung dari `kuliah-output/`, jadi tetap ada walau audionya sudah dibersihkan.

**Dari terminal:** `./supermeet.py kuliah "<folder>"`; path Windows (`C:\...`) juga diterima. Opsi: `--dry-run` (hanya tampilkan rencana), `--force` (ulang yang sudah ada), `--out DIR`.

Nama file yang terbaca: `MK1 - P04 - 2026-09-22.m4a` atau `MK2 - P04 - 2026-09-22 (bagian 1 dari 3).opus` (kode matkul - nomor pertemuan - tanggal; spasi dan huruf besar-kecil bebas). Yang tidak terbaca masuk ke **Perlu nama** di halaman. Aturannya:

- Beberapa bagian digabung jadi satu transkrip dengan timestamp yang nyambung. Pertemuan **menunggu** sampai semua bagiannya ada.
- Rekaman di bawah 60 detik dilewati (biasanya kepencet). Yang transkripnya sudah ada dilewati, jadi aman dijalankan ulang tiap minggu.
- Folder yang namanya diawali `_` (mis. `_Bukan Kuliah`) diabaikan, baik di zip maupun di folder.
- Audio asli tidak pernah disentuh. Upload itu salinan, dan **Bersihkan audio yang udah diproses** hanya menghapus salinan itu.
- Format audio: `.opus .m4a .mp3 .wav .flac .ogg .aac .webm`.
- Timestamp `[1:35:12]` untuk kuliah 1 jam ke atas, `[05:12]` di bawahnya, kira-kira tiap menit. Kuliah multi-file mendapat satu `.srt` per bagian, supaya pas dengan file audionya.

**Istilah per matkul** ada di `kuliah/matkul.toml` (pribadi, tidak ikut git; `setup.sh` menyalinnya dari `kuliah/matkul.example.toml`): satu tabel `[KODE]` per matkul dengan `nama` (nama lengkap untuk header) dan `istilah` (daftar istilah, dipisah koma, dikirim ke Whisper supaya ejaannya benar). Buat pendek saja. Mengganti istilah = matkul itu ditranskripsi ulang di run berikutnya (pakai `--force` kalau transkripnya sudah ada). Bagian yang selesai disimpan di `transcripts/kuliah/`, jadi run yang terpotong lanjut dari situ. Kode yang tidak ada di file tetap diproses, hanya tanpa nama lengkap dan istilah.

**GPU:** server `serve` memegang GPU terus supaya perekaman live langsung siap. Menekan Proses di halaman membuatnya melepas model dulu, lalu memuat lagi setelah proses kuliah selesai. Kalau rapat live dimulai di tengah proses, proses kuliah **dihentikan** (rapat didahulukan, yang sudah selesai aman, tekan Proses lagi nanti). Dari terminal, `kuliah` menunggu GPU selama `serve` menyala, jadi pakai halamannya atau matikan `serve` dulu.

**Halaman kuliah tanpa login.** Siapa pun di jaringan yang sama bisa upload, hapus, dan mengunduh transkrip. Di Wi-Fi publik atau kampus, buat rule firewall dengan `-Profile Private` supaya port-nya tertutup di jaringan publik, atau jalankan `serve` hanya saat dibutuhkan.

---

## GPU dan koeksistensi

Dua pipeline masing-masing meminta sekitar 2 GB model. Pada kartu kecil (misalnya 6 GB VRAM), memuat yang kedua saat yang pertama memegang kartu tidak gagal dengan bersih; ia **nge-hang**. `core/gpu.py` mengatasinya dengan lease berbasis `flock` (file kunci default `/tmp/supermeet-gpu.lock`):

- Batch memegang lease hanya selama langkah yang memuat model. Ringkasan dan render PDF tidak butuh GPU, jadi kartu dikembalikan lebih awal.
- Server realtime memegang selama modelnya resident, dan melepas saat handoff supaya batch bisa masuk.
- Yang datang belakangan **mengantre**, bukan menabrak. Watcher yang kalah mencoba lagi di poll berikutnya, file-nya menunggu di inbox.

Artinya `watch` dan `serve` aman jalan bersamaan. Untuk melihat siapa yang memegang: `./supermeet.py gpu`.

---

## Konfigurasi (`.env`)

`setup.sh` menyalin `.env.example` ke `.env`. Format `KEY=value`, baris `#` adalah komentar. Env var yang sudah di-export di shell selalu menang atas isi `.env`. Nilai kosong dianggap tidak di-set.

**Key dan identitas**

| Var | Default | Buat apa |
|---|---|---|
| `GEMINI_API_KEY` | - | Ringkasan batch (key gratis dari aistudio.google.com/apikey). `GOOGLE_API_KEY` terbaca kalau ini kosong. Tidak perlu untuk Transcribe only dan mode kuliah |
| `HF_TOKEN` | - | Diarization pyannote. Terima syarat model di huggingface.co dulu; tanpa token, diarization dilewati (transkrip tanpa label speaker) |
| `ANTHROPIC_API_KEY` | - | Hanya untuk asisten live (`serve`) |
| `OPENROUTER_API_KEY` | - | Opsional: asisten live lewat OpenRouter. `ANTHROPIC_API_KEY` tidak pernah dikirim ke OpenRouter |
| `SUPERMEET_USER_NAME` | `user` | Nama yang dipakai di prompt ringkasan |

**Batch**

| Var | Default | Buat apa |
|---|---|---|
| `SUPERMEET_GEMINI_MODEL` | `gemini-3.8-flash` | `gemini-3.8-flash` atau `gemini-3.5-flash-lite`; `watch` menanyakan tiap run kalau tidak di-set |
| `SUPERMEET_EXTRA_PROMPT` | - | Konteks tambahan untuk ringkasan |
| `SUPERMEET_TRANSCRIBE_ONLY` | - | `1` = tanpa ringkasan, hanya PDF transkrip |
| `SUPERMEET_API_TIMEOUT` | `1800` | Timeout koneksi panggilan Gemini (detik) |
| `SUPERMEET_DEVICE` | `auto` | `auto` (CUDA kalau ada, kalau tidak CPU), `cuda`, atau `cpu` |
| `SUPERMEET_WHISPER_MODEL` | `large-v3` (CUDA) / `large-v3-turbo` (CPU) | Model Whisper untuk batch dan kuliah |
| `SUPERMEET_WATCHER_LOCK` | `/tmp/transcriber-watcher.lock.d` | Lokasi kunci satu-watcher-sekaligus |
| `SUPERMEET_GPU_WAIT` | `1800` | Berapa lama batch menunggu GPU sebelum menyerah (detik) |
| `SUPERMEET_GPU_LOCK` | `/tmp/supermeet-gpu.lock` | Lokasi file kunci GPU |
| `TRANSCRIBE_CLOUD_ROOT` | - | Folder akar inbox/output (lihat [Folder cloud](#folder-cloud-opsional)) |
| `TRANSCRIBE_INBOX` | `<akar>/transcribe-inbox` | Folder inbox |
| `TRANSCRIBE_OUTPUT` | `<akar>/transcribe-output/summaries` | Folder ringkasan |
| `TRANSCRIBE_TRANSCRIPT_OUTPUT` | `<akar>/transcribe-output/transcripts` | Folder transkrip |

**Realtime** (semua berprefix `RTSM_`, dibaca di `realtime/config.py`)

| Var | Default | Buat apa |
|---|---|---|
| `RTSM_HOST` | `0.0.0.0` | Alamat bind server |
| `RTSM_PORT` | `8848` | Port server |
| `RTSM_LANGUAGE` | `id` | Bahasa transkrip live |
| `RTSM_WHISPER_MODEL` | `large-v3-turbo` | `large-v3` lebih akurat untuk nama diri, lag 2.5 dtk vs 1.4 dtk |
| `RTSM_WHISPER_DEVICE` | `auto` | `auto` (ikut `SUPERMEET_DEVICE`), `cuda`, atau `cpu` |
| `RTSM_WHISPER_COMPUTE` | `int8` | Tipe komputasi |
| `RTSM_CLAUDE_MODEL` | `claude-haiku-5-5` (Anthropic) / `anthropic/claude-haiku-5.5` (OpenRouter) | Format ID berbeda per penyedia |
| `RTSM_CLAUDE_EFFORT` | `low` | `medium` kalau pertanyaan untuk host mulai generik |
| `RTSM_QUESTIONS_INTERVAL` | `45` | Detik antar batch pertanyaan |
| `RTSM_SUMMARY_INTERVAL` | `60` | Detik antar pembaruan ringkasan |
| `RTSM_MIN_NEW_CHARS` | `350` | Jangan panggil Claude kalau belum cukup omongan baru |
| `RTSM_GPU_WAIT_TIMEOUT` | `1800` | Berapa lama server menunggu batch melepas GPU (detik) |
| `RTSM_MODEL_LOAD_TIMEOUT` | `180` | Batas waktu memuat model STT (detik) |
| `RTSM_HANDOFF_WAIT_TIMEOUT` | `600` | Berapa lama sesi baru menunggu handoff arsip sebelum mulai tetap (detik) |

**Kuliah**

| Var | Default | Buat apa |
|---|---|---|
| `KULIAH_OUTPUT` | `kuliah-output/` | Folder hasil transkrip kuliah |
| `KULIAH_UPLOADS` | `kuliah-uploads/` | Tempat salinan upload menunggu diproses |
| `KULIAH_COMPUTE` | `int8` | `float16` bisa dicoba kalau VRAM cukup; bandingkan dulu di satu kuliah |
| `KULIAH_LOCK` | `/tmp/supermeet-kuliah.lock` | File kunci agar hanya satu run kuliah berjalan |

Ambang filter halusinasi ada di `core/filters.py`: satu tempat, dipakai ketiga alur.

---

## Privasi

- **Audio tidak pernah meninggalkan mesinmu.** Transkripsi dan diarization jalan lokal (Whisper, pyannote).
- **Alur batch:** teks transkrip (bukan audio) dikirim ke Gemini untuk diringkas. Di free tier, Google boleh memakainya untuk mengembangkan produknya. Pilih **Transcribe only** kalau tidak mau ada yang terkirim.
- **Realtime copilot:** potongan teks transkrip dikirim ke Anthropic (atau OpenRouter kalau `OPENROUTER_API_KEY` di-set) selama meeting berlangsung.
- **Mode kuliah:** tidak ada yang keluar dari mesin.
- Download model Whisper/pyannote dari Hugging Face terjadi sekali di awal.
- Halaman server (`serve`, termasuk `/kuliah`) tidak punya login; siapa pun di jaringan yang sama bisa mengaksesnya.
- `.env` berisi key, sudah di-ignore git; jangan di-commit.

---

## Batasan

- **Tanpa GPU NVIDIA semuanya jalan di CPU dan jauh lebih lambat.** Dengan GPU, WhisperX large-v3 jalan sekitar 8x realtime. Di CPU (termasuk Mac, karena ctranslate2 tidak punya backend GPU Apple) dipakai `large-v3-turbo`. Realtime copilot di CPU praktis kurang nyaman.
- **Mac Intel tidak didukung** (wheel torch dan ctranslate2 yang dibutuhkan WhisperX tidak ada untuk Intel). Dukungan Apple Silicon belum diuji di Mac asli.
- **Berorientasi bahasa Indonesia.** Prompt ringkasan berbahasa Indonesia dan bahasa transkripsi default-nya `id`. Untuk bahasa lain, ubah prompt (`SYSTEM_PROMPT` di `batch/pipeline.py`, preset di `supermeet.py`), `RTSM_LANGUAGE` untuk realtime, dan konstanta `LANGUAGE` di `batch/pipeline.py` serta `kuliah/config.py`.
- **Diarization butuh akun Hugging Face:** terima syarat model pyannote dan isi `HF_TOKEN`. Tanpa itu, tidak ada label speaker.
- **Klien capture hanya untuk Windows** (WASAPI loopback). Di Linux, pakai tombol mic di UI.
- **Akurasi realtime di bawah batch.** Turbo + potongan per-utterance lebih sering meleset, terutama nama orang dan akronim. Itu sebabnya ada handoff arsip.
- **Tidak ada diarization realtime** (pyannote terlalu berat untuk streaming di kartu kecil).
- **Satu sesi realtime dalam satu waktu.**
- Free tier Gemini punya limit harian; hari yang padat bisa membuat file masuk `audio/failed/`.

Pengukuran referensi pada laptop RTX 4050 (6 GB VRAM):

| | large-v3-turbo | large-v3 |
|---|---|---|
| Kecepatan | 19.5x realtime | 8.4x realtime |
| VRAM | 1219 MiB | 1898 MiB |
| Load | 20 dtk | 45 dtk |
| Lag realtime (audio 90 dtk) | 1.4 dtk rata-rata | 2.5 dtk rata-rata |

Keduanya mengikuti audio tanpa menumpuk. Turbo jadi default realtime karena lag lebih rendah dan load lebih cepat; large-v3 dipakai alur batch dan kuliah untuk akurasi.

---

## Troubleshooting

| Gejala | Penyebab / solusi |
|---|---|
| Ada yang kurang saat setup | Jalankan `./supermeet.py doctor`; ia menunjukkan apa yang belum beres |
| `ERROR: GEMINI_API_KEY not set` | Isi key di `.env`, atau jalankan dengan Transcribe only. Validasi key: `doctor --online` |
| `WARNING: HF_TOKEN not set` | Diarization dilewati. Isi `HF_TOKEN` dan terima syarat model pyannote di huggingface.co |
| File ada di `audio/failed/` | Lihat `logs/watcher.log`, beresin penyebabnya, taruh balik ke inbox |
| Error `429` dari Gemini | Limit free tier. Per menit: ditunggu otomatis. Harian: taruh balik besok |
| Watcher bilang `BUSY` / "GPU dipakai ..." | Proses lain memegang GPU. `./supermeet.py gpu` untuk melihat siapa; watcher mencoba lagi sendiri |
| `FATAL: inbox does not exist` | Buat foldernya, atau cek `TRANSCRIBE_CLOUD_ROOT` / `TRANSCRIBE_INBOX`. Cek: `python core/config.py` |
| Folder Google Drive tidak terbaca (WSL / Mac) | Aktifkan mode **Mirror files**, bukan Stream files. Di Mac, beri Terminal izin ke folder itu |
| Mac: PDF gagal dengan error `libpango` | `brew install pango`, lalu jalankan lewat `./supermeet.py` (ia mengatur `DYLD_FALLBACK_LIBRARY_PATH` sendiri) |
| Mic HP tidak bisa dipakai | Butuh HTTPS: `./supermeet.py serve --https` |
| Panel di iPad terbuka tapi tidak pernah tersambung | Sertifikat belum dipercaya penuh; ikuti langkah pasang CA di atas |
| Panel kosong, tidak ada transkrip | Belum ada klien capture yang tersambung (lihat header UI) |
| Transkrip kuliah salah eja istilah | Tambah ke `istilah` di `kuliah/matkul.toml`, lalu jalankan ulang dengan `--force` |

---

## Struktur proyek

```
supermeet.py            satu pintu masuk
setup.sh                setup sekali jalan: .venv, dependensi, .env, doctor
.env.example            template konfigurasi
requirements.txt
make-cert.sh            CA lokal + sertifikat server untuk --https
core/                   dipakai bersama
  config.py             path, folder inbox/output
  env.py                pembaca .env
  doctor.py             checklist setup
  filters.py            filter halusinasi (satu definisi untuk semua alur)
  gpu.py                lease GPU lintas proses (no-op tanpa CUDA)
  device.py             pilihan device + model Whisper (CUDA / CPU)
  pdf.py  timestamps.py  quiet.py
batch/
  pipeline.py           WhisperX -> pyannote -> filter -> Gemini -> PDF
  watcher.sh            pemantau inbox
realtime/
  app.py  stt.py  brain.py  session.py  config.py
  kuliah_web.py         halaman /kuliah
  web/                  UI browser
  ingest/win_capture.py klien capture Windows
kuliah/                 mode kuliah (rencana, transkripsi, output, matkul.toml)
tests/                  unit test
```

Dibuat saat berjalan (di-ignore git): `.venv/`, `whisper/`, `audio/`, `transcripts/` (cache transkrip), `sessions/`, `logs/`, `certs/`, `kuliah-uploads/`, `kuliah-output/`.

## Tes

```bash
python -m unittest discover -s tests -t .
```

## Lisensi

[MIT](LICENSE)
