# ClipForge

Ubah video panjang (tautan YouTube atau file sendiri) menjadi klip pendek vertikal dengan gerakan kamera otomatis dan teks per kata.

```
Browser (static/index.html)  ──/api/*──▶  FastAPI (server/app.py)
                                            ├─ yt-dlp            unduh YouTube
                                            ├─ faster-whisper    transkripsi kata-per-kata
                                            ├─ heuristik / LLM   nilai dan namai momen
                                            └─ FFmpeg + OpenCV   render: kamera virtual + teks (ASS)
```

Halaman web disajikan oleh server yang sama, jadi tidak perlu pengaturan CORS.

## Menjalankan

**Docker (paling mudah)**

```bash
cp .env.example .env        # opsional
docker compose up --build
# buka http://localhost:8000
```

**Langsung di mesin**

Butuh Python 3.10+ dan FFmpeg (dengan libass; bawaan hampir semua distribusi).

```bash
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000
# buka http://localhost:8000
```

Lencana di kanan atas halaman menunjukkan status: hijau berarti server lengkap, kuning menyebut komponen yang belum terpasang, merah muda berarti halaman berjalan sebagai simulasi tanpa server.

Model Whisper diunduh otomatis saat transkripsi pertama (`small` ≈ 500 MB).

## Alur kerja

1. **Muat sumber.** Tautan YouTube (`POST /api/source`) atau unggah file (`POST /api/upload`).
2. **Cari klip terbaik** (`POST /api/analyze`). Server mengunduh, menyalin audio menjadi teks, membuat kandidat jendela waktu di batas kalimat, menilainya, lalu merapatkan titik potong ke kata pertama dan terakhir. Progres dibaca lewat `GET /api/jobs/{id}`.
3. **Edit** di halaman: rentang waktu, rasio, gerakan, gaya teks, bingkai, dan (untuk podcast) pembicara aktif.
4. **Render** (`POST /api/render`). Hasilnya MP4 H.264 + AAC di `/api/renders/...`.

### Penilaian momen

Tanpa kunci API, server memberi skor heuristik: kerapatan ucapan, pertanyaan, angka, kata pemicu, dan penekanan. Dengan `ANTHROPIC_API_KEY`, 3× jumlah klip kandidat teratas dikirim ke model untuk dipilih dan diberi judul. Kalau panggilan gagal, server otomatis kembali ke heuristik. Transkrip diperlakukan sebagai data, bukan instruksi.

### AI Motion

| Gerakan | Efek |
|---|---|
| Zoom pelan | Kamera mendekat 100% → 115% sepanjang klip |
| Punch-in | Zoom cepat di awal tiap kalimat, lalu mengendur |
| Parallax | Zoom 112% dengan geseran horizontal lambat |
| Diam | Tanpa zoom |

### Bingkai

| Bingkai | Perilaku |
|---|---|
| Ikuti wajah | Kamera mengikuti wajah terbesar (deteksi 4× per detik, Haar cascade) dengan zona mati dan peredaman supaya tidak bergetar |
| Pembicara aktif | Kamera berpindah ke orang yang sedang bicara (potong, atau perpindahan halus 0,35 detik) |
| Layar terbagi | Khusus 9:16. Dua orang ditumpuk atas-bawah; yang bicara terang dan berbingkai kuning, yang mendengarkan diredupkan; teks di garis pemisah |
| Tengah | Tanpa pelacakan |

Wajah diposisikan sekitar 40% dari tepi atas bingkai vertikal. Bila mode Pembicara aktif atau Layar terbagi menemukan kurang dari dua wajah, render otomatis memakai Ikuti wajah.

### Pembicara aktif (podcast dua orang)

`server/speakers.py` bekerja dalam lima langkah:

1. Bingkai kecil (10 fps) didekode dan wajah dideteksi (Haar cascade frontal; varian kontras kedua dipakai bila jumlah wajah kurang dari yang diharapkan).
2. Deteksi dikelompokkan menjadi satu jalur per orang. Jalur dengan kehadiran kurang dari 40% jalur terkuat dibuang sebagai deteksi palsu.
3. Gerak di area mulut diukur per jalur, dikurangi gerak di area dahi supaya anggukan tidak dihitung sebagai bicara.
4. Transkrip menentukan **kapan** ada ucapan; energi mulut menentukan **siapa** yang bicara di tiap jendela ±2 detik.
5. Histeresis: ucapan lebih pendek dari 0,9 detik (sahutan seperti "iya") tidak memindahkan kamera, dan titik pergantian ditajamkan ke saat pembicara baru mulai bicara.

Di editor, tekan **Deteksi pembicara** untuk melihat foto tiap orang dan bilah giliran bicara. **Ketuk sebuah segmen untuk menukar pembicaranya** bila deteksi keliru; koreksi itu ikut terkirim saat render. Tanpa menekan tombol itu, render tetap mendeteksi sendiri. Waktu analisis kira-kira 1,3× durasi klip di CPU biasa.

**Teks:** Karaoke (kata aktif berwarna kuning), Bounce (kata aktif membesar), Blok (kotak solid, kata aktif merah muda).

## Konfigurasi

Lihat `.env.example`. Yang paling berpengaruh:

| Variabel | Fungsi |
|---|---|
| `WHISPER_MODEL` | `tiny`…`large-v3`. Lebih besar = lebih akurat dan lebih lambat |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | Pakai `cuda` + `float16` bila ada GPU NVIDIA |
| `ANTHROPIC_API_KEY` | Mengaktifkan penilaian oleh AI |
| `MAX_UPLOAD_MB`, `MAX_SOURCE_MIN` | Batas ukuran unggahan dan durasi sumber |
| `CLIPFORGE_WORKERS` | Jumlah job paralel (render memakan CPU) |

## Uji

```bash
pip install -r server/requirements-dev.txt
python server/tests/smoke_test.py
```

Uji ini membuat video sintetis, mengunggahnya, menganalisis dengan transkrip palsu, lalu merender ketiga rasio dan ketiga gaya teks sungguhan. Tidak butuh internet atau model Whisper.

Uji pembicara aktif (butuh `pip install scikit-image`; memakan beberapa menit):

```bash
python server/tests/speaker_test.py             # adegan mudah dan sulit
python server/tests/speaker_test.py --hard-only
```

Uji ini memakai adegan sintetis: dua orang dari foto astronaut NASA (domain publik) dengan gerak mulut yang disimulasikan, pendengar mengangguk, dan satu sahutan singkat. Adegan sulit menambah derau, mulut kecil, dan kompresi kuat. **Ini belum mewakili rekaman podcast nyata.**

## Pemecahan masalah

- **Unduhan YouTube gagal / "Sign in to confirm you're not a bot".** yt-dlp cepat usang: `pip install -U yt-dlp`. Beberapa video atau alamat IP server datacenter memerlukan cookie login; lihat dokumentasi yt-dlp tentang `cookiefile`.
- **"detektor wajah dihapus di OpenCV 5" di log.** Pasang `opencv-python-headless<5`. Render tetap jalan, hanya tanpa pelacakan.
- **Teks tampil dengan font aneh.** Pasang font (`fonts-liberation` di Debian/Ubuntu) atau atur `CAPTION_FONT` ke font yang terpasang.
- **Hasil render agak buram.** Sumber 1080p dipotong ke 9:16 lalu diperbesar ke 1080×1920. Itu batas fisik sumbernya; unggah 4K bila ada.
- **Transkrip lambat.** Turunkan `WHISPER_MODEL` ke `base`, atau pakai GPU.
- **Job hilang setelah server restart.** Status job disimpan di memori. Sumber, transkrip, dan hasil render tetap ada di `CLIPFORGE_DATA`.

## Sebelum dibuka ke publik

Server ini titik awal, bukan produk jadi:

- **Belum ada autentikasi.** Siapa pun yang bisa mengakses alamatnya bisa mengunggah dan merender. Letakkan di belakang login atau jaringan privat.
- **Belum ada pembatasan laju atau kuota per pengguna.** Render dan transkripsi mahal.
- **Belum ada pembersihan otomatis** folder `sources/`, `renders/`, `thumbs/`. Tambahkan cron penghapus berkas lama.
- **Hak cipta.** Gunakan hanya untuk video milikmu atau yang kamu punya izin untuk dipotong dan diunggah ulang. Mengunduh dari YouTube dapat melanggar ketentuan layanannya.

## Batasan yang diketahui

- Deteksi wajah Haar hanya menangkap wajah menghadap depan dan cukup besar di bingkai. Pembicara yang menoleh ke samping akan hilang dari jalurnya. Detektor berbasis jaringan saraf (misalnya YuNet) akan lebih tangguh; belum dipasang karena modelnya harus diunduh terpisah.
- Ambang keputusan pembicara aktif (`MIN_ENERGY`, `SWITCH_CONF`, dan sejenisnya di `speakers.py`) dikalibrasi dari data sintetis. Rekaman nyata dengan pencahayaan, kompresi, atau gaya bicara berbeda mungkin butuh penyesuaian. Bila hasil sering keliru, koreksi manual di editor adalah jalan keluarnya.
- Pembicara aktif membaca gerak mulut, jadi butuh dua wajah dalam satu bingkai. Untuk podcast yang sudah dipotong per kamera, pakai Ikuti wajah.
- Ucapan yang tumpang tindih tidak dipisahkan: kamera mengikuti pembicara yang gerak mulutnya lebih kuat.
- Maksimum empat wajah per analisis; Layar terbagi memakai dua yang paling sering terlihat.
- Siaran langsung dan playlist tidak didukung.
