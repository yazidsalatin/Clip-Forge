"""Konfigurasi ClipForge. Semua nilai bisa diubah lewat variabel lingkungan."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

DATA_DIR = Path(os.getenv("CLIPFORGE_DATA", BASE_DIR / "data")).resolve()
SOURCES_DIR = DATA_DIR / "sources"
RENDERS_DIR = DATA_DIR / "renders"
THUMBS_DIR = DATA_DIR / "thumbs"
STATIC_DIR = BASE_DIR / "static"

# Transkripsi (faster-whisper). "small" cukup untuk CPU; pakai "large-v3" + GPU untuk akurasi terbaik.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8")

# Penilaian momen dengan LLM (opsional). Tanpa ANTHROPIC_API_KEY, server memakai penilaian heuristik.
LLM_MODEL = os.getenv("CLIPFORGE_MODEL", "claude-sonnet-5")

# Batas pemakaian
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))
MAX_SOURCE_MIN = int(os.getenv("MAX_SOURCE_MIN", "240"))
WORKERS = int(os.getenv("CLIPFORGE_WORKERS", "2"))

# Font teks otomatis (harus terpasang di server; Arial dipetakan ke Liberation Sans di Linux)
CAPTION_FONT = os.getenv("CAPTION_FONT", "Arial")

# Asal yang boleh memanggil API dari halaman lain. Kosong = hanya same-origin.
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

for _d in (SOURCES_DIR, RENDERS_DIR, THUMBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
