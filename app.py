"""ClipForge server. Jalankan: uvicorn app:app --host 0.0.0.0 --port 8000"""
import importlib.util
import json
import mimetypes
import os
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import jobs
import pipeline
import render
import speakers

app = FastAPI(title="ClipForge", version="1.0")

if config.CORS_ORIGINS:
    app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])

ALLOWED_EXT = {".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi"}
NAME_RE = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"


# ---------------------------------------------------------------- model permintaan

class SourceReq(BaseModel):
    url: str = Field(..., max_length=500)


class AnalyzeReq(BaseModel):
    sourceId: str
    count: int = Field(6, ge=1, le=12)
    length: int = Field(30, ge=8, le=120)
    language: Optional[str] = Field(None, max_length=8)


class RenderClip(BaseModel):
    id: str = Field(..., max_length=32)
    title: str = Field("", max_length=200)
    start: float = Field(..., ge=0)
    end: float
    aspect: str = "9:16"
    motion: str = "ken"
    subjectTracking: bool = True
    framing: Optional[str] = Field(None, pattern="^(follow|speaker|split|center)$")
    switchStyle: str = Field("cut", pattern="^(cut|glide)$")
    speakers: Optional[dict] = None
    captions: Optional[dict] = None


class SpeakersReq(BaseModel):
    sourceId: str
    start: float = Field(..., ge=0)
    end: float


class RenderReq(BaseModel):
    sourceId: str
    clips: List[RenderClip] = Field(..., min_length=1, max_length=12)


# ---------------------------------------------------------------- rute

@app.get("/api/health")
def health():
    def has(mod: str) -> bool:
        return importlib.util.find_spec(mod) is not None

    has_llm = bool(config.LLM_API_KEY) and (
        (bool(os.getenv("GEMINI_API_KEY")) and has("google.generativeai")) or
        (bool(os.getenv("ANTHROPIC_API_KEY")) and has("anthropic"))
    )
    features = {
        "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
        "whisper": has("faster_whisper"),
        "ytdlp": has("yt_dlp"),
        "opencv": has("cv2"),
        "llm": has_llm,
        "mode": "cloud_ai" if has_llm else "local_standalone",
    }
    names = {"ffmpeg": "ffmpeg", "whisper": "faster-whisper", "ytdlp": "yt-dlp", "opencv": "opencv"}
    return {"ok": True, "version": app.version, "features": features,
            "missing": [names[k] for k in names if not features[k]]}


@app.post("/api/source")
def register_youtube(req: SourceReq):
    try:
        m = pipeline.fetch_youtube_meta(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    sid = m["videoId"]
    old = {}
    try:
        old = pipeline.load_meta(sid)
    except (FileNotFoundError, ValueError):
        pass
    meta = {**old, "kind": "youtube", "title": m["title"], "duration": m["duration"],
            "url": req.url.strip(), "videoId": sid, "thumbnail": m["thumbnail"]}
    pipeline.save_meta(sid, meta)
    return {"sourceId": sid, "kind": "youtube", "title": m["title"], "duration": m["duration"],
            "thumbnail": m["thumbnail"]}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, "Format tidak didukung. Gunakan MP4, MOV, WebM, MKV, M4V, atau AVI.")
    sid = "up" + uuid.uuid4().hex[:10]
    d = pipeline.source_dir(sid)
    d.mkdir(parents=True)
    dest = d / f"source{ext}"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    try:
        with open(dest, "wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File terlalu besar. Batas server: {config.MAX_UPLOAD_MB} MB.")
                f.write(chunk)
        try:
            info = await run_in_threadpool(pipeline.probe, dest)
        except (ValueError, RuntimeError):
            raise HTTPException(400, "File ini tidak bisa dibaca sebagai video.")
        if info["duration"] > config.MAX_SOURCE_MIN * 60:
            raise HTTPException(413, f"Video terlalu panjang. Batas server: {config.MAX_SOURCE_MIN} menit.")
    except HTTPException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    title = Path(file.filename or "video").stem[:120]
    pipeline.save_meta(sid, {"kind": "upload", "title": title, "duration": info["duration"], "file": dest.name})
    return {"sourceId": sid, "kind": "upload", "title": title, "duration": info["duration"]}


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    try:
        pipeline.load_meta(req.sourceId)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    job = jobs.submit("analyze", pipeline.ANALYZE_STAGES,
                      lambda j: pipeline.run_analyze(j, req.sourceId, req.count, req.length, req.language))
    return {"jobId": job.id}


@app.post("/api/render")
def render_clips(req: RenderReq):
    try:
        meta = pipeline.load_meta(req.sourceId)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if not (pipeline.source_dir(req.sourceId) / "transcript.json").exists() and \
            any(c.captions for c in req.clips):
        raise HTTPException(409, "Analisis video ini belum selesai, jadi teks otomatis belum tersedia.")
    for c in req.clips:
        try:
            c.speakers = speakers.sanitize(c.speakers)
        except ValueError as e:
            raise HTTPException(400, f"Klip '{c.id}': {e}")
        if c.end - c.start < 1:
            raise HTTPException(400, f"Klip '{c.id}' terlalu pendek.")
        if c.end - c.start > 180:
            raise HTTPException(400, f"Klip '{c.id}' terlalu panjang (maksimal 180 detik).")
    stages = [f"Klip {i + 1}: {(c.title or c.id)[:40]}" for i, c in enumerate(req.clips)]

    def work(job: jobs.Job) -> dict:
        src = pipeline.video_path(req.sourceId)
        if src is None:
            raise FileNotFoundError("File sumber tidak ada di server.")
        info = pipeline.probe(src)
        tr_path = pipeline.source_dir(req.sourceId) / "transcript.json"
        tr = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.exists() else {"segments": [], "duration": 0}
        files = []
        for i, c in enumerate(req.clips):
            job.set(i, 0.0)
            name = f"{req.sourceId}_{c.id}_{uuid.uuid4().hex[:6]}.mp4"
            out = config.RENDERS_DIR / name
            work_dir = config.DATA_DIR / "work" / f"{job.id}_{c.id}"
            try:
                render.render_clip(src, info, c, tr, out, work_dir, lambda p, i=i: job.set(i, p))
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            files.append({"id": c.id, "title": c.title or c.id, "url": f"/api/renders/{name}",
                          "size": out.stat().st_size})
        return {"files": files}

    job = jobs.submit("render", stages, work)
    return {"jobId": job.id}


SPEAKER_STAGES = ["Mendeteksi wajah", "Membaca gerak mulut", "Menentukan giliran bicara"]


@app.post("/api/speakers")
def detect_speakers(req: SpeakersReq):
    """Analisis pembicara aktif untuk satu rentang. Hasilnya dipakai editor (dan bisa dikoreksi) lalu dikirim
    kembali saat render supaya render tidak menganalisis ulang."""
    try:
        pipeline.load_meta(req.sourceId)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if req.end - req.start < 3:
        raise HTTPException(400, "Rentang terlalu pendek untuk menganalisis pembicara (minimal 3 detik).")
    if req.end - req.start > 180:
        raise HTTPException(400, "Rentang terlalu panjang (maksimal 180 detik).")

    def work(job: jobs.Job) -> dict:
        src = pipeline.video_path(req.sourceId)
        if src is None:
            raise FileNotFoundError("File sumber tidak ada di server.")
        info = pipeline.probe(src)
        tr_path = pipeline.source_dir(req.sourceId) / "transcript.json"
        tr = json.loads(tr_path.read_text(encoding="utf-8")) if tr_path.exists() else None
        end = min(req.end, info["duration"] or req.end)
        return speakers.analyze(src, info, req.start, end, tr, progress=job.set,
                                thumb_dir=config.THUMBS_DIR,
                                thumb_prefix=f"spk_{req.sourceId}_{int(req.start * 10)}")

    job = jobs.submit("speakers", SPEAKER_STAGES, work)
    return {"jobId": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job tidak ditemukan (mungkin server dijalankan ulang).")
    return job.to_dict()


def _safe_file(directory: Path, name: str) -> Path:
    if not name or any(ch not in NAME_RE for ch in name) or name.startswith("."):
        raise HTTPException(400, "Nama file tidak valid.")
    p = directory / name
    if not p.is_file():
        raise HTTPException(404, "File tidak ditemukan.")
    return p


@app.get("/api/thumbs/{name}")
def get_thumb(name: str):
    return FileResponse(_safe_file(config.THUMBS_DIR, name), media_type="image/jpeg")


@app.get("/api/renders/{name}")
def get_render(name: str):
    p = _safe_file(config.RENDERS_DIR, name)
    return FileResponse(p, media_type="video/mp4", filename=p.name)


@app.get("/api/media/{sid}")
def get_media(sid: str):
    try:
        p = pipeline.video_path(sid)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if p is None:
        raise HTTPException(404, "Video belum diunduh. Jalankan analisis terlebih dulu.")
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "video/mp4")


# Antarmuka web: file HTML disajikan dari folder static/ sehingga API dan halaman satu asal (tanpa CORS).
if config.STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(" ClipForge Web siap digunakan di: http://localhost:8000")
    print(" Mode: Mandiri Lokal (Tanpa API Key)")
    print(" Tekan Ctrl + C untuk menghentikan server")
    print("=======================================================\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
