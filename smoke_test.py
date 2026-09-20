"""Uji asap end-to-end tanpa internet dan tanpa model Whisper.

Membuat video sintetis, mengunggahnya, menjalankan analisis dengan transkrip palsu, lalu merender klip
sungguhan lewat FFmpeg + OpenCV. Jalankan dari folder server/:  python tests/smoke_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

tmp = Path(tempfile.mkdtemp(prefix="clipforge_test_"))
os.environ["CLIPFORGE_DATA"] = str(tmp / "data")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import app as app_module  # noqa: E402
import pipeline  # noqa: E402

DEMO = tmp / "demo.mp4"
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=24",
     "-f", "lavfi", "-i", "sine=frequency=440:duration=24", "-c:v", "libx264", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-shortest", str(DEMO)], check=True)

# Transkrip palsu: kalimat 4 detik, tiap kata 0.4 detik.
WORDS = "jadi sebenarnya kesalahan pertama yang bikin rugi adalah tidak menghitung angka 10 persen dengan benar ya".split()


def fake_transcript(sid, video, job, stage, language=None):
    segs, t = [], 0.0
    while t < 22:
        words = []
        for k in range(10):
            words.append({"start": t + k * 0.4, "end": t + k * 0.4 + 0.35, "word": WORDS[k % len(WORDS)]})
        text = " ".join(w["word"] for w in words) + ("?" if int(t) % 8 == 0 else ".")
        segs.append({"start": t, "end": t + 4.0, "text": text, "words": words})
        t += 4.0
    job.set(stage, 1.0)
    tr = {"language": "id", "duration": 24.0, "segments": segs}
    (pipeline.source_dir(sid) / "transcript.json").write_text(json.dumps(tr), encoding="utf-8")
    return tr


pipeline.get_transcript = fake_transcript
client = TestClient(app_module.app)


def wait(job_id, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["status"] == "done":
            return j
        if j["status"] == "error":
            raise SystemExit(f"JOB GAGAL: {j['error']}")
        time.sleep(0.3)
    raise SystemExit("job timeout")


def check(cond, msg):
    print(("  ok   " if cond else "  GAGAL ") + msg)
    if not cond:
        raise SystemExit(1)


print("health")
h = client.get("/api/health").json()
check(h["ok"] and h["features"]["ffmpeg"], f"server hidup, fitur: {h['features']}")

print("tautan non-YouTube ditolak")
r = client.post("/api/source", json={"url": "http://169.254.169.254/latest/meta-data"})
check(r.status_code == 400, f"status {r.status_code}: {r.json().get('detail')}")

print("unggah")
with open(DEMO, "rb") as f:
    r = client.post("/api/upload", files={"file": ("demo.mp4", f, "video/mp4")})
check(r.status_code == 200, r.text[:120])
sid = r.json()["sourceId"]
check(abs(r.json()["duration"] - 24) < 1, f"durasi terbaca {r.json()['duration']:.1f}s")

print("unggah file bukan video ditolak")
r = client.post("/api/upload", files={"file": ("x.mp4", b"bukan video", "video/mp4")})
check(r.status_code == 400, r.json()["detail"])

print("analisis")
jid = client.post("/api/analyze", json={"sourceId": sid, "count": 2, "length": 8}).json()["jobId"]
res = wait(jid)["result"]
clips = res["clips"]
check(len(clips) == 2, f"{len(clips)} klip: " + ", ".join(f"{c['start']}s+{c['len']}s skor {c['score']}" for c in clips))
check(all(c["thumb"] for c in clips), "thumbnail dibuat")
t = client.get(clips[0]["thumb"])
check(t.status_code == 200 and t.headers["content-type"] == "image/jpeg", "thumbnail bisa diambil")
m = client.get(res["mediaUrl"])
check(m.status_code == 200 and len(m.content) > 1000, "media sumber bisa diputar")

print("kamera virtual")
import numpy as np  # noqa: E402
import render as render_module  # noqa: E402

ts = [i * 0.25 for i in range(41)]                       # 10 detik, wajah bergeser 0.3 -> 0.7
xs = [0.3 + 0.4 * (i / 40) for i in range(41)]
ys = [0.4] * 41
fx, fy, z, drift = render_module.camera_path(300, 10.0, ts, xs, ys, True, "ken", [0.0])
check(fx[0] < 0.36 and fx[-1] > 0.55, f"kamera mengikuti wajah (x {fx[0]:.2f} -> {fx[-1]:.2f})")
check(float(np.max(np.abs(np.diff(fx)))) < 0.005, "gerak halus, tidak melompat")
check(abs(z[0] - 1.0) < 0.01 and abs(z[-1] - 1.15) < 0.01, f"zoom pelan 1.00 -> {z[-1]:.2f}")
fx2, _, z2, _ = render_module.camera_path(300, 10.0, [], [], [], True, "none", [])
check(abs(fx2[0] - 0.5) < 1e-9 and z2[0] == 1.0, "tanpa wajah: bingkai tetap di tengah")
ass = render_module.build_ass([{"w": "halo", "s": 0.0, "e": 0.4}, {"w": "dunia.", "s": 0.4, "e": 0.9}], 1080, 1920, "karaoke")
check("\\k40" in ass and "PlayResY: 1920" in ass, "berkas teks ASS terbentuk")

print("render 3 gaya teks dan 3 rasio")
combos = [("9:16", "karaoke", "ken", True), ("1:1", "bounce", "punch", True), ("16:9", "block", "drift", False)]
for aspect, cap, motion, track in combos:
    c = clips[0]
    body = {"sourceId": sid, "clips": [{"id": c["id"], "title": c["title"], "start": c["start"],
            "end": c["start"] + c["len"], "aspect": aspect, "motion": motion, "subjectTracking": track,
            "captions": {"style": cap}}]}
    jid = client.post("/api/render", json=body).json()["jobId"]
    files = wait(jid)["result"]["files"]
    out = client.get(files[0]["url"])
    check(out.status_code == 200, f"{aspect} {cap}/{motion}: {files[0]['size'] // 1024} KB")
    p = tmp / f"out_{aspect.replace(':', 'x')}.mp4"
    p.write_bytes(out.content)
    info = pipeline.probe(p)
    want = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}[aspect]
    check((info["width"], info["height"]) == want, f"resolusi {info['width']}x{info['height']}")
    check(abs(info["duration"] - c["len"]) < 0.5 and info["has_audio"], f"durasi {info['duration']:.1f}s, audio ada")

print("render tanpa teks")
c = clips[1]
body = {"sourceId": sid, "clips": [{"id": c["id"], "start": c["start"], "end": c["start"] + c["len"],
        "aspect": "9:16", "motion": "none", "subjectTracking": False, "captions": None}]}
wait(client.post("/api/render", json=body).json()["jobId"])
check(True, "selesai")

print("keamanan jalur file")
check(client.get("/api/renders/..%2Fmeta.json").status_code in (400, 404), "traversal ditolak")
check(client.get("/api/media/../../etc").status_code in (400, 404), "id sumber liar ditolak")

print("\nSEMUA UJI LULUS  (contoh keluaran:", tmp, ")")
