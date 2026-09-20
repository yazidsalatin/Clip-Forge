"""Uji pembicara aktif dengan adegan podcast sintetis.

Adegan: dua orang duduk berdampingan (foto astronaut NASA dari scikit-image, domain publik; orang kedua
dicerminkan dan diberi rona berbeda). Gerak mulut disimulasikan dengan elips gelap yang membuka-menutup,
pendengar mengangguk, dan ada sahutan singkat. Jalankan dari folder server/:  python tests/speaker_test.py
Butuh: pip install scikit-image
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

tmp = Path(tempfile.mkdtemp(prefix="clipforge_spk_"))
os.environ["CLIPFORGE_DATA"] = str(tmp / "data")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from skimage import data as skdata  # noqa: E402

import pipeline  # noqa: E402
import render  # noqa: E402
import speakers  # noqa: E402

FPS, DUR = 30, 16.0
# Kebenaran dasar: (mulai, akhir, pembicara). 0 = kiri, 1 = kanan. Sahutan pendengar tidak memindahkan kamera.
TRUTH = [(0.0, 5.0, 0), (5.0, 9.5, 1), (9.5, 15.0, 0)]
BACKCHANNEL = (11.0, 11.5, 1)


def check(cond, msg):
    print(("  ok   " if cond else "  GAGAL ") + msg)
    if not cond:
        raise SystemExit(1)


def make_video(path: Path, hard: bool):
    base = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2BGR)
    left = base.copy()
    right = cv2.flip(base, 1)
    right = cv2.convertScaleAbs(right, alpha=0.92, beta=6)
    right[..., 0] = np.clip(right[..., 0] * 0.65, 0, 255)      # lebih sedikit biru
    right[..., 2] = np.clip(right[..., 2] * 1.20, 0, 255)      # lebih banyak merah
    left[..., 0] = np.clip(left[..., 0] * 1.2, 0, 255)         # kiri kebiruan
    mouth_l, mouth_r = (287, 340), (992, 340)
    ax = (8, 4) if hard else (16, 11)
    bob = 3.0 if hard else 1.8
    rng = np.random.default_rng(7)

    def speaking(k, t):
        return any(a <= t < b and s == k for a, b, s in TRUTH + [BACKCHANNEL])

    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x720", "-r", str(FPS),
         "-i", "pipe:0", "-f", "lavfi", "-i", f"sine=frequency=220:duration={DUR}", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", "28" if hard else "20", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)], stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        t = i / FPS
        canvas = np.full((720, 1280, 3), (52, 40, 46), np.uint8)
        for k, (img, cx, mouth) in enumerate(((left, 320, mouth_l), (right, 960, mouth_r))):
            dy = int(round(bob * np.sin(2 * np.pi * (0.8 + 0.3 * k) * t + k)))
            y0 = 204 + dy
            canvas[y0:y0 + 512, cx - 256:cx + 256] = img
            if speaking(k, t):
                op = abs(np.sin(2 * np.pi * 3.1 * t + k)) * (0.5 + 0.5 * abs(np.sin(2 * np.pi * 1.3 * t)))
                cv2.ellipse(canvas, (mouth[0], mouth[1] + dy), (ax[0], max(1, int(ax[1] * op) + 1)), 0, 0, 360,
                            (25, 20, 70), -1)
        if hard:
            canvas = np.clip(canvas + rng.normal(0, 3.0, canvas.shape), 0, 255).astype(np.uint8)
        enc.stdin.write(canvas.tobytes())
    enc.stdin.close()
    enc.wait()


def transcript():
    segs = []
    for a, b, _ in TRUTH + [BACKCHANNEL]:
        words, t = [], a + 0.2
        while t < b - 0.3:
            words.append({"start": t, "end": t + 0.3, "word": "kata"})
            t += 0.4
        segs.append({"start": a, "end": b, "text": "kata " * len(words), "words": words})
    segs.sort(key=lambda s: s["start"])
    return {"language": "id", "duration": DUR, "segments": segs}


def run(hard: bool):
    label = "sulit (derau, mulut kecil, kompresi kuat)" if hard else "mudah"
    print(f"\n=== adegan {label} ===")
    video = tmp / ("podcast_hard.mp4" if hard else "podcast_easy.mp4")
    make_video(video, hard)
    info = pipeline.probe(video)
    tr = transcript()
    thumb_dir = tmp / "thumbs"
    thumb_dir.mkdir(exist_ok=True)

    print("analisis")
    res = speakers.analyze(video, info, 0.0, DUR, tr, thumb_dir=thumb_dir, thumb_prefix="t")
    check(len(res["tracks"]) == 2, f"dua orang terdeteksi: x = {[t['x'] for t in res['tracks']]}")
    xs = [t["x"] for t in res["tracks"]]
    check(xs[0] < 0.4 and xs[1] > 0.6, "posisi kiri dan kanan benar")
    check(all(t["thumb"] for t in res["tracks"]) and all((thumb_dir / Path(t["thumb"]).name).exists() for t in res["tracks"]),
          "foto wajah tiap pembicara dibuat")
    turns = res["turns"]
    print("   giliran:", [(t["start"], t["end"], t["track"]) for t in turns])
    check([t["track"] for t in turns] == [0, 1, 0], f"urutan pembicara benar (keyakinan {res['confidence']})")
    check(abs(turns[1]["start"] - 5.0) < 0.6 and abs(turns[2]["start"] - 9.5) < 0.6, f"batas giliran dalam 0,6 detik ({turns[1]['start']}, {turns[2]['start']})")
    check(not any(t["start"] > 10.5 and t["end"] < 12 and t["track"] == 1 for t in turns),
          "sahutan singkat 11,0-11,5 tidak memindahkan kamera")
    return video, info, tr, res


def frame_at(mp4: Path, t: float):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t}", "-i", str(mp4), "-frames:v", "1",
                        "-f", "image2pipe", "-vcodec", "png", "pipe:1"], capture_output=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)


class Spec:
    def __init__(self, **kw):
        self.id, self.title, self.start, self.end = "c1", "uji", 0.0, DUR
        self.aspect, self.motion, self.subjectTracking = "9:16", "ken", True
        self.framing, self.switchStyle, self.speakers = "speaker", "cut", None
        self.captions = {"style": "karaoke"}
        self.__dict__.update(kw)


def bias(img):
    """>0 kalau dominan biru (orang kiri), <0 kalau dominan merah (orang kanan)."""
    b, r = img[..., 0].mean(), img[..., 2].mean()
    return (b - r) / (b + r)


HARD_ONLY = '--hard-only' in sys.argv
if not HARD_ONLY:
    video, info, tr, res = run(hard=False)

    print("\nrender: pembicara aktif (potong)")
    out = tmp / "speaker_cut.mp4"
    render.render_clip(video, info, Spec(speakers=None), tr, out, tmp / "w1", lambda p: None)
    pi = pipeline.probe(out)
    check((pi["width"], pi["height"]) == (1080, 1920) and abs(pi["duration"] - DUR) < 0.5, "1080x1920, durasi sesuai")
    f_a, f_b, f_c = frame_at(out, 2.5), frame_at(out, 7.0), frame_at(out, 12.5)
    print(f"   bias warna: t=2.5 -> {bias(f_a):+.3f}, t=7 -> {bias(f_b):+.3f}, t=12.5 -> {bias(f_c):+.3f}")
    check(bias(f_a) > bias(f_b) + 0.03 and bias(f_c) > bias(f_b) + 0.03, "kamera di orang kiri saat A bicara, di kanan saat B bicara")
    cv2.imwrite(str(tmp / "cut_a.png"), f_a)
    cv2.imwrite(str(tmp / "cut_b.png"), f_b)

    print("render: pembicara aktif (halus) dengan data dari klien")
    out = tmp / "speaker_glide.mp4"
    data = speakers.sanitize({"tracks": res["tracks"], "turns": res["turns"]})
    render.render_clip(video, info, Spec(speakers=data, switchStyle="glide", motion="none"), tr, out, tmp / "w2", lambda p: None)
    check(out.exists() and out.stat().st_size > 10000, "selesai")

    print("render: layar terbagi")
    out = tmp / "split.mp4"
    render.render_clip(video, info, Spec(framing="split", speakers=data, motion="none",
                                         captions={"style": "bounce"}), tr, out, tmp / "w3", lambda p: None)
    pi = pipeline.probe(out)
    check((pi["width"], pi["height"]) == (1080, 1920), "1080x1920")
    s_a, s_b = frame_at(out, 2.5), frame_at(out, 7.0)
    top_a, bot_a = s_a[:960], s_a[960:]
    top_b, bot_b = s_b[:960], s_b[960:]
    print(f"   kecerahan t=2.5: atas {top_a.mean():.0f} bawah {bot_a.mean():.0f} | t=7: atas {top_b.mean():.0f} bawah {bot_b.mean():.0f}")
    check(top_a.mean() > bot_a.mean() and bot_b.mean() > top_b.mean(), "panel pembicara aktif lebih terang")
    check(bias(top_a) > bias(bot_a) + 0.1 and bias(top_b) > bias(bot_b) + 0.1, "atas = orang kiri, bawah = orang kanan")
    cv2.imwrite(str(tmp / "split_a.png"), s_a)
    cv2.imwrite(str(tmp / "split_b.png"), s_b)

    print("render: satu wajah -> jatuh ke 'ikuti wajah' tanpa gagal")
    one = {"tracks": [res["tracks"][0]], "turns": [{"start": 0, "end": DUR, "track": 0}]}
    out = tmp / "fallback.mp4"
    render.render_clip(video, info, Spec(framing="split", speakers=speakers.sanitize(one), motion="none"), tr, out,
                       tmp / "w4", lambda p: None)
    check(out.exists(), "render selesai")

    print("sanitize")
    for bad, why in (({"tracks": [{"id": 0, "x": 2, "y": .5}], "turns": []}, "posisi di luar 0..1"),
                     ({"tracks": [{"id": 0, "x": .5, "y": .5}], "turns": [{"start": 0, "end": 1, "track": 9}]}, "jalur tak ada"),
                     ({"tracks": "x", "turns": []}, "bentuk salah")):
        try:
            speakers.sanitize(bad)
            check(False, f"seharusnya menolak: {why}")
        except ValueError:
            check(True, f"menolak: {why}")


run(hard=True)
print("\nSEMUA UJI PEMBICARA LULUS (contoh keluaran:", tmp, ")")
