"""Render satu klip.

Alur: FFmpeg mendekode bingkai sumber -> OpenCV memotong/menskala tiap bingkai mengikuti kamera virtual
(zoom, geser, pelacakan wajah) -> FFmpeg menyandikan hasilnya bersama audio dan teks per kata (ASS).
"""
import re
import subprocess
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import config
import speakers as speakers_mod

FPS = 30
SIZES = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}

AMBER = "3CB4FF"     # BGR untuk #FFB43C
MAGENTA = "8A5CFF"   # BGR untuk #FF5C8A


# ---------------------------------------------------------------- kata & teks (ASS)

def words_in_range(tr: dict, start: float, end: float) -> List[dict]:
    dur = end - start
    out = []
    for seg in tr["segments"]:
        if seg["end"] < start - 0.5 or seg["start"] > end + 0.5:
            continue
        for w in seg["words"]:
            if w["start"] >= start - 0.05 and w["end"] <= end + 0.05:
                s = min(max(0.0, w["start"] - start), dur)
                e = min(max(s + 0.05, w["end"] - start), dur)
                out.append({"w": w["word"], "s": s, "e": e})
    return out


def _ts(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(w: str) -> str:
    return w.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def _group(words: List[dict], max_words: int, max_chars: int) -> List[List[dict]]:
    groups, cur = [], []
    for w in words:
        if cur:
            chars = sum(len(x["w"]) + 1 for x in cur) + len(w["w"])
            gap = w["s"] - cur[-1]["e"]
            if len(cur) >= max_words or chars > max_chars or gap > 0.7 or re.search(r"[.!?…]$", cur[-1]["w"]):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: List[dict], out_w: int, out_h: int, style: str, center: bool = False) -> str:
    portrait = out_h > out_w
    square = out_h == out_w
    size = round(out_h * (0.048 if portrait else 0.06 if square else 0.055))
    margin_v = round(out_h * (0.16 if portrait else 0.08 if square else 0.07))
    margin_h = round(out_w * 0.08)
    align = 5 if center else 2             # 5 = tengah layar (pas di garis pemisah layar terbagi)
    if center:
        margin_v = 0
    max_words = 3 if (portrait or square) else 5
    max_chars = 22 if (portrait or square) else 38

    if style == "block":
        border, outline, shadow, outline_col, back = 3, round(size * 0.22), 0, "&H40301A10", "&H00000000"
    else:
        border, outline, shadow, outline_col, back = 1, max(3, round(size * 0.09)), 2, "&H00000000", "&H80000000"
    primary = f"&H00{AMBER}" if style == "karaoke" else "&H00FFFFFF"

    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {out_w}\nPlayResY: {out_h}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{config.CAPTION_FONT},{size},{primary},&H00FFFFFF,{outline_col},{back},-1,0,0,0,100,100,0,0,"
        f"{border},{outline},{shadow},{align},{margin_h},{margin_h},{margin_v},1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []
    for g in _group(words, max_words, max_chars):
        if style == "karaoke":
            parts = []
            for k, w in enumerate(g):
                nxt = g[k + 1]["s"] if k + 1 < len(g) else w["e"]
                cs = max(1, round((nxt - w["s"]) * 100))
                parts.append(f"{{\\k{cs}}}{_esc(w['w'])}")
            lines.append(f"Dialogue: 0,{_ts(g[0]['s'])},{_ts(g[-1]['e'] + 0.05)},Default,,0,0,0,,{' '.join(parts)}")
        else:
            active_col = MAGENTA if style == "block" else AMBER
            for k, w in enumerate(g):
                end = g[k + 1]["s"] if k + 1 < len(g) else w["e"] + 0.05
                texts = []
                for j, x in enumerate(g):
                    if j == k:
                        pop = "" if style == "block" else "\\fscx88\\fscy88\\t(0,110,\\fscx118\\fscy118)"
                        texts.append(f"{{\\c&H{active_col}&{pop}}}{_esc(x['w'])}{{\\r}}")
                    else:
                        texts.append(_esc(x["w"]))
                lines.append(f"Dialogue: 0,{_ts(w['s'])},{_ts(end)},Default,,0,0,0,,{' '.join(texts)}")
    return head + "\n".join(lines) + "\n"


# ---------------------------------------------------------------- kamera virtual

def _smooth(x: np.ndarray) -> np.ndarray:
    return x * x * (3 - 2 * x)


def sample_faces(src: Path, start: float, dur: float, dw: int, dh: int, cv2) -> Tuple[list, list, list]:
    """Deteksi wajah 4x per detik pada bingkai kecil. Kembalikan waktu dan pusat wajah ternormalisasi."""
    if not hasattr(cv2, "CascadeClassifier"):
        print("[clipforge] OpenCV ini tidak punya detektor wajah (dihapus di OpenCV 5); pakai opencv 4.x. "
              "Pelacakan subjek dilewati.")
        return [], [], []
    sw = 480
    sh = max(2, int(round(dh * sw / dw)))
    sh -= sh % 2
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an",
           "-vf", f"fps=4,scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    ts, xs, ys, i, fsz = [], [], [], 0, sw * sh
    while True:
        buf = proc.stdout.read(fsz)
        if len(buf) < fsz:
            break
        img = np.frombuffer(buf, np.uint8).reshape(sh, sw)
        faces = cascade.detectMultiScale(img, 1.15, 5, minSize=(int(sw * 0.05), int(sw * 0.05)))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            ts.append(i / 4.0)
            xs.append((x + w / 2) / sw)
            ys.append((y + h / 2) / sh)
        i += 1
    proc.stdout.close()
    proc.wait()
    return ts, xs, ys


def _follow(target: np.ndarray, dead: float, gain: float = 0.08) -> np.ndarray:
    """Kamera baru bergerak bila subjek keluar dari zona mati; geraknya diperhalus."""
    cam = float(target[0])
    out = np.empty_like(target)
    for i, t in enumerate(target):
        d = float(t) - cam
        if abs(d) > dead:
            cam += (d - np.sign(d) * dead) * gain
        out[i] = cam
    return out


def _zoom_curve(n: int, dur: float, motion: str, seg_starts: List[float], shot_t=None):
    """Kurva zoom dan geseran per bingkai. shot_t (detik sejak potongan terakhir) membuat zoom pelan
    diulang di tiap shot pada mode pembicara aktif."""
    t = np.arange(n) / FPS
    z = np.ones(n)
    drift = np.zeros(n)
    if motion == "ken":
        if shot_t is not None:
            z = 1.02 + 0.08 * _smooth(np.clip(shot_t / 6.0, 0, 1))
        else:
            z = 1.0 + 0.15 * _smooth(np.clip(t / max(dur, 1e-3), 0, 1))
    elif motion == "punch":
        z = np.full(n, 1.03)
        starts = np.array(sorted(seg_starts)) if seg_starts else np.array([])
        for i, tt in enumerate(t):
            k = np.searchsorted(starts, tt, side="right") - 1
            if k >= 0:
                dt = tt - starts[k]
                z[i] = 1.03 + 0.11 * min(dt / 0.1, 1.0) * np.exp(-dt / 0.8)
    elif motion == "drift":
        z = np.full(n, 1.12)
        drift = 0.03 * np.sin(2 * np.pi * t / 9.0)
    return z, drift


def camera_path(n: int, dur: float, ts, xs, ys, track: bool, motion: str, seg_starts: List[float]):
    t = np.arange(n) / FPS
    if track and ts:
        fx = _follow(np.interp(t, ts, xs), 0.035)
        fy = _follow(np.interp(t, ts, ys), 0.05)
    else:
        fx = np.full(n, 0.5)
        fy = np.full(n, 0.5)
    z, drift = _zoom_curve(n, dur, motion, seg_starts)
    return fx, fy, z, drift


def _rect(fx, fy, z, drift, dw, dh, aspect, track):
    if aspect <= dw / dh:
        bch, bcw = dh, dh * aspect
    else:
        bcw, bch = dw, dw / aspect
    cw, ch = bcw / z, bch / z
    cx = fx * dw + drift * dw
    cy = fy * dh + (0.10 * ch if track else 0.0)
    x0 = min(max(cx - cw / 2, 0.0), dw - cw)
    y0 = min(max(cy - ch / 2, 0.0), dh - ch)
    return x0, y0, cw, ch


# ---------------------------------------------------------------- tata letak

def _warp(cv2, frame, x0, y0, cw, ch, out_w, out_h):
    m = np.array([[cw / out_w, 0, x0], [0, ch / out_h, y0]], np.float32)
    interp = cv2.INTER_CUBIC if cw < out_w else cv2.INTER_LINEAR
    return cv2.warpAffine(frame, m, (out_w, out_h), flags=interp | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


class SingleLayout:
    """Satu bingkai: kamera virtual mengikuti (fx, fy) dengan zoom z."""

    def __init__(self, fx, fy, z, drift, dw, dh, out_w, out_h, track):
        self.fx, self.fy, self.z, self.drift = fx, fy, z, drift
        self.dw, self.dh, self.out_w, self.out_h, self.track = dw, dh, out_w, out_h, track
        self.aspect = out_w / out_h

    def compose(self, i, frame, cv2):
        x0, y0, cw, ch = _rect(self.fx[i], self.fy[i], self.z[i], self.drift[i], self.dw, self.dh,
                               self.aspect, self.track)
        return _warp(cv2, frame, x0, y0, cw, ch, self.out_w, self.out_h)


class SplitLayout:
    """Layar terbagi 9:16: dua pembicara ditumpuk atas-bawah. Yang sedang bicara terang dan berbingkai
    kuning; yang mendengarkan diredupkan."""

    def __init__(self, panels, z, act, dw, dh, out_w, out_h):
        self.z, self.act = z, act
        self.dw, self.dh = dw, dh
        self.pw, self.ph = out_w, out_h // 2
        aspect = self.pw / self.ph
        if aspect <= dw / dh:
            bch, bcw = dh, dh * aspect
        else:
            bcw, bch = dw, dw / aspect
        self.base = []
        for p in panels:
            face_w = p["w"] * dw
            cw = min(bcw, max(face_w / 0.30, bcw / 2.2))     # wajah kira-kira 30% lebar panel
            ch = cw / aspect
            self.base.append((p["x"] * dw, p["y"] * dh + 0.10 * ch, cw, ch))

    def compose(self, i, frame, cv2):
        parts = []
        for k, (cx, cy, cw, ch) in enumerate(self.base):
            cw2, ch2 = cw / self.z[i], ch / self.z[i]
            x0 = min(max(cx - cw2 / 2, 0.0), self.dw - cw2)
            y0 = min(max(cy - ch2 / 2, 0.0), self.dh - ch2)
            p = _warp(cv2, frame, x0, y0, cw2, ch2, self.pw, self.ph)
            a = float(self.act[i, k])
            p = cv2.convertScaleAbs(p, alpha=0.55 + 0.45 * a)
            th = int(round(9 * a))
            if th > 0:
                cv2.rectangle(p, (0, 0), (self.pw - 1, self.ph - 1), (0x3C, 0xB4, 0xFF), th * 2)
            parts.append(p)
        out = np.vstack(parts)
        out[self.ph - 3:self.ph + 3] = (0x2B, 0x10, 0x16)
        return out


def _activity(who: np.ndarray, ids: List[int], fps: int) -> np.ndarray:
    """Aktivitas 0..1 per panel, dilandaikan 0,2 detik supaya sorotan berpindah halus."""
    act = np.zeros((len(who), len(ids)))
    for k, tid in enumerate(ids):
        act[:, k] = (who == tid).astype(float)
    kernel = np.ones(max(1, int(0.2 * fps))) / max(1, int(0.2 * fps))
    for k in range(act.shape[1]):
        act[:, k] = np.convolve(np.pad(act[:, k], len(kernel), mode="edge"), kernel, mode="same")[len(kernel):-len(kernel)]
    return np.clip(act, 0, 1)


# ---------------------------------------------------------------- render

def render_clip(src: Path, info: dict, spec, transcript: dict, out_path: Path, workdir: Path,
                progress: Callable[[float], None]) -> None:
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python-headless belum terpasang di server.")

    W, H = info["width"], info["height"]
    dw = W if W <= 1920 else 1920
    dh = int(round(H * dw / W))
    dw -= dw % 2
    dh -= dh % 2

    start = max(0.0, float(spec.start))
    end = min(float(spec.end), info["duration"] or float(spec.end))
    dur = end - start
    if dur < 1.0:
        raise ValueError("Klip terlalu pendek untuk dirender (minimal 1 detik).")
    out_w, out_h = SIZES.get(spec.aspect, SIZES["9:16"])
    n = int(round(dur * FPS))
    motion = spec.motion if spec.motion in ("ken", "punch", "drift", "none") else "none"
    framing = getattr(spec, "framing", None) or ("follow" if spec.subjectTracking else "center")
    if framing == "split" and (out_h != 1920 or out_w != 1080):
        framing = "speaker"                      # layar terbagi hanya untuk 9:16

    words = words_in_range(transcript, start, end)
    seg_starts = [s["start"] - start for s in transcript["segments"] if start <= s["start"] < end]

    base = 0.0
    layout = None
    if framing in ("speaker", "split"):
        data = getattr(spec, "speakers", None)
        if not data:
            base = 0.2
            res = speakers_mod.analyze(src, info, start, end, transcript,
                                       progress=lambda st, p: progress(0.2 * (st + p) / 3.0))
            data = {"tracks": res["tracks"], "turns": res["turns"]}
        tracks = sorted(data["tracks"], key=lambda t: t["x"])
        if len(tracks) < 2:
            print("[clipforge] kurang dari dua pembicara terdeteksi; pakai bingkai 'ikuti wajah'.")
            framing = "follow"
        else:
            turns = data["turns"] or [{"start": start, "end": end, "track": tracks[0]["id"]}]
            fx, fy, who, shot_t = speakers_mod.frame_targets(
                turns, tracks, n, FPS, start, glide=(getattr(spec, "switchStyle", "cut") == "glide"))
            if framing == "split":
                z, _ = _zoom_curve(n, dur, motion, seg_starts)
                top2 = sorted(sorted(tracks, key=lambda t: -t.get("presence", 1))[:2], key=lambda t: t["x"])
                act = _activity(who, [t["id"] for t in top2], FPS)
                layout = SplitLayout(top2, z, act, dw, dh, out_w, out_h)
            else:
                z, drift = _zoom_curve(n, dur, motion, seg_starts, shot_t)
                layout = SingleLayout(fx, fy, z, drift, dw, dh, out_w, out_h, True)

    if layout is None:
        track = framing == "follow"
        ts, xs, ys = sample_faces(src, start, dur, dw, dh, cv2) if track else ([], [], [])
        fx, fy, z, drift = camera_path(n, dur, ts, xs, ys, track, motion, seg_starts)
        layout = SingleLayout(fx, fy, z, drift, dw, dh, out_w, out_h, track)
    progress(base + 0.03)

    workdir.mkdir(parents=True, exist_ok=True)
    style = (spec.captions or {}).get("style") if spec.captions else None
    ass_name = None
    if style in ("karaoke", "bounce", "block") and words:
        ass_name = "captions.ass"
        (workdir / ass_name).write_text(
            build_ass(words, out_w, out_h, style, center=isinstance(layout, SplitLayout)), encoding="utf-8")

    dec_cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an",
               "-vf", f"fps={FPS},scale={dw}:{dh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    enc_cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}",
               "-r", str(FPS), "-i", "pipe:0", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
               "-map", "0:v:0", "-map", "1:a:0?"]
    if ass_name:
        enc_cmd += ["-vf", f"ass={ass_name}"]
    enc_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out_path.resolve())]

    log_path = workdir / "ffmpeg.log"
    with open(log_path, "wb") as log:
        dec = subprocess.Popen(dec_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=log, cwd=str(workdir))
        fsz = dw * dh * 3
        last = None
        try:
            for i in range(n):
                buf = dec.stdout.read(fsz)
                if len(buf) == fsz:
                    last = np.frombuffer(buf, np.uint8).reshape(dh, dw, 3)
                if last is None:
                    raise RuntimeError("Gagal membaca bingkai dari video sumber.")
                enc.stdin.write(layout.compose(i, last, cv2).tobytes())
                if i % 15 == 0:
                    progress(base + 0.03 + (0.97 - base) * i / n)
            enc.stdin.close()
            rc = enc.wait()
        except BrokenPipeError:
            rc = enc.wait()
        finally:
            dec.kill()
            dec.wait()
            if enc.poll() is None:
                enc.kill()
                enc.wait()
    if rc != 0 or not out_path.exists():
        tail = log_path.read_text(errors="ignore").strip().splitlines()[-3:]
        raise RuntimeError("Render gagal: " + (" | ".join(tail) or "FFmpeg berhenti tanpa pesan"))
    progress(1.0)
