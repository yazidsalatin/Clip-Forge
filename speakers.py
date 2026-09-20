"""Deteksi pembicara aktif untuk video dua orang (atau lebih).

Cara kerja
1. Bingkai kecil (10 fps) didekode dari rentang klip, lalu wajah dideteksi (Haar cascade frontal).
2. Deteksi dikelompokkan menjadi satu "jalur" per orang: pusat wajah yang berdekatan = orang yang sama.
3. Untuk tiap jalur, gerak di area mulut diukur (selisih antar bingkai), dikurangi gerak di area dahi
   supaya anggukan kepala tidak dihitung sebagai bicara.
4. Transkrip menentukan KAPAN ada ucapan; energi mulut menentukan SIAPA yang bicara pada jendela itu.
5. Histeresis: sahutan singkat ("iya", "hmm") dan gerak sesaat tidak memindahkan kamera.

Hasilnya berupa jalur (posisi tiap orang) dan giliran bicara (turns) dalam waktu absolut sumber,
sehingga tetap berlaku bila rentang klip digeser di editor.
"""
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SAMPLE_FPS = 10
SAMPLE_W = 800
DETECT_STRIDE = 2
MAX_TRACKS = 4

# Ambang keputusan (sudah diuji pada data sintetis; sesuaikan bila video nyata berperilaku lain)
MIN_ENERGY = 0.5        # energi mulut minimum agar sebuah jendela dianggap "terlihat bicara"
MIN_CONF = 0.12         # selisih relatif terbaik-vs-kedua minimum agar jendela dianggap terputuskan
SWITCH_CONF = 0.20      # keyakinan yang dibutuhkan untuk memindahkan pembicara
SWITCH_MIN_DUR = 0.9    # ucapan lebih pendek dari ini dianggap sahutan, tidak memindahkan kamera
MIN_SHOT = 1.2          # jeda minimum antar perpindahan (detik)
CUT_LEAD = 0.15         # potong sedikit sebelum pembicara baru mulai, seperti editor sungguhan
CHUNK = 2.0             # jendela ucapan panjang dipecah per ~2 detik agar bisa mendeteksi interupsi

Progress = Callable[[int, float], None]


def _cv2():
    import cv2
    return cv2


def _cascade(cv2):
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("OpenCV ini tidak punya detektor wajah. Pasang opencv-python-headless<5.")
    return cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))


def sample_size(src_w: int, src_h: int) -> Tuple[int, int]:
    sw = min(SAMPLE_W, src_w)
    sw -= sw % 2
    sh = int(round(src_h * sw / src_w))
    sh -= sh % 2
    return sw, sh


def iter_frames(src: Path, start: float, dur: float, sw: int, sh: int, fps: int = SAMPLE_FPS):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-an",
           "-vf", f"fps={fps},scale={sw}:{sh}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsz = sw * sh * 3
    try:
        while True:
            buf = proc.stdout.read(fsz)
            if len(buf) < fsz:
                break
            yield np.frombuffer(buf, np.uint8).reshape(sh, sw, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- 1. deteksi wajah

def detect_faces(frame_bgr, cascade, cv2, expected: int = 0) -> List[Tuple[float, float, float, float]]:
    """Deteksi di gambar mentah; bila menemukan lebih sedikit wajah dari yang diharapkan, coba juga versi
    yang diekualisasi dan gabungkan. Haar peka terhadap kontras, dan tiap varian kadang gagal di kondisi
    yang dilewati varian lain. Varian kedua hanya dijalankan bila perlu karena ia menggandakan biaya."""
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    msz = max(28, int(w * 0.035))
    found: List[Tuple[float, float, float, float]] = []

    def run(g):
        for (x, y, fw, fh) in cascade.detectMultiScale(g, 1.1, 6, minSize=(msz, msz)):
            cx, cy = x + fw / 2, y + fh / 2
            for i, (ox, oy, ow, oh) in enumerate(found):
                if abs(cx - ox) < 0.5 * max(fw, ow) and abs(cy - oy) < 0.5 * max(fh, oh):
                    if fw > ow:
                        found[i] = (cx, cy, float(fw), float(fh))
                    break
            else:
                found.append((cx, cy, float(fw), float(fh)))

    run(gray)
    if len(found) < expected:
        run(cv2.equalizeHist(gray))
    return found


# ---------------------------------------------------------------- 2. jalur per orang

def _components(points: np.ndarray, thr: float) -> np.ndarray:
    """Komponen terhubung pada grid sel berukuran thr/2 (tetangga 8 arah). Cepat dan tanpa scipy."""
    cell = max(thr / 2.0, 1.0)
    keys = np.floor(points / cell).astype(int)
    occupied = {}
    for idx, k in enumerate(map(tuple, keys)):
        occupied.setdefault(k, []).append(idx)
    label_of = {}
    n_lab = 0
    for k in occupied:
        if k in label_of:
            continue
        stack = [k]
        label_of[k] = n_lab
        while stack:
            cx, cy = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (cx + dx, cy + dy)
                    if nb in occupied and nb not in label_of:
                        label_of[nb] = n_lab
                        stack.append(nb)
        n_lab += 1
    out = np.empty(len(points), int)
    for k, idxs in occupied.items():
        out[idxs] = label_of[k]
    return out


def _smooth_1d(a: np.ndarray, k: int = 5) -> np.ndarray:
    if len(a) < k:
        return a
    pad = k // 2
    padded = np.pad(a, pad, mode="edge")
    med = np.array([np.median(padded[i:i + k]) for i in range(len(a))])
    padded = np.pad(med, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def build_tracks(dets: List[list], n: int, min_presence: float = 0.15) -> List[dict]:
    """dets[i] = daftar (cx,cy,w,h) pada bingkai i. Kembalikan jalur terurut dari kiri ke kanan."""
    pts = [(i, *f) for i, fs in enumerate(dets) for f in fs]
    if not pts:
        return []
    arr = np.array(pts, float)                                # kolom: frame, cx, cy, w, h
    thr = 0.75 * float(np.median(arr[:, 3]))
    labels = _components(arr[:, 1:3], thr)
    tracks = []
    for lab in np.unique(labels):
        sel = arr[labels == lab]
        frames = np.unique(sel[:, 0].astype(int))
        presence = len(frames) / max(n, 1)
        if presence < min_presence:
            continue
        pos = np.full((n, 4), np.nan)
        for row in sel:                                        # satu deteksi per bingkai: yang terbesar
            i = int(row[0])
            if i < n and (np.isnan(pos[i, 0]) or row[3] > pos[i, 2]):
                pos[i] = (row[1], row[2], row[3], row[4])
        idx = np.where(~np.isnan(pos[:, 0]))[0]
        for c in range(4):                                     # isi celah; ujung dipertahankan
            pos[:, c] = np.interp(np.arange(n), idx, pos[idx, c])
            pos[:, c] = _smooth_1d(pos[:, c])
        tracks.append({"pos": pos, "presence": presence, "n_det": len(frames),
                       "cx": float(np.median(pos[:, 0]))})
    # Deteksi palsu (pola pakaian, latar) muncul sesekali; wajah asli hampir selalu ada. Ambang relatif
    # terhadap jalur terkuat lebih tahan daripada angka mutlak. Kalibrasi: asli 0,78-1,0; palsu 0,04-0,17.
    if tracks:
        top = max(t["presence"] for t in tracks)
        tracks = [t for t in tracks if t["presence"] >= 0.4 * top]
    tracks.sort(key=lambda t: -t["presence"])
    tracks = tracks[:MAX_TRACKS]
    tracks.sort(key=lambda t: t["cx"])
    return tracks


# ---------------------------------------------------------------- 3. energi mulut

def _roi(cx, cy, w, h, sw, sh):
    x0, x1 = int(max(0, cx - w / 2)), int(min(sw, cx + w / 2))
    y0, y1 = int(max(0, cy - h / 2)), int(min(sh, cy + h / 2))
    return (x0, y0, x1, y1) if x1 - x0 >= 3 and y1 - y0 >= 3 else None


def mouth_energy(src: Path, start: float, dur: float, tracks: List[dict], sw: int, sh: int,
                 cv2, progress: Optional[Callable[[float], None]] = None) -> np.ndarray:
    n = tracks[0]["pos"].shape[0]
    energy = np.zeros((len(tracks), n))
    prev = None
    for i, frame in enumerate(iter_frames(src, start, dur, sw, sh)):
        if i >= n:
            break
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if prev is not None:
            for k, t in enumerate(tracks):
                cx, cy, w, h = t["pos"][i]
                mouth = _roi(cx, cy + 0.25 * h, 0.46 * w, 0.24 * h, sw, sh)
                ref = _roi(cx, cy - 0.22 * h, 0.60 * w, 0.22 * h, sw, sh)
                if mouth is None:
                    continue
                dm = float(cv2.absdiff(g[mouth[1]:mouth[3], mouth[0]:mouth[2]],
                                       prev[mouth[1]:mouth[3], mouth[0]:mouth[2]]).mean())
                dr = 0.0
                if ref is not None:
                    dr = float(cv2.absdiff(g[ref[1]:ref[3], ref[0]:ref[2]],
                                           prev[ref[1]:ref[3], ref[0]:ref[2]]).mean())
                energy[k, i] = max(0.0, dm - 0.7 * dr)
        prev = g
        if progress and i % 10 == 0:
            progress(min(0.99, i / max(n, 1)))
    return energy


# ---------------------------------------------------------------- 4. giliran bicara

def _speech_windows(words: List[dict], start: float, end: float) -> List[Tuple[float, float]]:
    ws = sorted((max(start, w["s"]), min(end, w["e"])) for w in words if w["e"] > start and w["s"] < end)
    wins: List[List[float]] = []
    for s, e in ws:
        if wins and s - wins[-1][1] < 0.35:
            wins[-1][1] = max(wins[-1][1], e)
        else:
            wins.append([s, e])
    out = []
    for s, e in wins:
        s, e = max(start, s - 0.1), min(end, e + 0.1)
        length = e - s
        pieces = max(1, int(round(length / CHUNK)))
        for p in range(pieces):
            out.append((s + length * p / pieces, s + length * (p + 1) / pieces))
    return out


def _change_point(e: np.ndarray, old: int, new: int, lo: float, hi: float, start: float, fps: int) -> float:
    """Titik pergantian terbaik dalam [lo, hi]: memaksimalkan (energi jalur baru - lama) sesudahnya dan
    meminimalkannya sebelumnya. Bila seri (jeda diam), pilih yang paling akhir supaya kamera pindah saat
    pembicara baru mulai bicara, bukan saat yang lama berhenti."""
    a = max(0, int((lo - start) * fps))
    b = min(e.shape[1], int((hi - start) * fps))
    if b - a < 3:
        return hi
    d = e[new, a:b] - e[old, a:b]
    c = np.concatenate([[0.0], np.cumsum(d)])
    score = c[-1] - 2.0 * c + 1e-3 * np.arange(len(c))
    return start + (a + int(np.argmax(score))) / fps


def decide_turns(energy: np.ndarray, start: float, end: float, words: List[dict],
                 presence: Optional[List[float]] = None, fps: int = SAMPLE_FPS) -> Tuple[List[dict], float]:
    """Fungsi murni: energi mulut (K x N) + waktu ucapan -> giliran bicara. Kembalikan (turns, keyakinan)."""
    k_tracks, n = energy.shape
    if k_tracks == 0:
        return [], 0.0
    base = np.percentile(energy, 20, axis=1, keepdims=True)
    e = np.maximum(0.0, energy - base)
    if n >= 3:
        e = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, mode="same"), 1, e)

    chunks = _speech_windows(words, start, end) if words else []
    if not chunks:                                        # tanpa transkrip: pakai seluruh rentang
        length = end - start
        pieces = max(1, int(round(length / CHUNK)))
        chunks = [(start + length * p / pieces, start + length * (p + 1) / pieces) for p in range(pieces)]

    events: List[Tuple[float, int]] = []                  # (waktu pergantian, jalur)
    cur: Optional[int] = None
    last_switch = -1e9
    confs = []
    for (t0, t1) in chunks:
        a = int((t0 - start) * fps)
        b = max(a + 1, int((t1 - start) * fps))
        m = e[:, a:min(b, n)].mean(axis=1) if a < n else np.zeros(k_tracks)
        order = np.argsort(-m)
        best = float(m[order[0]])
        second = float(m[order[1]]) if k_tracks > 1 else 0.0
        conf = (best - second) / (best + second + 1e-6) if k_tracks > 1 else 1.0
        cand = int(order[0]) if (best >= MIN_ENERGY and conf >= MIN_CONF) else None
        if cand is None:
            continue
        confs.append(conf)
        if cur is None:
            cur, last_switch = cand, t0
            events.append((start, cand))
        elif cand != cur and (t1 - t0) >= SWITCH_MIN_DUR and conf >= SWITCH_CONF and t0 - last_switch >= MIN_SHOT:
            tau = _change_point(e, cur, cand, max(last_switch + 0.5, t0 - CHUNK), t1, start, fps)
            events.append((tau, cand))
            cur, last_switch = cand, tau

    if not events:
        fallback = int(np.argmax(presence)) if presence else 0
        return [{"start": start, "end": end, "track": fallback, "conf": 0.0}], 0.0

    turns: List[dict] = []
    for i, (tau, tr) in enumerate(events):
        st = start if i == 0 else max(turns[-1]["start"] + 0.5, tau - CUT_LEAD)
        if i > 0:
            turns[-1]["end"] = st
        turns.append({"start": st, "end": end, "track": tr})
    overall = float(np.mean(confs)) if confs else 0.0
    for t in turns:
        t["conf"] = round(overall, 2)
    return turns, overall


# ---------------------------------------------------------------- 5. orkestrasi

def _face_thumb(src: Path, t: float, box: Tuple[float, float, float, float], out: Path, cv2) -> bool:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", str(src), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not r.stdout:
        return False
    img = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return False
    h, w = img.shape[:2]
    cx, cy, fw, fh = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    half = max(fw, fh) * 0.85
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    crop = cv2.resize(img[y0:y1, x0:x1], (96, 96), interpolation=cv2.INTER_AREA)
    return bool(cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 85]))


def analyze(src: Path, info: dict, start: float, end: float, transcript: Optional[dict],
            progress: Optional[Progress] = None, thumb_dir: Optional[Path] = None,
            thumb_prefix: str = "spk") -> dict:
    """Analisis rentang [start, end] (detik sumber). progress(tahap, 0..1) dengan tahap 0..2."""
    cv2 = _cv2()
    cascade = _cascade(cv2)
    prog = progress or (lambda s, p: None)
    sw, sh = sample_size(info["width"], info["height"])
    dur = end - start
    n_est = max(1, int(dur * SAMPLE_FPS))

    dets = []
    recent: List[int] = []
    current: list = []
    for i, frame in enumerate(iter_frames(src, start, dur, sw, sh)):
        if i % DETECT_STRIDE == 0:            # wajah tidak bergerak cepat: deteksi tiap DETECT_STRIDE bingkai
            expected = max(recent[-30:]) if recent else 2
            current = detect_faces(frame, cascade, cv2, expected)
            recent.append(len(current))
        dets.append(current)
        if i % 5 == 0:
            prog(0, min(0.99, i / n_est))
    n = len(dets)
    prog(0, 1.0)
    notes: List[str] = []
    tracks = build_tracks(dets, n)

    def norm(t, i):
        cx, cy, w, h = t["pos"][min(i, n - 1)]
        return cx / sw, cy / sh, w / sw, h / sh

    out_tracks = []
    for k, t in enumerate(tracks):
        x, y, w, h = (float(np.median(t["pos"][:, c])) / (sw if c in (0, 2) else sh) for c in range(4))
        out_tracks.append({"id": k, "label": chr(65 + k), "x": round(x, 4), "y": round(y, 4),
                           "w": round(w, 4), "h": round(h, 4), "presence": round(t["presence"], 2), "thumb": None})

    if not tracks:
        notes.append("Tidak ada wajah terdeteksi. Deteksi ini hanya mengenali wajah menghadap depan.")
        return {"start": start, "end": end, "tracks": [], "turns": [], "confidence": 0.0, "notes": notes}

    if thumb_dir is not None:
        for k, t in enumerate(tracks):
            areas = t["pos"][:, 2] * t["pos"][:, 3]
            i = int(np.argmax(areas))
            name = f"{thumb_prefix}_{k}.jpg"
            box = (t["pos"][i][0] / sw, t["pos"][i][1] / sh, t["pos"][i][2] / sw, t["pos"][i][3] / sh)
            if _face_thumb(src, start + i / SAMPLE_FPS, box, thumb_dir / name, cv2):
                out_tracks[k]["thumb"] = f"/api/thumbs/{name}"

    if len(tracks) == 1:
        notes.append("Hanya satu wajah terdeteksi, jadi tidak ada pergantian pembicara.")
        prog(1, 1.0)
        prog(2, 1.0)
        x, y = out_tracks[0]["x"], out_tracks[0]["y"]
        return {"start": start, "end": end, "tracks": out_tracks, "confidence": 1.0, "notes": notes,
                "turns": [{"start": start, "end": end, "track": 0, "conf": 1.0, "x": x, "y": y}]}

    energy = mouth_energy(src, start, dur, tracks, sw, sh, cv2, progress=lambda p: prog(1, p))
    prog(1, 1.0)
    prog(2, 0.0)
    words = []
    if transcript:
        words = [{"s": w["start"], "e": w["end"]} for s in transcript["segments"] for w in s["words"]]
    turns, conf = decide_turns(energy, start, end, words, [t["presence"] for t in tracks])
    for t in turns:
        a = max(0, int((t["start"] - start) * SAMPLE_FPS))
        b = max(a + 1, int((t["end"] - start) * SAMPLE_FPS))
        seg = tracks[t["track"]]["pos"][a:b]
        t["x"] = round(float(np.median(seg[:, 0])) / sw, 4)
        t["y"] = round(float(np.median(seg[:, 1])) / sh, 4)
        t["start"], t["end"] = round(t["start"], 2), round(t["end"], 2)
    if not words:
        notes.append("Transkrip belum ada, jadi seluruh rentang dianggap ucapan. Hasil kurang akurat.")
    if conf < 0.25:
        notes.append("Keyakinan rendah: gerak mulut sulit dibedakan. Periksa giliran bicara dan koreksi bila perlu.")
    prog(2, 1.0)
    return {"start": start, "end": end, "tracks": out_tracks, "turns": turns,
            "confidence": round(conf, 2), "notes": notes}


# ---------------------------------------------------------------- dipakai render

def sanitize(obj: Optional[dict]) -> Optional[dict]:
    """Validasi data pembicara dari klien sebelum dipakai render. Kembalikan None bila kosong."""
    if not obj:
        return None
    try:
        tracks = [{"id": int(t["id"]), "x": float(t["x"]), "y": float(t["y"]), "w": float(t.get("w", 0.1)),
                   "h": float(t.get("h", 0.15)), "presence": float(t.get("presence", 1.0))}
                  for t in obj.get("tracks", [])][:MAX_TRACKS]
        turns = []
        for t in obj.get("turns", [])[:500]:
            item = {"start": float(t["start"]), "end": float(t["end"]), "track": int(t["track"])}
            if t.get("x") is not None and t.get("y") is not None:
                item["x"], item["y"] = float(t["x"]), float(t["y"])
            turns.append(item)
    except (KeyError, TypeError, ValueError):
        raise ValueError("Data pembicara tidak valid.")
    ids = {t["id"] for t in tracks}
    if any(t["track"] not in ids for t in turns):
        raise ValueError("Giliran bicara merujuk ke pembicara yang tidak ada.")
    if any(not (0 <= v <= 1) for t in tracks for v in (t["x"], t["y"])):
        raise ValueError("Posisi pembicara harus berada di antara 0 dan 1.")
    turns.sort(key=lambda t: t["start"])
    return {"tracks": tracks, "turns": turns}


def frame_targets(turns: List[dict], tracks: List[dict], n: int, fps: int, start: float,
                  glide: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per bingkai keluaran: posisi kamera (fx, fy), indeks pembicara, dan waktu sejak awal shot."""
    by_id = {t["id"]: t for t in tracks}
    fx, fy = np.zeros(n), np.zeros(n)
    who = np.zeros(n, int)
    shot_t = np.zeros(n)
    starts = np.array([t["start"] for t in turns])
    tx = np.array([t.get("x", by_id[t["track"]]["x"]) for t in turns])
    ty = np.array([t.get("y", by_id[t["track"]]["y"]) for t in turns])
    for i in range(n):
        t = start + i / fps
        k = int(np.searchsorted(starts, t, side="right")) - 1
        k = max(0, k)
        x, y = tx[k], ty[k]
        if glide and k > 0 and t - starts[k] < 0.35:
            u = (t - starts[k]) / 0.35
            u = u * u * (3 - 2 * u)
            x = tx[k - 1] + (tx[k] - tx[k - 1]) * u
            y = ty[k - 1] + (ty[k] - ty[k - 1]) * u
        fx[i], fy[i] = x, y
        who[i] = turns[k]["track"]
        shot_t[i] = max(0.0, t - starts[k])
    return fx, fy, who, shot_t
