"""Align a phone recording (video + Sensor Logger IMU) with the radar takes, and pack everything into one file.

Clocks (all mapped to the laptop's clock, ``t_host`` = epoch seconds, the clock of every radar frame):

* **radar → host:** each radar frame already carries ``t_host`` (``cubes.h5``). The recorder beeps once when a
  recording starts and twice when it stops; ``runs/TCC_TAKES.csv`` has those beep times on the host clock.
* **video → host:** the phone's microphone hears the beeps. We find the 1.2 kHz tones in the video's sound track and
  match them with the takes' beep times: ``t_host = a * t_video + b`` (a = 1 unless enough beeps span a long time).
* **IMU → host:** Sensor Logger stamps samples with the phone's clock (epoch). The phone and the laptop clocks agree
  to a fraction of a second; the remaining offset is found by cross-correlating the gyro's rotation rate with the
  amount of motion in the video (twist the stick at the start): ``t_host = t_phone + lag``.

A manual correction (ms) can be added to both; the GUI (`ravidata align`) shows the result side by side.

The pack (``*.tccpack.h5``) holds one phone recording (video embedded, IMU) and every radar take inside it, each
radar frame with its time on the host clock, the nearest video frame and the rig attitude. See ``load_pack``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .logsetup import get_logger

log = get_logger("align")

FORMAT = "ravidata-tccpack-v1"
BEEP_HZ = 1200.0
SCAN_VERSION = 3
SMALL = (48, 86)               # (width, height) of the frames used for the motion signal


# =============================================================================================== IMU (Sensor Logger)
@dataclass
class Imu:
    path: Path
    meta: dict
    sensors: dict[str, dict[str, np.ndarray]]      # sensor -> {"t": phone epoch s, column -> values}

    @property
    def t_range(self) -> tuple[float, float]:
        g = self.sensors.get("Gyroscope") or next(iter(self.sensors.values()))
        return float(g["t"][0]), float(g["t"][-1])


def load_sensorlogger(path: str | Path) -> Imu:
    """Read a Sensor Logger export (zip of CSVs, or a folder of them). Times become epoch seconds (phone clock)."""
    path = Path(path)
    files: dict[str, bytes] = {}
    if path.is_dir():
        for p in path.glob("*.csv"):
            files[p.name] = p.read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if n.lower().endswith(".csv") and "/" not in n.strip("/"):
                    files[n] = z.read(n)
    meta = {}
    if "Metadata.csv" in files:
        rows = list(csv.DictReader(io.StringIO(files.pop("Metadata.csv").decode("utf-8", "replace"))))
        meta = rows[0] if rows else {}
    sensors = {}
    for name, raw in files.items():
        txt = raw.decode("utf-8", "replace")
        if txt.count("\n") < 2:
            continue
        try:
            a = np.genfromtxt(io.StringIO(txt), delimiter=",", names=True, dtype=float)
        except Exception:      # pragma: no cover - unexpected file
            continue
        if a.dtype.names is None or "time" not in a.dtype.names:
            continue
        d = {"t": np.asarray(a["time"], dtype=np.float64) / 1e9}
        for c in a.dtype.names:
            if c not in ("time", "seconds_elapsed"):
                d[c] = np.asarray(a[c], dtype=np.float64)
        sensors[name[:-4]] = d
    if "Gyroscope" not in sensors:
        raise ValueError(f"{path.name}: no Gyroscope.csv (turn on Gyroscope in Sensor Logger)")
    return Imu(path, meta, sensors)


def gyro_rate(imu: Imu) -> tuple[np.ndarray, np.ndarray]:
    g = imu.sensors["Gyroscope"]
    return g["t"], np.sqrt(g["x"] ** 2 + g["y"] ** 2 + g["z"] ** 2)


# =============================================================================================== video scan
@dataclass
class VideoScan:
    path: Path
    width: int
    height: int
    rotation: int                    # display-matrix rotation (degrees, PyAV/FFmpeg convention; 0 for most files)
    pts: np.ndarray                  # seconds, one per decoded frame
    motion: np.ndarray               # mean abs difference to the previous frame (small grey), first = 0
    audio_fs: int
    tones: list                      # [(t, duration, snr_db)] of 1.2 kHz tones
    duration: float

    @property
    def n_frames(self) -> int:
        return int(self.pts.size)


def detect_tones(x: np.ndarray, fs: int, f0: float = BEEP_HZ, min_snr_db: float = 15.0,
                 min_dur: float = 0.08) -> list[tuple[float, float, float]]:
    """Segments where the band around ``f0`` is well above the neighbouring bands."""
    from scipy.signal import stft
    if x.size < fs // 4:
        return []
    f, t, Z = stft(x.astype(np.float32), fs, nperseg=512, noverlap=384)
    P = np.abs(Z) ** 2
    band = (f > f0 - 50) & (f < f0 + 50)
    ref = ((f > f0 - 600) & (f < f0 - 200)) | ((f > f0 + 300) & (f < f0 + 1300))
    r = 10 * np.log10(P[band].mean(0) + 1e-20) - 10 * np.log10(P[ref].mean(0) + 1e-20)
    on = r > min_snr_db
    out, i = [], 0
    while i < on.size:
        if on[i]:
            j = i
            while j < on.size and on[j]:
                j += 1
            if t[j - 1] - t[i] >= min_dur:                # the recorder's beeps last 180 ms
                out.append((float(t[i]), float(t[j - 1] - t[i]), float(r[i:j].max())))
            i = j
        else:
            i += 1
    return out


def group_beeps(tones: list, gap: float = 0.6) -> list[tuple[float, str]]:
    """[(t_first_tone, "single"|"double"|"multi")]: tones closer than ``gap`` s belong to one signal."""
    out = []
    i = 0
    while i < len(tones):
        j = i
        while j + 1 < len(tones) and tones[j + 1][0] - tones[j][0] < gap:
            j += 1
        n = j - i + 1
        out.append((tones[i][0], "single" if n == 1 else "double" if n == 2 else "multi"))
        i = j + 1
    return out


def scan_video(path: str | Path, progress: Callable[[float], None] | None = None, use_cache: bool = True) -> VideoScan:
    """One pass over the video: frame times, a motion signal and the beeps in the sound. Cached next to the video."""
    import av
    path = Path(path)
    st = path.stat()
    cache = path.with_name(path.name + ".scan.npz")
    if use_cache and cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            info = json.loads(str(z["info"]))
            if info.get("size") == st.st_size and info.get("version") == SCAN_VERSION:
                return VideoScan(path, info["width"], info["height"], info["rotation"], z["pts"], z["motion"],
                                 info["audio_fs"], [tuple(x) for x in info["tones"]], info["duration"])
        except Exception:  # pragma: no cover - stale cache
            pass
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        dur = float(c.duration / 1e6) if c.duration else 0.0
        rot = 0
        width, height = vs.codec_context.width, vs.codec_context.height
        pts, motion = [], []
        prev = None
        for fr in c.decode(video=0):
            t = fr.time if fr.time is not None else (pts[-1] + 1 / 30 if pts else 0.0)
            if not pts:
                rot = int(round(getattr(fr, "rotation", 0) or 0))     # display matrix (phone videos: often -90)
            g = fr.reformat(width=SMALL[0], height=SMALL[1], format="gray").to_ndarray().astype(np.float32)
            motion.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
            pts.append(float(t))
            if progress and dur and len(pts) % 30 == 0:
                progress(min(0.9, 0.9 * t / dur))
    fs = 16000
    audio = []
    try:
        with av.open(str(path)) as c:
            if c.streams.audio:
                rs = av.AudioResampler(format="flt", layout="mono", rate=fs)
                for fr in c.decode(audio=0):
                    for o in rs.resample(fr):
                        audio.append(o.to_ndarray().reshape(-1))
                for o in rs.resample(None):
                    audio.append(o.to_ndarray().reshape(-1))
    except Exception as e:  # pragma: no cover
        log.warning("audio decode failed: %s", e)
    x = np.concatenate(audio) if audio else np.zeros(0, np.float32)
    tones = detect_tones(x, fs)
    order = np.argsort(pts)
    scan = VideoScan(path, width, height, rot, np.asarray(pts)[order], np.asarray(motion, np.float32)[order], fs,
                     tones, dur or (pts[-1] if pts else 0.0))
    if use_cache:
        try:
            info = {"size": st.st_size, "version": SCAN_VERSION, "width": width, "height": height, "rotation": rot,
                    "audio_fs": fs, "tones": tones, "duration": scan.duration}
            np.savez(cache, pts=scan.pts, motion=scan.motion, info=json.dumps(info))
        except Exception:  # pragma: no cover - read-only folder
            pass
    if progress:
        progress(1.0)
    return scan


# =============================================================================================== radar takes
@dataclass
class Take:
    run_id: str
    run_dir: Path | None
    config: str
    t_beep_start: float              # host clock (epoch s)
    t_beep_end: float
    row: dict = field(default_factory=dict)
    cubes_name: str = "cubes.h5"     # the 2nd, 3rd ... recording of one run is cubes_02.h5, cubes_03.h5 (session.py)

    @property
    def has_cubes(self) -> bool:
        return self.run_dir is not None and (self.run_dir / self.cubes_name).exists()


def _cubes_for(run_dir: Path, t0: float, t1: float) -> str:
    """Which cubes*.h5 of a run holds the take recorded between the beeps t0..t1 (host clock): the file whose
    frames overlap that window most. One file (the usual case) -> cubes.h5 without opening anything."""
    files = sorted(run_dir.glob("cubes*.h5"))
    if len(files) <= 1:
        return "cubes.h5"
    import h5py
    best, best_ov = "cubes.h5", -1.0
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                t = f["t_host"]
                if t.shape[0] == 0:
                    continue
                a, b = float(t[0]), float(t[-1])
        except (OSError, KeyError):
            continue
        ov = min(b, t1) - max(a, t0)
        if ov > best_ov:
            best, best_ov = p.name, ov
    return best


def find_takes(runs_dir: str | Path) -> list[Take]:
    """Every recorded segment: rows of ``TCC_TAKES.csv`` (run dirs resolved by name under ``runs_dir``)."""
    runs_dir = Path(runs_dir)
    takes: list[Take] = []
    seen = set()
    p = runs_dir / "TCC_TAKES.csv"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    t0, t1 = float(row["t_beep_start"]), float(row["t_beep_end"])
                except (KeyError, ValueError):
                    continue
                name = Path(str(row.get("run_dir", "")).replace("\\", "/")).name
                rd = runs_dir / name if name and (runs_dir / name).exists() else None
                key = (name, t0)
                if key in seen:
                    continue
                seen.add(key)
                cubes = _cubes_for(rd, t0, t1) if rd is not None else "cubes.h5"
                rid = name or f"take_{t0:.0f}"
                if cubes != "cubes.h5":                    # unique id per recording: <run>_02 for cubes_02.h5
                    rid = f"{rid}_{Path(cubes).stem.split('_')[-1]}"
                takes.append(Take(rid, rd, row.get("config", ""), t0, t1, dict(row), cubes))
    takes.sort(key=lambda t: t.t_beep_start)
    return takes


def load_take_radar(take: Take) -> dict | None:
    """The radar frames of a take (None if its cubes.h5 is not on this computer)."""
    if not take.has_cubes:
        return None
    import h5py

    from .config import RadarConfig
    from .cube import CubeShape, decode
    with h5py.File(take.run_dir / take.cubes_name, "r") as f:
        cfg = RadarConfig.from_dict(json.loads(f.attrs["config_json"]))
        words = f["words"][:]
        shapes = f["shape"][:]
        t_host = f["t_host"][:]
        fi = f["frame_index"][:]
    n = len(t_host)
    if n == 0:
        return None
    s0 = shapes[0]
    same = (shapes == s0).all(axis=1)
    cube = np.zeros((n, int(s0[0]), int(s0[1]), int(s0[2])), np.complex64)
    for i in range(n):
        if same[i]:
            cube[i] = decode(words[i, : int(np.prod(shapes[i]))], CubeShape(*map(int, shapes[i])), cfg)
    d = cfg.derive()
    out = {"cfg": cfg, "words": words, "shape": shapes, "t_host": t_host, "frame_index": fi, "cube": cube,
           "range_m": np.arange(cube.shape[-1]) * d.bin_mm / 1000.0, "v_axis": None}
    nd = cube.shape[1]
    if nd > 1:
        vres = (cfg.veloc_res_mm_s or 0) / 1000.0 or (2 * (cfg.veloc_max_mm_s or 1000) / 1000.0 / nd)
        out["v_axis"] = (np.arange(nd) - nd // 2) * vres
    return out


# =============================================================================================== alignment
@dataclass
class Alignment:
    video_a: float = 1.0                 # t_host = video_a * t_video + video_b (+ manual_video_ms)
    video_b: float = 0.0
    video_ok: bool = False
    video_matches: list = field(default_factory=list)   # [{run_id, beep, t_video, t_host, resid_ms}]
    imu_lag: float = 0.0                 # t_host = t_phone + imu_lag (+ manual_imu_ms)
    imu_corr: float = 0.0
    imu_ok: bool = False
    imu_method: str = "clock"            # "gyro-video correlation" | "clock" (phone vs laptop clock only)
    radar_check: dict = field(default_factory=dict)   # radar motion vs video motion: {run_id: {lag_ms, corr}}
    manual_video_ms: float = 0.0
    manual_imu_ms: float = 0.0
    notes: list = field(default_factory=list)

    def video_to_host(self, t):
        return self.video_a * np.asarray(t, dtype=np.float64) + self.video_b + self.manual_video_ms / 1000.0

    def host_to_video(self, t):
        return (np.asarray(t, dtype=np.float64) - self.video_b - self.manual_video_ms / 1000.0) / self.video_a

    def imu_to_host(self, t):
        return np.asarray(t, dtype=np.float64) + self.imu_lag + self.manual_imu_ms / 1000.0

    def host_to_imu(self, t):
        return np.asarray(t, dtype=np.float64) - self.imu_lag - self.manual_imu_ms / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)


def match_beeps(beeps: list[tuple[float, str]], takes: list[Take], tol: float = 0.08,
                hint_b: float | None = None, hint_window: float = 30.0) -> tuple[float, float, list[dict]]:
    """Find t_host = a * t_video + b from the beeps heard in the video and the takes' beep times.

    Every (video beep, radar beep) pair of the same kind proposes an offset b; the offset most pairs agree on wins
    (works for one take or for a whole block of steps with identical lengths). ``hint_b`` (e.g. from the IMU's
    phone clock) limits the search to +-``hint_window`` s."""
    radar = [(t.t_beep_start, "single", t.run_id, "start") for t in takes] + \
            [(t.t_beep_end, "double", t.run_id, "end") for t in takes]
    cands = []
    beeps = [(tv, "double" if kv == "multi" else kv) for tv, kv in beeps]
    for tv, kv in beeps:
        for th, kh, _, _ in radar:
            if kv == kh:
                b = th - tv
                if hint_b is None or abs(b - hint_b) <= hint_window:
                    cands.append(b)
    if not cands:
        return 1.0, 0.0, []
    cands = np.sort(np.asarray(cands))
    best_b, best_n = cands[0], 0
    for b in cands:
        n = int(np.sum(np.abs(cands - b) <= tol))
        if n > best_n:
            best_b, best_n = b, n
    best_b = float(np.median(cands[np.abs(cands - best_b) <= tol]))
    pairs = []
    for tv, kv in beeps:
        for th, kh, rid, which in radar:
            if kv == kh and abs((th - tv) - best_b) <= tol:
                pairs.append((tv, th, rid, which))
    a, b = 1.0, best_b
    if len(pairs) >= 3:
        tv = np.array([p[0] for p in pairs])
        th = np.array([p[1] for p in pairs])
        if tv.max() - tv.min() > 20:
            A = np.vstack([tv, np.ones_like(tv)]).T
            a_fit, b_fit = np.linalg.lstsq(A, th, rcond=None)[0]
            if abs(a_fit - 1) < 0.002:                       # clock drift of at most 2 ms/s
                a, b = float(a_fit), float(b_fit)
    matches = [{"run_id": rid, "beep": which, "t_video": round(tv, 4), "t_host": round(th, 4),
                "resid_ms": round(1000 * (th - (a * tv + b)), 1)} for tv, th, rid, which in pairs]
    return a, b, matches


def imu_video_lag(imu: Imu, scan: VideoScan, video_a: float, video_b: float,
                  search: tuple[float, float] = (-3.0, 3.0), step: float = 0.01) -> tuple[float, float, float]:
    """lag with t_host = t_phone + lag, from the correlation of |gyro| with the video motion. Returns
    (lag, corr, best correlation more than 0.5 s away from it)."""
    tg, wg = gyro_rate(imu)
    tv = video_a * scan.pts[1:] + video_b
    mv = scan.motion[1:]
    if tv.size < 30:
        return 0.0, 0.0, 0.0
    lo, hi = max(tv[0], tg[0] + search[1]), min(tv[-1], tg[-1] + search[0])
    if hi - lo < 5:
        return 0.0, 0.0, 0.0
    grid = np.arange(lo, hi, step)
    v = np.interp(grid, tv, mv)
    v = (v - v.mean()) / (v.std() + 1e-9)
    res = []
    for lag in np.arange(search[0], search[1] + step / 2, step):
        g = np.interp(grid, tg + lag, wg)
        g = (g - g.mean()) / (g.std() + 1e-9)
        res.append((lag, float((g * v).mean())))
    res = np.array(res)
    k = int(np.argmax(res[:, 1]))
    far = np.abs(res[:, 0] - res[k, 0]) > 0.5
    second = float(res[far, 1].max()) if far.any() else 0.0
    return float(res[k, 0]), float(res[k, 1]), second


def radar_motion(radar: dict, r_min: float = 0.25, r_max: float = 4.5) -> tuple[np.ndarray, np.ndarray]:
    """How much the radar picture changes frame to frame (relative), at the frames' host times. When the rig is
    twisted, the static scene moves for the radar too, so this peaks with the gyro and the video motion."""
    C = radar["cube"]
    m = (radar["range_m"] > r_min) & (radar["range_m"] < r_max)
    X = C.sum(axis=1)[:, :, m]                                    # coherent over Doppler/intervals
    num = np.abs(np.diff(X, axis=0)).sum(axis=(1, 2))
    den = np.abs(X[1:]).sum(axis=(1, 2)) + 1e-12
    return radar["t_host"][1:], num / den


def radar_video_check(al: Alignment, scan: VideoScan, radar: dict, search: float = 1.0) -> tuple[float, float]:
    """Independent check of the video alignment: the lag (s) that best lines the radar's motion signal up with the
    video motion on the host clock, and its correlation. A positive lag means the radar sees a movement later
    than the aligned video does (e.g. the laptop's sound output latency delays the beeps)."""
    tr, mr = radar_motion(radar)
    tv, mv = al.video_to_host(scan.pts[1:]), scan.motion[1:].astype(float)
    lo, hi = max(tr[0], tv[0]) + search, min(tr[-1], tv[-1]) - search
    if hi - lo < 5 or tr.size < 30:
        return 0.0, 0.0
    grid = np.arange(lo, hi, 0.01)
    a = np.interp(grid, tr, mr)
    a = (a - a.mean()) / (a.std() + 1e-12)
    best = (0.0, -1.0)
    for lag in np.arange(-search, search, 0.005):
        b = np.interp(grid, tv + lag, mv)
        b = (b - b.mean()) / (b.std() + 1e-12)
        c = float((a * b).mean())
        if c > best[1]:
            best = (float(lag), c)
    return best


def check_with_radar(al: Alignment, scan: VideoScan | None, radars: dict[str, dict | None]) -> None:
    """Fill ``al.radar_check`` and add a note; does not change the alignment (the correlation is often weak)."""
    if scan is None:
        return
    for rid, r in radars.items():
        if r is None:
            continue
        lag, c = radar_video_check(al, scan, r)
        al.radar_check[rid] = {"lag_ms": round(1000 * lag), "corr": round(c, 2)}
        if c >= 0.25:
            al.notes.append(f"radar check {rid}: the radar sees the motion {1000 * lag:+.0f} ms after the video "
                            f"(corr {c:.2f}); try video offset {1000 * lag:+.0f} ms")
        else:
            al.notes.append(f"radar check {rid}: inconclusive (corr {c:.2f}); a sharp tap on the rig helps")


def auto_align(scan: VideoScan | None, imu: Imu | None, takes: list[Take]) -> Alignment:
    al = Alignment()
    if scan is not None:
        beeps = group_beeps(scan.tones)
        a, b, m = match_beeps(beeps, takes)
        al.video_a, al.video_b, al.video_matches = a, b, m
        al.video_ok = len(m) >= 2
        if not m:
            al.notes.append("no beep of the takes was found in the video sound: align the video by hand")
        elif len(m) == 1:
            al.notes.append("only one beep matched: the offset is set, the drift is not checked")
    if imu is not None:
        if scan is not None and al.video_ok:
            lag, c, c2 = imu_video_lag(imu, scan, al.video_a, al.video_b)
            al.imu_lag, al.imu_corr = lag, c
            al.imu_ok = c > 0.3 and c - c2 > 0.1
            al.imu_method = "gyro-video correlation" if al.imu_ok else "clock"
            if not al.imu_ok:
                al.imu_lag = 0.0
                al.notes.append(f"gyro vs video motion inconclusive (corr {c:.2f}, next {c2:.2f}): using the phone "
                                "clock as is; twist the stick at the start of a recording")
        else:
            al.notes.append("IMU on the phone clock only (no video alignment to correlate with)")
    return al


# =============================================================================================== rig attitude
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """q (..., 4) as (w, x, y, z) -> rotation matrices (..., 3, 3), device -> world (world z up)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rig_axes(imu: Imu) -> dict:
    """The rig's up / right / back axes in the phone's frame, from the mean gravity (Android: +g = up).
    back = +z (out of the screen, towards the operator); the camera and the radar look along -z."""
    g = imu.sensors.get("Gravity")
    if g is None:
        up = np.array([0.0, 1.0, 0.0])
    else:
        m = np.array([np.mean(g["x"]), np.mean(g["y"]), np.mean(g["z"])])
        k = int(np.argmax(np.abs(m[:2])))
        up = np.zeros(3)
        up[k] = np.sign(m[k]) or 1.0
    back = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, back)
    return {"up": up, "back": back, "right": right, "forward": -back,
            "phone_orientation": "portrait" if abs(up[1]) > 0 else "landscape"}


def attitude(imu: Imu, al: Alignment, t_host: np.ndarray) -> dict:
    """Rig attitude at the given host times: quaternion (w, x, y, z; device -> world, z up), boresight and up
    vectors in the world, pitch (boresight above the horizon), roll (about the boresight) and yaw (deg)."""
    o = imu.sensors.get("Orientation")
    t_host = np.asarray(t_host, dtype=np.float64)
    if o is None or "qw" not in o:
        nan = np.full(t_host.shape, np.nan)
        return {"quat": np.full(t_host.shape + (4,), np.nan), "pitch_deg": nan, "roll_deg": nan, "yaw_deg": nan,
                "forward": np.full(t_host.shape + (3,), np.nan), "up": np.full(t_host.shape + (3,), np.nan)}
    q = np.stack([o["qw"], o["qx"], o["qy"], o["qz"]], axis=1)
    for i in range(1, len(q)):                              # sign continuity for interpolation
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    tq = al.imu_to_host(o["t"])
    qi = np.stack([np.interp(t_host, tq, q[:, k]) for k in range(4)], axis=-1)
    qi /= np.linalg.norm(qi, axis=-1, keepdims=True) + 1e-12
    R = _quat_to_R(qi)
    ax = rig_axes(imu)
    fwd = R @ ax["forward"]
    up = R @ ax["up"]
    right = R @ ax["right"]
    pitch = np.degrees(np.arcsin(np.clip(fwd[..., 2], -1, 1)))
    roll = np.degrees(np.arctan2(-right[..., 2], up[..., 2]))
    yaw = np.degrees(np.arctan2(fwd[..., 0], fwd[..., 1]))     # 0 = north, + = east (world x east, y north)
    return {"quat": qi, "pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw, "forward": fwd, "up": up}


# =============================================================================================== pack
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def takes_in(takes: list[Take], al: Alignment, scan: VideoScan | None, imu: Imu | None) -> list[Take]:
    """The takes that fall inside the phone recording (video span, else IMU span)."""
    if scan is not None and al.video_ok:
        lo, hi = float(al.video_to_host(scan.pts[0])), float(al.video_to_host(scan.pts[-1]))
    elif imu is not None:
        lo, hi = (float(x) for x in al.imu_to_host(np.array(imu.t_range)))
    else:
        return []
    return [t for t in takes if t.t_beep_end >= lo and t.t_beep_start <= hi]


def write_pack(out: str | Path, scan: VideoScan | None, imu: Imu | None, al: Alignment, takes: list[Take],
               session_meta: dict | None = None, embed_video: bool = True,
               progress: Callable[[float], None] | None = None) -> Path:
    """One HDF5 file: video (+ frame times on the host clock), IMU (+ host times), every take's radar frames with
    their host time, nearest video frame and rig attitude, and the alignment. See ``load_pack``."""
    import h5py

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = FORMAT
        f.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        f.attrs["alignment_json"] = json.dumps(al.to_dict())
        f.attrs["session_meta_json"] = json.dumps(session_meta or {})
        f.attrs["clock"] = "t_host = laptop clock, epoch seconds (the radar frames' clock)"
        if scan is not None:
            g = f.create_group("video")
            g.attrs.update({"name": scan.path.name, "sha256": _sha256(scan.path), "width": scan.width,
                            "height": scan.height, "rotation": scan.rotation, "embedded": bool(embed_video),
                            "duration": scan.duration})
            g.create_dataset("t_video", data=scan.pts)
            g.create_dataset("t_host", data=al.video_to_host(scan.pts))
            g.create_dataset("motion", data=scan.motion)
            if embed_video:
                g.create_dataset("mp4", data=np.frombuffer(scan.path.read_bytes(), dtype=np.uint8))
            tones = np.array(scan.tones, dtype=np.float64).reshape(-1, 3)
            b = g.create_dataset("beep_tones", data=tones)
            b.attrs["columns"] = "t_video, duration_s, snr_db"
        if progress:
            progress(0.3)
        if imu is not None:
            gi = f.create_group("imu")
            gi.attrs.update({"name": imu.path.name, "meta_json": json.dumps(imu.meta),
                             "frame": "Android device frame: x right, y up (screen), z out of the screen; "
                                      "Orientation q = device -> world (x east, y north, z up)"})
            ax = rig_axes(imu)
            gi.attrs["rig_axes_json"] = json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ax.items()})
            for name, d in imu.sensors.items():
                s = gi.create_group(name)
                s.create_dataset("t_phone", data=d["t"])
                s.create_dataset("t_host", data=al.imu_to_host(d["t"]))
                for c, v in d.items():
                    if c != "t":
                        s.create_dataset(c, data=v)
        if progress:
            progress(0.5)
        gt = f.create_group("takes")
        for i, tk in enumerate(takes):
            g = gt.create_group(tk.run_id)
            g.attrs["take_json"] = json.dumps({"run_id": tk.run_id, "config": tk.config, "t_beep_start": tk.t_beep_start,
                                               "t_beep_end": tk.t_beep_end, "cubes": tk.cubes_name, "row": tk.row})
            r = load_take_radar(tk)
            g.attrs["has_radar"] = r is not None
            if r is None:
                continue
            g.attrs["config_json"] = json.dumps(r["cfg"].to_dict())
            for k in ("t_host", "frame_index", "words", "shape", "range_m"):
                g.create_dataset(k, data=r[k], compression="gzip" if k == "words" else None)
            g.create_dataset("cube", data=r["cube"], compression="gzip", compression_opts=1)
            if r["v_axis"] is not None:
                g.create_dataset("v_axis", data=r["v_axis"])
            if scan is not None:
                tv = al.host_to_video(r["t_host"])
                g.create_dataset("video_frame", data=np.clip(np.searchsorted(scan.pts, tv), 0, scan.n_frames - 1))
            if imu is not None:
                att = attitude(imu, al, r["t_host"])
                ga = g.create_group("attitude")
                for k, v in att.items():
                    ga.create_dataset(k, data=v)
            if progress:
                progress(0.5 + 0.5 * (i + 1) / max(1, len(takes)))
    tmp.replace(out)
    return out


def load_pack(path: str | Path) -> dict:
    """Everything in a pack as plain numpy arrays and dicts:

    ``{"alignment", "session_meta", "video": {...}, "imu": {sensor: {...}}, "takes": {run_id: {...}}}``.
    Each take has ``t_host``, ``cube`` (frames, Doppler/intervals, RX, range), ``range_m``, ``config`` and, when
    available, ``video_frame`` (index into ``video["t_video"]``) and ``attitude`` (pitch/roll/yaw per frame).
    The video file itself is ``video["mp4"]`` (bytes) when it was embedded."""
    import h5py
    out: dict = {}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != FORMAT:
            raise ValueError(f"{path}: not an {FORMAT} file")
        out["alignment"] = json.loads(f.attrs["alignment_json"])
        out["session_meta"] = json.loads(f.attrs["session_meta_json"])
        if "video" in f:
            g = f["video"]
            out["video"] = {k: g[k][()] for k in g if k != "mp4"} | dict(g.attrs)
            if "mp4" in g:
                out["video"]["mp4"] = g["mp4"][()].tobytes()
        if "imu" in f:
            out["imu"] = {n: {k: s[k][()] for k in s} for n, s in f["imu"].items()}
        out["takes"] = {}
        for rid, g in f["takes"].items():
            d = {k: g[k][()] for k in g if k != "attitude"}
            d["take"] = json.loads(g.attrs["take_json"])
            if "config_json" in g.attrs:
                d["config"] = json.loads(g.attrs["config_json"])
            if "attitude" in g:
                d["attitude"] = {k: g["attitude"][k][()] for k in g["attitude"]}
            out["takes"][rid] = d
    return out


def session_meta_for(runs_dir: Path, takes: list[Take]) -> dict:
    """The TCC_SESSION measurements of the first take's session (if any)."""
    for t in takes:
        sid = t.row.get("session")
        if sid and (runs_dir / f"TCC_SESSION_{sid}.json").exists():
            return json.loads((runs_dir / f"TCC_SESSION_{sid}.json").read_text(encoding="utf-8"))
    return {}


def find_phone_files(folder: str | Path) -> tuple[Path | None, Path | None]:
    """The video and the Sensor Logger zip in a folder (newest of each)."""
    folder = Path(folder)
    vids = sorted([p for p in folder.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".3gp", ".mkv")],
                  key=lambda p: p.stat().st_mtime)
    zips = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".zip"], key=lambda p: p.stat().st_mtime)
    return (vids[-1] if vids else None), (zips[-1] if zips else None)


def headless_pack(video: Path | None, imu_zip: Path | None, runs_dir: Path, out: Path | None = None,
                  embed_video: bool = True, manual_video_ms: float = 0.0,
                  manual_imu_ms: float = 0.0) -> tuple[Path, Alignment, list[Take]]:
    """`ravidata pack`: scan, auto-align, (manual offsets, as in the viewer), write.
    Returns (pack path, alignment, takes packed)."""
    scan = scan_video(video) if video else None
    imu = load_sensorlogger(imu_zip) if imu_zip else None
    all_takes = find_takes(runs_dir)
    al = auto_align(scan, imu, all_takes)
    al.manual_video_ms, al.manual_imu_ms = float(manual_video_ms), float(manual_imu_ms)
    takes = takes_in(all_takes, al, scan, imu)
    check_with_radar(al, scan, {tk.run_id: load_take_radar(tk) for tk in takes})
    if out is None:
        stem = (takes[0].run_id if takes else (video or imu_zip).stem).replace(" ", "_")
        out = runs_dir / "packs" / f"{stem}.tccpack.h5"
    return write_pack(out, scan, imu, al, takes, session_meta_for(runs_dir, takes), embed_video), al, takes


def fmt_alignment(al: Alignment) -> str:
    """Short human summary."""
    lines = []
    if al.video_matches:
        res = [abs(m["resid_ms"]) for m in al.video_matches]
        lines.append(f"video: {len(al.video_matches)} beeps matched, max residual {max(res):.0f} ms, "
                     f"scale {al.video_a:.5f}")
    else:
        lines.append("video: no beep matched")
    lines.append(f"IMU: lag {al.imu_lag * 1000:+.0f} ms ({al.imu_method}, corr {al.imu_corr:.2f})")
    if al.manual_video_ms or al.manual_imu_ms:
        lines.append(f"manual: video {al.manual_video_ms:+.0f} ms, IMU {al.manual_imu_ms:+.0f} ms")
    lines += al.notes
    return "\n".join(lines)
