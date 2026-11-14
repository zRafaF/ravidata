"""Paths, takes, sync, radar loading and table I/O shared by every analysis step."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("RAVIDATA_DATA", ROOT / "data" / "ravidata"))
TAKES = ["pilot_chairs", "static_pass1", "static_pass2", "indoor_course", "outdoor_course"]
STATIC = ["pilot_chairs", "static_pass1", "static_pass2"]
CARRIED = ["indoor_course", "outdoor_course"]
ENV = {"pilot_chairs": "indoor", "static_pass1": "indoor", "static_pass2": "indoor", "indoor_course": "indoor",
       "outdoor_course": "outdoor"}
QUICK = os.environ.get("RAVIDATA_QUICK", "") not in ("", "0")
BIN_M = 0.04


def set_quick(q: bool):
    global QUICK
    QUICK = bool(q)
    os.environ["RAVIDATA_QUICK"] = "1" if q else ""      # inherited by worker processes (spawn on Windows)


def out_dir(*parts) -> Path:
    """derived/analysis-v1 (full runs) or derived/analysis-quick (quick tests: never mixed with full results)."""
    p = DATA / "derived" / ("analysis-quick" if QUICK else "analysis-v1")
    p = p.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def log(msg: str):
    """Timestamped line; printed above any progress bar."""
    line = time.strftime("%H:%M:%S ") + msg
    try:
        from tqdm import tqdm
        tqdm.write(line)
    except ImportError:
        print(line, flush=True)


def bar(iterable=None, total=None, desc="", unit="it", leave=True):
    """Progress bar with rate and time left (tqdm; a plain iterator without it)."""
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable if iterable is not None else _NoBar()
    return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, dynamic_ncols=True, mininterval=1.0,
                smoothing=0.05)


class _NoBar:
    def update(self, n=1):
        pass

    def set_postfix_str(self, s):
        pass

    def close(self):
        pass


def save_table(df, name: str, takes=None) -> Path:
    """Write a table. With `takes`, rows of other takes already in the table are kept (steps can run per take)."""
    import pandas as pd
    p = out_dir("tables") / f"{name}.parquet"
    if takes is not None and p.exists():
        old = pd.read_parquet(p)
        if "take" in old.columns:
            old = old[~old["take"].isin(list(takes))]
            df = pd.concat([old, df], ignore_index=True) if len(df) else old
    df.to_parquet(p, index=False)
    log(f"  -> {p.relative_to(DATA)} ({len(df):,} rows)")
    return p


def load_table(name: str, required: bool = True):
    import pandas as pd
    p = out_dir("tables") / f"{name}.parquet"
    if not p.exists():
        if required:
            raise SystemExit(f"{p} missing: run the step that makes it first (python -m analysis --help)")
        return None
    return pd.read_parquet(p)


def save_json(obj, name: str) -> Path:
    p = out_dir("tables") / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    log(f"  -> {p.relative_to(DATA)}")
    return p


def load_json(name: str, default=None):
    p = out_dir("tables") / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def manifest(step: str, args: dict, seconds: float):
    """One line per run in runs.jsonl: what ran, with which settings, on which commit (every run is kept)."""
    try:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = ""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "step": step, "quick": QUICK, "seconds": round(seconds, 1),
           "commit": commit, "python": sys.version.split()[0], "args": args}
    with open(out_dir() / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# ------------------------------------------------------------------------------------------------ takes
def take_json(take: str) -> dict:
    return json.loads((DATA / "takes" / take / "take.json").read_text(encoding="utf-8"))


def available_takes(names=None) -> list[str]:
    names = names or TAKES
    return [t for t in names if (DATA / "takes" / t / "take.json").exists()]


@dataclass
class Sync:
    a: float
    b: float
    offset: float
    imu_lag: float

    def video_to_host(self, t_video):
        return self.a * np.asarray(t_video) + self.b + self.offset

    def host_to_video(self, t_host):
        return (np.asarray(t_host) - self.b - self.offset) / self.a

    def phone_to_host(self, t_phone):
        return np.asarray(t_phone) + self.imu_lag + self.offset


def sync_of(take: str) -> Sync:
    s = take_json(take)["sync"]
    return Sync(float(s["video_a"]), float(s["video_b"]), float(s.get("recommended_video_offset_s", 0.0)),
                float(s.get("imu_lag_s", 0.0)))


@dataclass
class Recording:
    take: str
    file: Path
    part: str
    config: str
    t0: float              # first beep (host clock)
    t1: float
    lead_in: float
    idx: int = 0

    @property
    def key(self) -> str:
        return f"{self.take}/{self.file.stem}"

    @property
    def is2d(self) -> bool:
        return "2d" in self.config


def recordings(take: str, configs=None) -> list[Recording]:
    tj = take_json(take)
    out = []
    for i, r in enumerate(tj["recordings"]):
        f = DATA / "takes" / take / r["file"]
        if not f.exists() or (configs and r["config"] not in configs):
            continue
        out.append(Recording(take, f, r.get("part") or "free", r["config"], float(r["t_beep_start"]),
                             float(r["t_beep_end"]), float(r.get("lead_in_s") or 0.0), i))
    return out


@dataclass
class Radar:
    """One recording's cube and times. cube: (frames, D, 3 RX, range bins) complex64 (D = 1 for 1D)."""
    rec: Recording
    cube: np.ndarray
    t: np.ndarray
    frame_index: np.ndarray
    range_m: np.ndarray
    v_axis: np.ndarray | None
    config: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.t)


QUICK_FRAMES = 240


def default_max_frames():
    return QUICK_FRAMES if QUICK else None


def _frame_slice(rec: Recording, t: np.ndarray, max_frames, skip_lead_in=True) -> slice:
    sel = np.ones(len(t), bool)
    if skip_lead_in and rec.lead_in > 0:                          # the sync lead-in is not part of the scene
        sel &= t >= rec.t0 + rec.lead_in
    idx = np.flatnonzero(sel)
    if max_frames:
        idx = idx[:max_frames]
    if len(idx) == 0:
        idx = np.arange(min(len(t), max_frames or len(t)))
    return slice(int(idx[0]), int(idx[-1]) + 1)


def radar_times(rec: Recording, max_frames: int | None = None) -> np.ndarray:
    """Host times of the frames load_radar(rec, max_frames) returns (without reading the cube)."""
    import h5py
    with h5py.File(rec.file, "r") as h:
        t = h["t_host"][:]
    return t[_frame_slice(rec, t, max_frames)]


def load_radar(rec: Recording, max_frames: int | None = None, skip_lead_in: bool = True) -> Radar:
    import h5py
    with h5py.File(rec.file, "r") as h:
        t = h["t_host"][:]
        sl = _frame_slice(rec, t, max_frames, skip_lead_in)
        cube = h["cube"][sl]
        fi = h["frame_index"][sl]
        rng = h["range_m"][:]
        v = h["v_axis"][:] if "v_axis" in h else None
        cfg = json.loads(h.attrs.get("config_json", "{}"))
    return Radar(rec, cube, t[sl], fi, rng, v, cfg)


# ------------------------------------------------------------------------------------------------ calibration
def camera_K(size: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Display intrinsics of the upright 1440 x 1440 video (optionally scaled to `size`), distortion, size."""
    cal = json.loads((DATA / "calibration" / "camera.calib.json").read_text(encoding="utf-8"))["display"]
    W, H = cal["image_size"]
    K, D = np.array(cal["K"], float), np.array(cal["dist"], float)
    if size and size != W:
        K = K.copy(); K[:2] *= size / W
        W = size
    return K, D, W


def tag_sizes() -> dict[int, float]:
    p = ROOT / "docs" / "protocols" / "targets" / "tags.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): v["size_mm"] / 1000.0 for k, v in d["tags"].items()}


TAG_KIND = {0: "person", 1: "person", 10: "floor_tag", 12: "floor_tag", 17: "floor_tag", 11: "chair", 13: "chair",
            15: "chair", 20: "broom", 21: "broom", 50: "broom", 51: "broom"}
