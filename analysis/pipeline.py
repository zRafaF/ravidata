"""Detection pipelines ("families") = preprocessing + detector + optional temporal filter / clustering / grid.

Every family has hand-set DEFAULTS (literature-style starting values, never tuned on this data) and a search
SPACE for the optimiser. run(rec, family, params) -> (detections, cost) where detections is a dict of arrays
(f, r, az, el, score) and cost the processing time per frame (ms, this Python implementation: a relative proxy,
not an embedded number)."""

from __future__ import annotations

import time

import numpy as np

from . import common as C
from . import detectors as D
from . import geometry as G
from .radar import features_cached as features

PRE_KEYS = ("bg", "bg_alpha", "doppler", "zero_band", "combine")
COMMON = dict(bg="none", bg_alpha=0.98, doppler="sum", zero_band=0, combine="mean", r_min=0.30, r_max=4.10,
              m=1, n=1, spread=1)

DEFAULTS = {
    "thr_nearest": dict(COMMON, k_db=12.0),
    "thr": dict(COMMON, k_db=12.0),
    "ca_cfar": dict(COMMON, kind="ca", guard=2, train=8, k_db=10.0),
    "os_cfar": dict(COMMON, guard=2, train=8, os_q=0.75, k_db=10.0),
    "rd_cfar": dict(COMMON, guard_d=1, guard_r=1, train_d=2, train_r=4, k_db=12.0),
    "cfar_dbscan": dict(COMMON, kind="ca", guard=2, train=8, k_db=10.0, window=5, eps=0.25, min_samples=3),
    "grid": dict(COMMON, front="ca", guard=2, train=8, k_db=8.0, pose="none", speed=1.0, **{
        "g_" + k: v for k, v in {"l_occ": 0.85, "l_free": 0.4, "decay": 0.9, "sigma_r": 0.06, "sigma_az": 8.0,
                                 "blur": 0.0, "morph": "none", "morph_k": 3, "thr_hi": 1.5, "thr_lo": 0.8,
                                 "min_cells": 3, "free": True}.items()}),
}
DEFAULTS["grid_imu"] = dict(DEFAULTS["grid"], pose="imu_speed")
DEFAULTS["grid_oracle"] = dict(DEFAULTS["grid"], pose="oracle")
DEFAULTS["app_threshold"] = {}
DEFAULTS["app_cfar"] = {}
FAMILIES = list(DEFAULTS)
TWO_D_ONLY = {"rd_cfar"}


def _space_common(t, two_d, carried):
    p = {}
    p["bg"] = t.suggest_categorical("bg", ["none", "ema"] if carried else ["none", "ema", "empty"])
    p["bg_alpha"] = t.suggest_float("bg_alpha", 0.8, 0.999) if p["bg"] == "ema" else 0.98
    if two_d:
        p["doppler"] = t.suggest_categorical("doppler", ["sum", "zero", "nonzero", "max"])
        p["zero_band"] = t.suggest_int("zero_band", 0, 2)
    p["combine"] = t.suggest_categorical("combine", ["mean", "coh", "max"])
    p["r_min"] = t.suggest_float("r_min", 0.26, 0.8)
    p["n"] = t.suggest_int("n", 1, 6)
    p["m"] = t.suggest_int("m", 1, p["n"])
    p["spread"] = t.suggest_int("spread", 0, 2)
    return p


def space(t, family: str, two_d: bool, carried: bool) -> dict:
    """Optuna search space (define-by-run)."""
    p = dict(DEFAULTS[family])
    if family.startswith("app_"):
        return p
    p.update(_space_common(t, two_d, carried))
    if family in ("thr", "thr_nearest"):
        p["k_db"] = t.suggest_float("k_db", 3, 35)
    if family in ("ca_cfar", "cfar_dbscan"):
        p["kind"] = t.suggest_categorical("kind", ["ca", "go", "so"])
    if family in ("ca_cfar", "os_cfar", "cfar_dbscan"):
        p["guard"] = t.suggest_int("guard", 0, 4)
        p["train"] = t.suggest_int("train", 2, 16)
        p["k_db"] = t.suggest_float("k_db", 3, 25)
    if family == "os_cfar":
        p["os_q"] = t.suggest_float("os_q", 0.5, 0.95)
    if family == "rd_cfar":
        for k, lo, hi in (("guard_d", 0, 2), ("guard_r", 0, 3), ("train_d", 1, 4), ("train_r", 2, 10)):
            p[k] = t.suggest_int(k, lo, hi)
        p["k_db"] = t.suggest_float("k_db", 4, 25)
    if family == "cfar_dbscan":
        p["window"] = t.suggest_int("window", 1, 12)
        p["eps"] = t.suggest_float("eps", 0.08, 0.6)
        p["min_samples"] = t.suggest_int("min_samples", 1, 8)
    if family.startswith("grid"):
        p["front"] = t.suggest_categorical("front", ["ca", "os", "thr"])
        p["guard"] = t.suggest_int("guard", 0, 4)
        p["train"] = t.suggest_int("train", 2, 16)
        p["k_db"] = t.suggest_float("k_db", 3, 25)
        p["g_l_occ"] = t.suggest_float("g_l_occ", 0.2, 2.0)
        p["g_free"] = t.suggest_categorical("g_free", [True, False])
        p["g_l_free"] = t.suggest_float("g_l_free", 0.05, 1.0)
        p["g_decay"] = t.suggest_float("g_decay", 0.5, 0.99)
        p["g_sigma_r"] = t.suggest_float("g_sigma_r", 0.03, 0.2)
        p["g_sigma_az"] = t.suggest_float("g_sigma_az", 2.0, 20.0)
        p["g_blur"] = t.suggest_float("g_blur", 0.0, 2.0)
        p["g_morph"] = t.suggest_categorical("g_morph", ["none", "open", "close", "erode", "dilate"])
        p["g_morph_k"] = t.suggest_categorical("g_morph_k", [3, 5])
        p["g_thr_lo"] = t.suggest_float("g_thr_lo", 0.1, 2.5)
        p["g_thr_hi"] = t.suggest_float("g_thr_hi", p["g_thr_lo"], 3.5)
        p["g_min_cells"] = t.suggest_int("g_min_cells", 1, 20)
        if family != "grid":
            p["speed"] = t.suggest_float("speed", 0.0, 1.6) if family == "grid_imu" else 1.0
    return p


def _front_mask(F, p, kind):
    if kind == "thr":
        return D.threshold_mask(F, p["k_db"])
    if kind == "os":
        return F.P > D.cfar_1d(F.P, "os", p["guard"], p["train"], p["k_db"], p.get("os_q", 0.75))
    return F.P > D.cfar_1d(F.P, kind, p["guard"], p["train"], p["k_db"])


def detect(rec: C.Recording, family: str, p: dict, max_frames=None, F=None, trig=None, poses=None):
    p = dict(DEFAULTS[family], **p)
    if family.startswith("app_"):
        return app_detect(rec, family, max_frames)
    F = F or features(rec, {k: p[k] for k in PRE_KEYS if k in p}, max_frames)
    t0 = time.perf_counter()
    if family in ("thr", "thr_nearest"):
        mask = D.threshold_mask(F, p["k_db"])
    elif family in ("ca_cfar", "cfar_dbscan"):
        mask = _front_mask(F, p, p["kind"])
    elif family == "os_cfar":
        mask = _front_mask(F, p, "os")
    elif family == "rd_cfar":
        mask = D.rd_mask(F, (p["guard_d"], p["guard_r"]), (p["train_d"], p["train_r"]), p["k_db"])
        if mask is None:
            return None, np.nan, F
    else:                                                       # grid families
        mask = _front_mask(F, p, p["front"])
    mask = D.mofn(mask, int(p["m"]), int(p["n"]), int(p["spread"]))
    d = D.peaks(F, mask, p["r_min"], p["r_max"], nearest_only=(family == "thr_nearest"))
    d["score"] = d["snr_db"]
    if family == "cfar_dbscan":
        d = __import__("analysis.cluster", fromlist=["x"]).temporal_dbscan(d, F.n, int(p["window"]), p["eps"],
                                                                           int(p["min_samples"]))
    if family.startswith("grid"):
        d = run_grid(F, d, p, trig, poses)
    ms = (time.perf_counter() - t0) * 1000 / max(F.n, 1)
    return d, ms, F


def run_grid(F, d, p, trig=None, poses=None):
    from .grid import Grid, float_trig
    from . import poses as PS
    g = Grid({k[2:]: v for k, v in p.items() if k.startswith("g_")}, trig or float_trig)
    deltas = None
    if p["pose"] != "none":
        deltas = poses if poses is not None else PS.deltas(F, p["pose"], p.get("speed", 1.0))
    starts = np.searchsorted(d["f"], np.arange(F.n + 1))
    out = {k: [] for k in ("f", "r", "az", "el", "score", "snr_db")}
    for i in range(F.n):
        if deltas is not None and i > 0 and np.all(np.isfinite(deltas[i])):
            g.move(*deltas[i])
        a, b = starts[i], starts[i + 1]
        g.update(d["r"][a:b], d["az"][a:b], d["snr_db"][a:b])
        r, az, sc = g.obstacles()
        out["f"].append(np.full(len(r), i, np.int32)); out["r"].append(r); out["az"].append(az)
        out["el"].append(np.zeros(len(r))); out["score"].append(sc); out["snr_db"].append(sc)
    return {k: np.concatenate(v) for k, v in out.items()}


def app_detect(rec: C.Recording, family: str, max_frames=None):
    """The recording app's own detectors (ravidata.processing: 'threshold' / 'cfar', nearest confirmed obstacle,
    their own preprocessing and defaults), replayed frame by frame."""
    import json
    from ravidata.config import RadarConfig
    from ravidata.cube import CubeShape
    from ravidata.frames import Frame
    from ravidata.processing.pipeline import Processor
    rd = C.load_radar(rec, max_frames=max_frames)
    cfg = RadarConfig.from_dict(rd.config) if rd.config else None
    proc = Processor(cfg)
    name = family.split("_", 1)[1]
    out = {k: [] for k in ("f", "r", "az", "el", "score", "snr_db")}
    t0 = time.perf_counter()
    for i in range(rd.n):
        c = rd.cube[i]
        res = proc.process(Frame(i, int(rd.frame_index[i]), 0.0, float(rd.t[i]), c, CubeShape(*c.shape)))
        o = res.outputs.get(name)
        if o is not None and o.nearest is not None:
            nd = o.nearest
            out["f"].append(i); out["r"].append(nd.range_m)
            az = getattr(nd, "az_deg", np.nan)
            out["az"].append(np.nan if az is None else az); out["el"].append(np.nan)
            out["score"].append(getattr(nd, "snr_db", 0.0)); out["snr_db"].append(getattr(nd, "snr_db", 0.0))
    ms = (time.perf_counter() - t0) * 1000 / max(rd.n, 1)
    d = {k: np.asarray(v, np.int32 if k == "f" else float) for k, v in out.items()}
    return d, ms, None
