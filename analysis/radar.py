"""Radar features: background removal, Doppler handling, RX combining, interferometric angles.

Numerical rules (thesis, section H): cube values are ~1e-2..1e-4 of full scale, so everything is done in float64;
phase differences come from the complex product ``angle(x_b * conj(x_a))`` (never atan(imag/real)); arcsin
arguments are clipped; powers are compared in dB only after the linear arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import common as C
from . import geometry as G

D_OVER_LAMBDA = 0.497
PAIRS = {"az": (2, 1), "el": (1, 0)}     # rotation 180: RX2-RX1 horizontal, RX1-RX0 vertical (orientation.pairs)


@dataclass
class Feat:
    key: str
    take: str
    config: str
    t: np.ndarray            # (n,) host s
    r: np.ndarray            # (B,) range axis, m
    P: np.ndarray            # (n, B) combined power, linear (float64)
    noise: np.ndarray        # (n,) per-frame noise estimate (median power of the usable bins)
    S_az: np.ndarray         # (n, B) complex128 sum of x_b conj(x_a) for the azimuth pair
    S_el: np.ndarray         # (n, B) idem elevation
    RD: np.ndarray | None    # (n, D, B) power per Doppler bin (2D configs)
    v: np.ndarray | None
    pre: dict

    @property
    def n(self):
        return len(self.t)

    @property
    def fps(self):
        return 1.0 / np.median(np.diff(self.t)) if self.n > 1 else np.nan


PRE_DEFAULT = dict(bg="none", bg_alpha=0.98, doppler="sum", zero_band=0, combine="mean")


def _empty_ref(rec: C.Recording):
    """Complex mean cube of the empty-scene part (S0) of the same take and config, if there is one."""
    for r in C.recordings(rec.take, [rec.config]):
        if r.part == "S0" and r.file != rec.file:
            rd = C.load_radar(r)
            return rd.cube.astype(np.complex128).mean(0)
    return None


@lru_cache(maxsize=48)
def _cube(key_file: str, take: str, idx: int, max_frames: int | None):
    rec = next(r for r in C.recordings(take) if str(r.file) == key_file)
    return C.load_radar(rec, max_frames=max_frames)


def features(rec: C.Recording, pre: dict | None = None, max_frames: int | None = None) -> Feat:
    p = dict(PRE_DEFAULT, **(pre or {}))
    rd = _cube(str(rec.file), rec.take, rec.idx, max_frames)
    x = rd.cube.astype(np.complex128)                      # (n, D, 3, B)
    if p["bg"] == "mean":                                  # non-causal (whole recording); removes static scene too
        x = x - x.mean(0, keepdims=True)
    elif p["bg"] == "ema":                                 # causal: x_k - bg_{k-1}, bg_k = a bg_{k-1} + (1-a) x_k
        from scipy.signal import lfilter
        a = float(p["bg_alpha"])
        zi = (a * x[0])[None]
        bg = lfilter([1 - a], [1, -a], x, axis=0, zi=zi)[0]
        x = x - np.concatenate([x[:1], bg[:-1]], 0)
    elif p["bg"] == "empty":                               # empty-scene reference (static takes only)
        ref = _empty_ref(rec)
        if ref is not None:
            x = x - ref[None]
    n, D, _, B = x.shape
    if D > 1:
        k0 = int(np.argmin(np.abs(rd.v_axis))) if rd.v_axis is not None else D // 2
        zb = int(p["zero_band"])
        dsel = {"sum": np.ones(D, bool), "zero": np.abs(np.arange(D) - k0) <= zb,
                "nonzero": np.abs(np.arange(D) - k0) > zb}.get(p["doppler"], np.ones(D, bool))
    else:
        dsel = np.ones(1, bool)
    pw = np.abs(x) ** 2                                    # (n, D, 3, B)
    if p["combine"] == "coh":
        RDc = np.abs(x.sum(2)) ** 2 / 3.0
    elif p["combine"] == "rx1":                            # one receiver (the legacy firmware used RX1)
        RDc = pw[:, :, 1]
    elif p["combine"] == "max":
        RDc = pw.max(2)
    else:
        RDc = pw.mean(2)
    P = RDc[:, dsel].max(1) if p["doppler"] == "max" else RDc[:, dsel].sum(1)
    xs = x[:, dsel]
    a, b = PAIRS["az"]
    S_az = (xs[:, :, b] * np.conj(xs[:, :, a])).sum(1)
    a, b = PAIRS["el"]
    S_el = (xs[:, :, b] * np.conj(xs[:, :, a])).sum(1)
    r_axis = rd.range_m - range_bias()                    # calibrated range (extrinsic step: r_meas = r_true + b)
    use = (r_axis >= G.R_MIN) & (r_axis <= G.R_MAX)
    noise = np.median(P[:, use], 1)
    return Feat(rec.key, rec.take, rec.config, rd.t, r_axis, P, noise, S_az, S_el,
                RDc if D > 1 else None, rd.v_axis, p)


_FCACHE: dict = {}


def features_cached(rec: C.Recording, pre: dict | None = None, max_frames: int | None = None, size: int = 8) -> Feat:
    """features() with a small LRU cache (the optimiser re-uses the same preprocessing across trials)."""
    p = dict(PRE_DEFAULT, **(pre or {}))
    if p["bg"] != "ema":
        p["bg_alpha"] = PRE_DEFAULT["bg_alpha"]                 # alpha only matters for ema
    key = (str(rec.file), tuple(sorted((k, round(v, 4) if isinstance(v, float) else v) for k, v in p.items())),
           max_frames)
    if key in _FCACHE:
        _FCACHE[key] = _FCACHE.pop(key)
        return _FCACHE[key]
    F = features(rec, p, max_frames)
    _FCACHE[key] = F
    while len(_FCACHE) > size:
        _FCACHE.pop(next(iter(_FCACHE)))
    return F


def phase_to_angle(phi, sign=1.0, offset_rad=0.0):
    """Interferometric angle (deg) from a phase difference (rad): arcsin(wrap(sign*phi - off) / (2 pi d/lambda))."""
    ph = np.angle(np.exp(1j * (sign * np.asarray(phi, float) - offset_rad)))
    return np.degrees(np.arcsin(np.clip(ph / (2 * np.pi * D_OVER_LAMBDA), -1.0, 1.0)))


@lru_cache(maxsize=1)
def range_bias() -> float:
    """Range bias b from the `extrinsic` step (0 before it has run); detections use range_m - b."""
    e = C.load_json("extrinsic", None) or {}
    return float(e.get("range_bias_m", 0.0))


@lru_cache(maxsize=1)
def angle_calib() -> dict:
    """Sign / phase offset per pair from the `extrinsic` step (defaults: sign +1, offset 0)."""
    d = C.load_json("angle_calib", None) or {}
    return {"az": d.get("az", {"sign": 1.0, "offset_rad": 0.0}), "el": d.get("el", {"sign": 1.0, "offset_rad": 0.0})}


def angles_at(F: Feat, f: np.ndarray, b: np.ndarray, span: int = 1):
    """Azimuth / elevation (deg) and coherence of detections (frame f, bin b), from the products summed over
    bins b-span..b+span (coherence = |sum| / sum |.|, 1 = one clean target)."""
    cal = angle_calib()
    out = []
    for which, S in (("az", F.S_az), ("el", F.S_el)):
        acc = np.zeros(len(f), np.complex128)
        mag = np.zeros(len(f))
        for d in range(-span, span + 1):
            bb = np.clip(b + d, 0, S.shape[1] - 1)
            s = S[f, bb]
            acc += s
            mag += np.abs(s)
        ang = phase_to_angle(np.angle(acc), cal[which]["sign"], cal[which]["offset_rad"])
        out.append((ang, np.abs(acc) / np.maximum(mag, 1e-300)))
    return out[0][0], out[1][0], out[0][1]
