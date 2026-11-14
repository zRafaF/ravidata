"""Candidate detectors (first iteration: two simple, transparent baselines).

Both report the *nearest* confirmed obstacle, which is the product's minimum output (1D
distance). If a height estimate is available the output also carries it, and an optional
height gate can ignore returns that are above head clearance or on the ground.

Confirmation is M-of-N persistence per range bin (with +-1 bin tolerance, since obstacles
migrate in range while the user walks).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .base import ParamSpec, Tunable

COMMON_SPECS = [
    ParamSpec("min_range_m", 0.2, 0.0, 3.0, 0.05, help="Closest range reported."),
    ParamSpec("max_range_m", 2.5, 0.2, 10.0, 0.1, help="Farthest range reported / alert range."),
    ParamSpec("persist_m", 2, 1, 10, 1, kind="int", help="M in M-of-N confirmation."),
    ParamSpec("persist_n", 3, 1, 10, 1, kind="int", help="N in M-of-N confirmation."),
    ParamSpec("height_gate", False, kind="bool", help="Use the height estimate to drop overhead/ground returns."),
    ParamSpec("head_clear_m", 2.1, 1.0, 3.0, 0.05, help="Returns higher than this are 'safe overhead' (if gated)."),
    ParamSpec("ground_max_m", 0.25, 0.0, 1.0, 0.05, help="Returns lower than this are 'ground' (if gated)."),
    ParamSpec("report", "nearest", kind="choice", choices=["nearest", "strongest"],
              help="Which confirmed obstacle drives the output: the nearest one (product default) or the strongest."),
    # EXP-010: horizontal FOV output levels. Detection always runs on the full data; this only decides which
    # confirmed obstacles may drive the output/alert. The gate is a CORRIDOR (straight path ahead, half-width in
    # metres), not a cone, and it is fail-safe: an obstacle is dropped only if it is outside the corridor by more
    # than fov_keep_sigma x its own lateral uncertainty (angle coherence -> sigma).
    ParamSpec("fov", "normal", kind="choice", choices=["normal", "narrow", "laser"],
              help="Output FOV: normal = every confirmed obstacle; narrow = only obstacles in a corridor of "
                   "+-narrow_half_m ahead; laser = +-laser_half_m (the straight line from boresight, "
                   "limited by the angle accuracy)."),
    ParamSpec("narrow_half_m", 0.45, 0.05, 2.0, 0.05, help="Half-width of the 'narrow' corridor."),
    ParamSpec("laser_half_m", 0.2, 0.02, 1.0, 0.02, help="Half-width of the 'laser' corridor."),
    ParamSpec("fov_keep_sigma", 1.0, 0.0, 4.0, 0.25,
              help="Keep an obstacle whose lateral offset is uncertain: dropped only if |lateral| - k*sigma > half-width."),
    ParamSpec("az_search_rows", 2, 0, 8, 1, kind="int",
              help="2D: Doppler rows around the detection's cell searched for an in-corridor angle."),
    ParamSpec("az_search_db", 10.0, 0.0, 30.0, 1.0,
              help="2D: only cells within this many dB of the detection's strongest cell may give its angle."),
    ParamSpec("az_min_snr_db", 12.0, 0.0, 40.0, 1.0,
              help="2D: ... and at least this far over the range-Doppler noise (no angle from noise cells)."),
]


def _az_sigma(az, conf, f):
    import math
    c = min(1.0, max(1e-3, conf))
    s_phi = math.sqrt(-2.0 * math.log(c))                        # rad, phase (wrapped normal)
    dl = getattr(f, "d_over_lambda", 0.486) or 0.486
    return s_phi / (2 * math.pi * dl * max(0.2, math.cos(math.radians(az))))


def nb(v, f, lo: int = 0) -> int:
    """EXP-011: a bin-count parameter scaled to the actual bin size (Features.bin_scale; 1 in 2D)."""
    return max(lo, int(round(float(v) * getattr(f, "bin_scale", 1.0))))


def annotate_az(d, f, kcell=None, p=None):
    """Horizontal angle, lateral offset and its 1-sigma for a detection (EXP-010).
    2D: the cells of the detection's range bin (+-1 bin) whose Doppler row is within az_search_rows of the
    detection's own cell (kcell; all rows if unknown), that are within az_search_db of the strongest of them
    and az_min_snr_db over the noise; of those, the one MOST inside the corridor is used (fail-safe: a target
    sharing its range with strong side clutter keeps its own angle). 1D: the bin's angle.
    sigma from the angle coherence c (wrapped normal: c = exp(-s^2/2))."""
    import math
    p = p or {}
    if getattr(f, "rd_az_deg", None) is not None:
        nd, R = f.rd_az_deg.shape
        b0, b1 = max(0, d.bin - 1), min(R, d.bin + 2)
        if kcell is not None:
            k = int(kcell[d.bin]); w = int(p.get("az_search_rows", 2))
            k0, k1 = max(0, k - w), min(nd, k + w + 1)
        else:
            k0, k1 = 0, nd
        pw = f.rd_pow[k0:k1, b0:b1]
        n = f.rd_noise if getattr(f, "rd_noise", None) == getattr(f, "rd_noise", None) and getattr(f, "rd_noise", 0) else None
        ok = pw >= pw.max() * 10 ** (-p.get("az_search_db", 10.0) / 10)
        if n:
            ok &= pw >= n * 10 ** (p.get("az_min_snr_db", 12.0) / 10)
        if kcell is not None:
            ok[int(kcell[d.bin]) - k0, d.bin - b0] = True    # the detection's own cell always counts
        if not ok.any():
            ok = pw >= pw.max()
        best = None
        for kk, bb in zip(*np.nonzero(ok)):
            az, conf = float(f.rd_az_deg[k0 + kk, b0 + bb]), float(f.rd_az_conf[k0 + kk, b0 + bb])
            if az != az:
                continue
            lat = d.range_m * math.sin(math.radians(az))
            sg = min(5.0, d.range_m * math.cos(math.radians(az)) * _az_sigma(az, conf, f))
            key = abs(lat) - p.get("fov_keep_sigma", 1.0) * sg
            if best is None or key < best[0]:
                best = (key, az, conf, lat, sg)
        if best is None:
            return
        _, d.az_deg, d.az_conf, d.lateral_m, d.lat_sigma_m = best
        return
    if getattr(f, "az_deg", None) is None:
        return
    az = float(f.az_deg[d.bin])
    conf = float(f.az_conf[d.bin]) if getattr(f, "az_conf", None) is not None else 1.0
    if az != az:
        return
    d.az_deg, d.az_conf = az, conf
    d.lateral_m = d.range_m * math.sin(math.radians(az))
    d.lat_sigma_m = min(5.0, d.range_m * math.cos(math.radians(az)) * _az_sigma(az, conf, f))


def fov_filter(dets, f, p, kcell=None):
    """Mark in_fov on every detection (EXP-010) and return the obstacles allowed to drive the output."""
    level = p.get("fov", "normal")
    half = {"narrow": p.get("narrow_half_m", 0.45), "laser": p.get("laser_half_m", 0.2)}.get(level)
    out = []
    for d in dets:
        annotate_az(d, f, kcell, p)
        if half is not None and d.lateral_m == d.lateral_m:
            d.in_fov = abs(d.lateral_m) - p.get("fov_keep_sigma", 1.0) * d.lat_sigma_m <= half
        if d.cls == "obstacle" and d.in_fov:
            out.append(d)
    return out


@dataclass
class Detection:
    bin: int
    range_m: float
    snr_db: float
    elev_deg: float
    height_m: float
    confirmed: bool = True
    cls: str = "obstacle"         # obstacle | overhead | ground
    az_deg: float = float("nan")  # EXP-010 horizontal angle (sign not yet verified) and its coherence
    az_conf: float = float("nan")
    lateral_m: float = float("nan")
    lat_sigma_m: float = float("nan")
    in_fov: bool = True           # inside the selected output FOV (fov param); outside = shown, not alerted


@dataclass
class DetectorOutput:
    name: str
    detections: list[Detection]
    nearest: Detection | None
    threshold_db: np.ndarray | None      # absolute threshold per bin (for plotting), dB
    raw_mask: np.ndarray                  # per-bin single-frame decisions
    confirmed_mask: np.ndarray
    alert: bool
    debug: dict = field(default_factory=dict)


class Detector(Tunable):
    name = "base"
    description = ""

    def __init__(self, **kw):
        self._hist: deque[np.ndarray] = deque(maxlen=10)   # before Tunable.__init__: overrides call on_param_change
        super().__init__(**kw)

    def reset(self):
        self._hist.clear()

    def on_param_change(self, name):
        if name in ("persist_n",):
            self._hist = deque(self._hist, maxlen=max(1, int(self.p["persist_n"])))

    def raw_decisions(self, f) -> tuple[np.ndarray, np.ndarray | None]:  # pragma: no cover
        raise NotImplementedError

    def process(self, f) -> DetectorOutput:
        raw, thr = self.raw_decisions(f)
        raw = raw & f.valid & (f.r_m >= self.p["min_range_m"]) & (f.r_m <= self.p["max_range_m"])
        dil = raw.copy()
        for k in range(1, nb(1, f, 1) + 1):         # +-1 bin at the reference bin size (EXP-011)
            dil[k:] |= raw[:-k]
            dil[:-k] |= raw[k:]
        if self._hist.maxlen != max(1, int(self.p["persist_n"])):
            self._hist = deque(self._hist, maxlen=max(1, int(self.p["persist_n"])))
        self._hist.append(dil)
        count = np.sum(np.stack(self._hist), axis=0)
        confirmed = raw & (count >= min(self.p["persist_m"], len(self._hist)))
        dets = []
        for b in np.flatnonzero(confirmed):
            h = float(f.height_m[b])
            cls = "obstacle"
            if self.p["height_gate"]:
                if h > self.p["head_clear_m"]:
                    cls = "overhead"
                elif h < self.p["ground_max_m"]:
                    cls = "ground"
            dets.append(Detection(int(b), float(f.r_m[b]), float(f.snr_db[b]), float(f.elev_deg[b]), h, True, cls))
        # merge contiguous bins -> keep the nearest bin of each blob
        merged = []
        gap = nb(1, f, 1)
        for d in dets:
            if merged and d.bin - merged[-1][-1].bin <= gap:
                merged[-1].append(d)
            else:
                merged.append([d])
        blobs = [min(g, key=lambda x: x.bin) for g in merged]
        for g, b in zip(merged, blobs):
            b.snr_db = max(x.snr_db for x in g)
        hazards = fov_filter(blobs, f, self.p)
        nearest = pick_output(hazards, self.p.get("report", "nearest"))
        return DetectorOutput(self.name, blobs, nearest, thr, raw, confirmed, nearest is not None)


def pick_output(hazards: list[Detection], mode: str) -> Detection | None:
    if not hazards:
        return None
    if mode == "strongest":
        return max(hazards, key=lambda x: x.snr_db)
    return min(hazards, key=lambda x: x.range_m)


class ThresholdNearest(Detector):
    name = "threshold"
    description = "Fixed SNR threshold over the noise floor (see floor_mode), nearest confirmed bin."
    SPECS = [ParamSpec("k_db", 12.0, 0.0, 40.0, 0.5, help="Required SNR over the tracked floor.")] + COMMON_SPECS

    def raw_decisions(self, f):
        thr = f.floor_db + self.p["k_db"]
        return f.p_db > thr, thr


class CfarNearest(Detector):
    name = "cfar"
    description = "1D CFAR along range (CA or OS) on the combined power, nearest confirmed bin."
    SPECS = [
        ParamSpec("kind", "os", kind="choice", choices=["ca", "os"], help="Cell-averaging or ordered-statistic."),
        ParamSpec("guard", 1, 0, 8, 1, kind="int", help="Guard cells each side."),
        ParamSpec("train", 4, 1, 32, 1, kind="int", help="Training cells each side."),
        ParamSpec("os_rank", 0.7, 0.1, 0.95, 0.05, help="OS-CFAR quantile of the training cells."),
        ParamSpec("k_db", 8.0, 0.0, 30.0, 0.5, help="Threshold over the local estimate."),
        ParamSpec("min_snr_db", 6.0, 0.0, 30.0, 0.5, help="Also require this SNR over the tracked floor."),
    ] + COMMON_SPECS

    def raw_decisions(self, f):
        p = 10 ** (f.p_db / 10)
        R = len(p)
        g, t = nb(self.p["guard"], f), nb(self.p["train"], f, 1)
        est = np.empty(R)
        for i in range(R):
            left = p[max(0, i - g - t): max(0, i - g)]
            right = p[min(R, i + g + 1): min(R, i + g + t + 1)]
            tr = np.concatenate([left, right])
            if tr.size == 0:
                est[i] = p[i]
            elif self.p["kind"] == "ca":
                est[i] = tr.mean()
            else:
                est[i] = np.quantile(tr, self.p["os_rank"])
        thr = 10 * np.log10(est + 1e-12) + self.p["k_db"]
        raw = (f.p_db > thr) & (f.snr_db > self.p["min_snr_db"])
        return raw, thr


DETECTORS: dict[str, type[Detector]] = {
    ThresholdNearest.name: ThresholdNearest,
    CfarNearest.name: CfarNearest,
}
