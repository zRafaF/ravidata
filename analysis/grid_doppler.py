"""Range-azimuth-Doppler occupancy grid (family `grid_doppler`, 2D recordings only).

The other pipelines collapse the Doppler axis into one power per range bin before detecting, so a range bin has one
azimuth and one evidence value whatever moves in it. Here Doppler is kept as a third axis, twice:

1. Detection and angle per range-Doppler cell: 2D CA-CFAR on the range-Doppler map, and the interferometric azimuth
   from the antenna-pair product of THAT cell (summed over +-1 range bin). Two reflectors in one range bin with
   different radial speeds (the floor below and a chair ahead, say) get separate detections and separate angles.
2. The grid has K = 3 Doppler layers, by closing speed (negative v = range decreasing, measured on the carried takes):
   layer 0: |closing| < v1 or receding (radial speed ~ 0: the floor below, walls abeam, anything while standing);
   layer 1: closing at v1..v2 (static things off the walking direction);
   layer 2: closing faster than v2 (static things near the walking direction close at the walking speed).
   Each layer is a log-odds range-azimuth grid (analysis.grid.Grid: decay, occupied stamp, free rays). Obstacles are
   read from the weighted sum of the layers, w = (w0, w1, 1): the weights say how much evidence at each closing speed
   counts, from Doppler gating (w0 = 0) to no use of Doppler (all 1)."""

from __future__ import annotations

import numpy as np

from . import detectors as D
from . import geometry as G
from . import radar as R
from .grid import Grid, float_trig

CLOSING_SIGN = -1.0          # v_axis < 0 = range decreasing (the scene's power sits at v < 0 while walking forward)
DEFAULTS = dict(guard_d=1, guard_r=1, train_d=2, train_r=4, k_db=10.0, gd_v1=0.2, gd_v2=0.6, gd_w0=0.3, gd_w1=0.7,
                gd_max=24)


def detections(rec, F, p, max_frames=None):
    """Range-Doppler cell detections of every frame: dict f, b, k (Doppler bin), r, v (m/s), az, snr_db, layer."""
    rd = R._cube(str(rec.file), rec.take, rec.idx, max_frames)
    RD = F.RD                                                       # (n, D, B) power, receivers combined
    n, Dn, B = RD.shape
    mask = RD > D.cfar_rd(RD, (int(p["guard_d"]), int(p["guard_r"])), (int(p["train_d"]), int(p["train_r"])), p["k_db"])
    gate = (F.r >= p["r_min"]) & (F.r <= p["r_max"])
    pad = np.pad(RD, ((0, 0), (0, 0), (1, 1)))
    mask &= gate[None, None] & (RD >= pad[:, :, :-2]) & (RD >= pad[:, :, 2:])          # local maximum along range
    f, k, b = np.nonzero(mask)
    if not len(f):
        z = np.zeros(0)
        return dict(f=z.astype(np.int32), b=z.astype(np.int32), k=z.astype(np.int32), r=z, v=z, az=z, snr_db=z,
                    layer=z.astype(np.int32))
    use = (F.r >= G.R_MIN) & (F.r <= G.R_MAX)
    noise = np.median(RD[:, :, use].reshape(n, -1), 1)
    snr = 10 * np.log10(np.maximum(RD[f, k, b], 1e-300) / np.maximum(noise[f], 1e-300))
    o = np.lexsort((-snr, f))                                        # strongest gd_max cells per frame
    f, k, b, snr = f[o], k[o], b[o], snr[o]
    keep = (np.arange(len(f)) - np.searchsorted(f, f)) < int(p["gd_max"])
    f, k, b, snr = f[keep], k[keep], b[keep], snr[keep]
    x = rd.cube                                                      # (n, D, 3, B) complex
    a_, b_ = R.PAIRS["az"]
    acc = np.zeros(len(f), np.complex128)
    for d in (-1, 0, 1):
        bb = np.clip(b + d, 0, B - 1)
        acc += x[f, k, b_, bb].astype(np.complex128) * np.conj(x[f, k, a_, bb].astype(np.complex128))
    cal = R.angle_calib()["az"]
    az = R.phase_to_angle(np.angle(acc), cal["sign"], cal["offset_rad"])
    v = np.asarray(rd.v_axis, float)[k]
    closing = CLOSING_SIGN * v
    layer = np.where(closing >= p["gd_v2"], 2, np.where(closing >= p["gd_v1"], 1, 0)).astype(np.int32)
    return dict(f=f.astype(np.int32), b=b.astype(np.int32), k=k.astype(np.int32), r=F.r[b].astype(float), v=v, az=az,
                snr_db=snr, layer=layer)


def run(rec, F, p, max_frames=None, trig=None):
    """Detections dict (f, r, az, el, score, snr_db) of the 3-layer grid; one or more obstacles per frame."""
    d = detections(rec, F, p, max_frames)
    gp = {k[2:]: v for k, v in p.items() if k.startswith("g_")}
    layers = [Grid(gp, trig or float_trig) for _ in range(3)]
    read = Grid(gp, trig or float_trig)                              # read-out: weighted sum of the layers
    w = (float(p["gd_w0"]), float(p["gd_w1"]), 1.0)
    order = np.lexsort((d["layer"], d["f"]))
    d = {k: v[order] for k, v in d.items()}
    key = d["f"].astype(np.int64) * 3 + d["layer"]
    starts = np.searchsorted(key, np.arange(3 * F.n + 1))
    out = {k: [] for k in ("f", "r", "az", "el", "score", "snr_db")}
    for i in range(F.n):
        L = 0.0
        for j, g in enumerate(layers):
            a, b = starts[3 * i + j], starts[3 * i + j + 1]
            g.update(d["r"][a:b], d["az"][a:b], d["snr_db"][a:b])
            if w[j] > 0:
                L = L + w[j] * g.L
        read.L = np.asarray(L, np.float32) if not np.isscalar(L) else np.zeros_like(layers[0].L)
        r, az, sc = read.obstacles()
        out["f"].append(np.full(len(r), i, np.int32)); out["r"].append(r); out["az"].append(az)
        out["el"].append(np.zeros(len(r))); out["score"].append(sc); out["snr_db"].append(sc)
    return {k: np.concatenate(v) for k, v in out.items()}
