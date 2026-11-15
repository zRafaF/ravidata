"""F12: causal running median of the nearest detection (threshold detector), indoor_course/01_D1, one approach.
Exports a text-free range-time raster (plus a faint copy) and the detector outputs / found-early-lost strips as JSON."""
from __future__ import annotations

import json

import numpy as np

from .. import common as C, pipeline as PL, radar as R
from . import common as PC

REC = "indoor_course/01_D1_tcc-1d-40mm"
T0, DUR = 12.6, 14.0                  # seconds after the first radar frame (truth falls from 1.7 m to 0.5 m at ~20 s)
RLIM = (0.2, 3.0)
TOL = 0.25


def nearest_per_frame(d, n):
    near = np.full(n, np.nan)
    if d is not None and len(d["f"]):
        o = np.lexsort((d["r"], d["f"]))
        f, r = d["f"][o], d["r"][o]
        first = np.r_[True, f[1:] != f[:-1]]
        near[f[first]] = r[first]
    return near


def classify(det, truth):
    """0 found (within TOL), 1 early (nearer than the truth by more than TOL), 2 lost (nothing, or farther)."""
    c = np.full(len(det), 2)
    has = np.isfinite(det)
    c[has & (det < truth - TOL)] = 1
    c[has & (np.abs(det - truth) <= TOL)] = 0
    return c


def runs(t, cls, dt):
    out, s = [], 0
    for i in range(1, len(cls) + 1):
        if i == len(cls) or cls[i] != cls[s]:
            out.append([float(t[s] - dt / 2), float(t[i - 1] + dt / 2), int(cls[s])])
            s = i
    return out


def run():
    rec = [r for r in C.recordings("indoor_course") if r.key == REC][0]
    rd = C.load_radar(rec)
    t = rd.t - rd.t[0]
    n = len(t)
    dt = float(np.median(np.diff(t)))
    tuned = json.loads((C.RESULTS / "analysis-v2" / "tables" / "opt_best_params_pooled_arc.json").read_text())["thr"]
    setups = [
        ("default", {}),
        ("tuned_med1", dict(tuned, med=1)),
        ("tuned", dict(tuned)),
    ]
    T = C.load_table("truth_frames")
    g = T[T.rec == REC].sort_values("f")
    truth = np.full(n, np.nan)
    m = min(n, len(g))
    truth[:m] = np.where(g.arc_complete.fillna(False).to_numpy(bool), g.arc_r.to_numpy(), np.nan)[:m]

    sel = np.flatnonzero((t >= T0) & (t < T0 + DUR))
    f0, f1 = sel[0], sel[-1] + 1
    ts, tr = t[f0:f1], truth[f0:f1]

    # raster
    x = rd.cube[f0:f1].astype(np.complex128)
    P = (np.abs(x) ** 2).sum(1).mean(1)
    r_ax = rd.range_m - R.range_bias()
    use = np.flatnonzero((r_ax >= RLIM[0]) & (r_ax <= RLIM[1]))
    a01 = PC.db_over_median(P)[:, use]                              # (frames, bins)
    rgb = PC.to_rgb(a01.T[::-1])                                   # far range on top
    import cv2
    rgb = cv2.resize(rgb, (rgb.shape[1], rgb.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    faint = (255 - (255 - rgb.astype(float)) * 0.28).astype(np.uint8)
    PC.save_png("f12_raster", rgb)
    PC.save_png("f12_raster_fraco", faint)
    db = float(np.median(np.diff(r_ax)))
    xlim = [float(ts[0] - dt / 2), float(ts[-1] + dt / 2)]
    ylim = [float(r_ax[use[0]] - db / 2), float(r_ax[use[-1]] + db / 2)]

    rows = []
    for name, p in setups:
        d, ms, F = PL.detect(rec, "thr", p)
        near = nearest_per_frame(d, n)[f0:f1]
        cls = classify(near, tr)
        ok = np.isfinite(tr)
        cnt = np.bincount(cls[ok], minlength=3) / max(ok.sum(), 1)
        dd = np.abs(np.diff(near))
        dd = dd[np.isfinite(dd)]
        pp = dict(PL.DEFAULTS["thr"], **p)
        rows.append(dict(
            id=name,
            params={k: float(pp[k]) for k in ("k_db", "med", "r_min", "m", "n", "spread", "noise_q", "integ")},
            rates=dict(found=float(cnt[0]), early=float(cnt[1]), lost=float(cnt[2]), n_frames=int(ok.sum())),
            flicker_cm=float(np.median(dd) * 100), flicker_mean_cm=float(np.mean(dd) * 100),
            det_frac=float(np.isfinite(near).mean()),
            points=[[float(a), float(b)] for a, b in zip(ts, near) if np.isfinite(b)],
            strip=runs(ts, cls, dt)))
        print(name, rows[-1]["rates"], rows[-1]["flicker_cm"], rows[-1]["det_frac"])
    PC.save_json("f12_dados", dict(rec=REC, t0=T0, dur=DUR, xlim=xlim, ylim=ylim, unit_x="s after first radar frame",
                                   unit_y="m", tol_m=TOL, raster="f12_raster", raster_faint="f12_raster_fraco",
                                   truth=[[float(a), float(b)] for a, b in zip(ts, tr) if np.isfinite(b)], rows=rows))


if __name__ == "__main__":
    run()
