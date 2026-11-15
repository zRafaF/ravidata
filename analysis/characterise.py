"""A1-A6: the sensor itself.

Tables:
  noise       per recording x RX x range bin: mean power (static returns + leakage), temporal noise (variance
              of the complex value over time; meaningful when the rig is still) and, in 2D, the power of the
              Doppler bins away from zero (|v| >= 0.75 m/s; noise + movers);
  targets     per static recording x tag x frame: measured range / power / SNR / angles at the tag's return,
              truth range / angles, power at twice the range (multipath ghost);
  timing      per recording: frames, rate, frame-index gaps (lost frames), inter-frame jitter;
  sync_check  carried takes: agreement of the radar's nearest return with the truth's nearest obstacle when the
              radar clock is shifted by a lag (-1 .. +1 s); the take's offset is NOT changed by this, it is a check."""

from __future__ import annotations

import numpy as np

from . import common as C
from . import geometry as G
from . import radar as R


def noise_rows(rec):
    rd = C.load_radar(rec, C.default_max_frames())
    x = rd.cube.astype(np.complex128)                                  # (n, D, 3, B)
    n, D, nrx, B = x.shape
    out = []
    k0 = int(np.argmin(np.abs(rd.v_axis))) if rd.v_axis is not None else 0
    mov = np.abs(np.arange(D) - k0) >= 3 if D > 1 else None
    for rx in range(nrx):
        xr = x[:, :, rx]
        xs = xr[:, k0] if D > 1 else xr[:, 0]
        mean_db = 10 * np.log10(np.maximum((np.abs(xs) ** 2).mean(0), 1e-300))
        tv_db = 10 * np.log10(np.maximum(np.var(xs, axis=0), 1e-300))
        dop = 10 * np.log10(np.maximum((np.abs(xr[:, mov]) ** 2).mean((0, 1)), 1e-300)) if mov is not None \
            else np.full(B, np.nan)
        for b in range(B):
            out.append((rec.key, rec.take, rec.config, rec.part, C.ENV[rec.take], rx, b, rd.range_m[b],
                        mean_db[b], tv_db[b], dop[b]))
    return out


def timing_row(rec):
    import h5py
    with h5py.File(rec.file, "r") as h:
        t = h["t_host"][:]
        fi = h["frame_index"][:].astype(np.int64)
    dt = np.diff(t)
    gaps = np.diff(fi)
    return dict(rec=rec.key, take=rec.take, config=rec.config, part=rec.part, frames=len(t),
                duration_s=float(t[-1] - t[0]), fps=float((len(t) - 1) / (t[-1] - t[0])),
                lost_frames=int(np.clip(gaps - 1, 0, None).sum()), lost_pct=float(np.clip(gaps - 1, 0, None).sum()
                                                                                 / max(1, fi[-1] - fi[0]) * 100),
                dt_median_ms=float(np.median(dt) * 1e3), dt_std_ms=float(np.std(dt) * 1e3),
                dt_p99_ms=float(np.percentile(dt, 99) * 1e3), dt_max_ms=float(dt.max() * 1e3))


def target_frames(tags, rec, window_m=0.25):
    """Per frame and tag: the return nearest the tag's range (peak within +-window_m)."""
    t = C.radar_times(rec, C.default_max_frames())
    q = tags[(tags["take"] == rec.take) & (tags.t_host >= t[0]) & (tags.t_host <= t[-1]) & (tags.reproj_px < 2)
             & (tags.view_deg < 70) & (tags.kind != "floor_tag")]
    if not len(q):
        return []
    rows = []
    for bg in ("none", "empty"):
        F = R.features(rec, {"bg": bg}, C.default_max_frames())
        for i, g in q.groupby("id"):
            p = g[["tx", "ty", "tz"]].median().to_numpy()
            r, az, el = G.polar(p)
            w = np.flatnonzero(np.abs(F.r - r) <= window_m)
            if not len(w):
                continue
            b = w[np.argmax(F.P[:, w], 1)]
            f = np.arange(F.n)
            a_az, a_el, coh = R.angles_at(F, f, b, span=0)
            b2 = np.clip(np.round(2 * r / (F.r[1] - F.r[0])).astype(int), 0, len(F.r) - 1)
            for k in range(F.n):
                rows.append((rec.key, rec.take, rec.config, rec.part, bg, int(i), C.TAG_KIND.get(int(i), "other"),
                             k, r, az, el, F.r[b[k]], 10 * np.log10(max(F.P[k, b[k]], 1e-300)),
                             10 * np.log10(max(F.P[k, b[k]], 1e-300) / max(F.noise[k], 1e-300)),
                             a_az[k], a_el[k], coh[k], 10 * np.log10(max(F.P[k, b2], 1e-300) / max(F.noise[k], 1e-300))))
    return rows


def sync_rows(rec, lags=np.arange(-2.5, 2.5001, 0.04)):
    from .evaluate import truth_arrays
    from . import pipeline as PL
    d, _, F = PL.detect(rec, "thr", {}, C.default_max_frames())
    ta = truth_arrays(rec.key, F.n)
    if ta is None:
        return []
    obs_r = ta[0]
    near_t = np.nanmin(np.where(np.isfinite(obs_r), obs_r, np.inf), 1)
    ok = np.isfinite(near_t)
    near_d = np.full(F.n, np.nan)
    if len(d["f"]):
        first = np.r_[True, d["f"][1:] != d["f"][:-1]]
        near_d[d["f"][first]] = np.minimum.reduceat(d["r"], np.flatnonzero(first))
    rows = []
    for lag in lags:
        tt = np.interp(F.t + lag, F.t[ok], near_t[ok]) if ok.sum() > 2 else np.full(F.n, np.nan)
        m = np.isfinite(near_d) & np.isfinite(tt)
        rows.append(dict(rec=rec.key, take=rec.take, config=rec.config, lag_s=float(lag),
                         agree=float((np.abs(near_d[m] - tt[m]) < 0.2).mean()) if m.any() else np.nan,
                         corr=float(np.corrcoef(near_d[m], tt[m])[0, 1]) if m.sum() > 10 else np.nan))
    return rows


def run(takes=None):
    import pandas as pd
    takes = C.available_takes(takes)
    tags = C.load_tags(required=False)
    nrows, trows, tim, srows = [], [], [], []
    for take in takes:
        for rec in C.bar(C.recordings(take), desc=f"sensor {take}", unit="rec"):
            nrows += noise_rows(rec)
            tim.append(timing_row(rec))
            if take in C.STATIC and tags is not None and rec.part != "S0":
                trows += target_frames(tags, rec)
            if take in C.CARRIED and C.load_table("truth_frames", required=False) is not None:
                srows += sync_rows(rec)
    N = pd.DataFrame(nrows, columns=["rec", "take", "config", "part", "env", "rx", "bin", "r", "mean_db", "tvar_db",
                                     "dop_db"])
    T = pd.DataFrame(trows, columns=["rec", "take", "config", "part", "bg", "id", "kind", "f", "r_true", "az_true",
                                     "el_true", "r_meas", "p_db", "snr_db", "az_meas", "el_meas", "coh",
                                     "ghost2_snr_db"])
    C.save_table(N, "noise", takes=takes)
    C.save_table(T, "targets", takes=takes)
    C.save_table(pd.DataFrame(tim), "timing", takes=takes)
    S = pd.DataFrame(srows)
    C.save_table(S, "sync_check", takes=takes)
    if len(T):
        g = T[T.bg == "empty"].groupby(["take", "id"]).agg(r_true=("r_true", "first"), r_err=("r_meas", "median"),
                                                           r_sd=("r_meas", "std"), snr=("snr_db", "median"))
        g["r_err"] -= g.r_true
        C.log("\n" + g.round(3).to_string())
    if len(S):
        best = S.loc[S.groupby("rec").agree.idxmax()][["rec", "lag_s", "agree"]]
        C.log("sync check (best lag per recording; the dataset offset is unchanged):\n" + best.round(2).to_string())
    return N, T
