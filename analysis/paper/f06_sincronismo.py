"""F06: video offset check. (a) ridge contrast at the truth range versus absolute video offset, three independent
estimators (map truth arc_r_fov, map truth sectors, camera height above the floor tags), z-scored, averaged over the
three recordings; indoor_course and outdoor_course (control). (b) range-time rasters of the same window of
indoor_course/01 (text-free) with truth points before (analysis-v1 truth, offset +0.56 s) and after (analysis-v2)."""
from __future__ import annotations

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .. import common as C, radar as R
from . import common as P

OLD, NEW = 0.56, -2.44
OFFS = np.round(np.arange(-6, 4.0001, 0.1), 2)
BIN = 0.04
V1 = C.RESULTS / "analysis-v1" / "tables"
V2 = C.RESULTS / "analysis-v2" / "tables"


def contrast(rec):
    with h5py.File(rec.file, "r") as h:
        t = h["t_host"][:]
        c = h["cube"][:].astype(np.complex64)
    L = 10 * np.log10((np.abs(c) ** 2).sum(1).mean(1) + 1e-15)
    Ln = L - (uniform_filter1d(L, 17, axis=1) * 17 - uniform_filter1d(L, 5, axis=1) * 5) / 12       # ridge contrast, dB
    return t, Ln


def prof_at(t, Ln, tq, rq):
    """Mean ridge contrast at the truth range, truth times shifted by lag = absolute offset - OLD (OFFS - OLD)."""
    ok = np.isfinite(rq) & (rq > 0.4) & (rq < 3.8)
    out = []
    for o in OFFS - OLD:
        tl = tq[ok] + o
        m = (tl > t[0]) & (tl < t[-1])
        if m.sum() < 30:
            out.append(np.nan); continue
        fi = np.clip(np.searchsorted(t, tl[m]), 0, len(t) - 1)
        bi = np.clip(np.round(rq[ok][m] / BIN).astype(int), 1, 126)
        out.append(np.maximum(np.maximum(Ln[fi, bi - 1], Ln[fi, bi]), Ln[fi, bi + 1]).mean())
    return np.array(out)


def z(p):
    return (p - np.nanmedian(p)) / np.nanstd(p)


def curves(take, F, S, tags):
    acc = {"arc_r_fov": [], "setores": [], "altura": []}
    peaks = {k: [] for k in acc}
    s = C.take_json(take)["sync"]
    q = tags[(tags["take"] == take) & (tags.kind == "floor_tag") & (tags.reproj_px < 2)]
    n = np.array([cv2.Rodrigues(np.array([a, b, c]))[0][:, 2] for a, b, c in zip(q.rx, q.ry, q.rz)])
    hgt = np.abs((n * q[["tx", "ty", "tz"]].to_numpy()).sum(1))
    th0 = s["video_a"] * q.t_video.to_numpy() + s["video_b"]            # host time for video offset 0
    for rec in C.recordings(take):
        t, Ln = contrast(rec)
        fr = F[F.rec == rec.key].sort_values("f"); tt = fr.t.to_numpy()
        sc = S[S.rec == rec.key].sort_values(["f", "sector"])
        O = sc.obs_r.to_numpy().reshape(-1, 14)[:len(tt)]
        # truth times were made with offset +0.56: absolute offset = 0.56 + lag (prof_at scans lag = OFFS - 0.56)
        pa = prof_at(t, Ln, tt, fr.arc_r_fov.to_numpy())
        ps = prof_at(t, Ln, np.repeat(tt, 6), O[:, 4:10].ravel())
        m = (hgt > 0.4) & (hgt < 3)
        ph = prof_at_abs(t, Ln, th0[m], hgt[m])
        for k, p in (("arc_r_fov", pa), ("setores", ps), ("altura", ph)):
            acc[k].append(z(p)); peaks[k].append(p)
    return acc, peaks


def prof_at_abs(t, Ln, th0, h):
    out = []
    for o in OFFS:
        tl = th0 + o
        m = (tl > t[0]) & (tl < t[-1])
        if m.sum() < 30:
            out.append(np.nan); continue
        fi = np.clip(np.searchsorted(t, tl[m]), 0, len(t) - 1)
        bi = np.clip(np.round(h[m] / BIN).astype(int), 1, 126)
        out.append(np.maximum(np.maximum(Ln[fi, bi - 1], Ln[fi, bi]), Ln[fi, bi + 1]).mean())
    return np.array(out)


def window(take_rec, F, dur=12.0):
    g = F[F.rec == take_rec].sort_values("f")
    a, tt = g.arc_r_fov.to_numpy(), g.t.to_numpy()
    best = (-1, tt[0])
    for t0 in np.arange(tt[0] + 2, tt[-1] - dur - 1, 1.0):
        m = (tt >= t0) & (tt < t0 + dur)
        v = a[m]
        if np.isfinite(v).mean() < 0.9:
            continue
        third = max(1, m.sum() // 3)
        best = max(best, (np.nanmedian(v[:third]) - np.nanmin(v[third:]) + 0.3 * (np.nanpercentile(v, 95) - np.nanpercentile(v, 5)), t0))
    return best[1]


def run():
    F1 = pd.read_parquet(V1 / "truth_frames.parquet"); S1 = pd.read_parquet(V1 / "truth_sectors.parquet")
    F2 = pd.read_parquet(V2 / "truth_frames.parquet")
    tags = pd.read_parquet(V1 / "tags.parquet")
    res = {"offsets": OFFS, "old": OLD, "new": NEW}
    for take in ("indoor_course", "outdoor_course"):
        acc, raw = curves(take, F1, S1, tags)
        out = {}
        for k in acc:
            zs = np.array(acc[k])
            mean = np.nanmean(zs, 0)
            out[k] = dict(mean=mean, each=zs, peak_offset=float(OFFS[np.nanargmax(mean)]),
                          peaks_each=[float(OFFS[np.nanargmax(p)]) for p in zs])
            print(take, k, "peak", out[k]["peak_offset"], out[k]["peaks_each"], "z@old %.2f z@new %.2f" % (
                mean[np.argmin(abs(OFFS - OLD))], mean[np.argmin(abs(OFFS - NEW))]))
        res[take] = out
    P.save_json("f06_curvas", res)
    # (b) rasters
    key = "indoor_course/01_D1_tcc-1d-40mm"
    rec = [r for r in C.recordings("indoor_course") if r.key == key][0]
    t, Ln = contrast(rec)
    DUR = 12.0
    t0 = window(key, F2, DUR)
    rd = C.load_radar(rec, skip_lead_in=False)
    rng = rd.range_m - R.range_bias()
    Pw = (np.abs(rd.cube.astype(np.complex64)) ** 2).sum(1).mean(1)
    w = (rd.t >= t0) & (rd.t < t0 + DUR)
    keep = (rng >= 0.2) & (rng <= 4.2)
    A = P.db_over_median(Pw[w], lo=3, hi=30)[:, keep]
    img = np.repeat(np.flipud(A.T), 2, axis=1)
    P.save_png("f06_raster", P.to_rgb(img))
    info = dict(xlim=[0, DUR], ylim=[float(rng[keep][0]), float(rng[keep][-1])], rec=key, t0_after_first_frame=t0 - rd.t[0])
    for nm, F in (("antes", F1), ("depois", F2)):
        g = F[(F.rec == key) & (F.t >= t0) & (F.t < t0 + DUR)]
        ok = np.isfinite(g.arc_r_fov)
        info[nm] = np.c_[g.t[ok] - t0, g.arc_r_fov[ok]]
    P.save_json("f06_raster", info)
    print("window", t0 - rd.t[0])


if __name__ == "__main__":
    run()
