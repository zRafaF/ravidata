"""Scoring detections against the truth (same representation for both: nearest point per 4 deg sector in the
camera FOV, positions in the radar range-azimuth plane).

Per frame: Hungarian matching on plane distance -> TP / FN / FP inside the rings 0.10 / 0.25 / 0.50 / 1.00 m;
GOSPA (c = 0.5 m, p = 2, alpha = 2) with its localisation / missed / false parts; corridor (nearest obstacle in
|X| < 0.5 m, 0.3 < Z < 3 m) truth vs detection. Per detection: greedy-by-score matching per ring (for AP, as in
nuScenes) and the false-positive cause (behind_obstacle / ghost / floor / unlabelled / clutter)."""

from __future__ import annotations

import numpy as np

from . import geometry as G
from .truth import CORRIDOR

RINGS = (0.10, 0.25, 0.50, 1.00)
GOSPA_C, GOSPA_P = 0.5, 2


def reduce_sectors(d):
    """Nearest detection per sector inside the FOV; plus counts of raw detections in / out of the FOV per frame."""
    az = np.where(np.isfinite(d["az"]), d["az"], 0.0)
    s = G.sector_of(az)
    inn = s >= 0
    f = d["f"]
    key = f.astype(np.int64) * 100 + s
    idx = np.flatnonzero(inn)
    o = idx[np.lexsort((d["r"][idx], key[idx]))]
    first = np.r_[True, key[o][1:] != key[o][:-1]] if len(o) else np.zeros(0, bool)
    keep = o[first]
    red = {k: v[keep] for k, v in d.items()}
    red["sector"] = s[keep]
    red["az_used"] = az[keep]
    return red, inn


def corridor_nearest(d, n_frames):
    az = np.where(np.isfinite(d["az"]), d["az"], 0.0)
    X, Z = G.plane(d["r"], az)
    m = (np.abs(X) < CORRIDOR["half"]) & (Z > CORRIDOR["z0"]) & (Z < CORRIDOR["z1"]) & (G.sector_of(az) >= 0)
    out = np.full(n_frames, np.inf)
    np.minimum.at(out, d["f"][m], d["r"][m])
    out[~np.isfinite(out)] = np.nan
    return out


def merge_ranges(r, score=None, tol=0.08):
    """Distinct ranges (values closer than tol merged, keeping the first / best-scored)."""
    if len(r) == 0:
        return r
    o = np.argsort(r)
    r = r[o]
    keep = np.r_[True, np.diff(r) > tol]
    return r[keep]


def _assign(Dm):
    from scipy.optimize import linear_sum_assignment
    return linear_sum_assignment(Dm)


def score_rec(d, obs_r, obs_az, floor_r, complete, cor_truth, covered):
    """d: detections of one recording; obs_r/obs_az/floor_r: (n, N_SECT) truth; covered: (n,) bool.
    Returns (frame table dict, detection table dict)."""
    n = obs_r.shape[0]
    red, inn = reduce_sectors(d)
    n_in = np.bincount(d["f"][inn], minlength=n)[:n]
    n_out = np.bincount(d["f"][~inn], minlength=n)[:n]
    tX, tZ = G.plane(obs_r, obs_az)
    starts = np.searchsorted(red["f"], np.arange(n + 1))
    dX, dZ = G.plane(red["r"], red["az_used"])
    fr = {k: np.zeros(n) for k in ["n_truth", "n_det", "gospa", "g_loc", "g_miss", "g_false"]
          + [f"tp{int(c * 100)}" for c in RINGS] + ["match_err"]}
    fr["match_err"][:] = np.nan
    det_match = np.full((len(red["f"]), len(RINGS)), False)
    det_dist = np.full(len(red["f"]), np.nan)
    for i in range(n):
        tm = np.isfinite(obs_r[i])
        T = np.stack([tX[i][tm], tZ[i][tm]], 1)
        a, b = starts[i], starts[i + 1]
        P = np.stack([dX[a:b], dZ[a:b]], 1)
        fr["n_truth"][i], fr["n_det"][i] = len(T), len(P)
        if len(T) and len(P):
            Dm = np.linalg.norm(T[:, None] - P[None], axis=2)
            # GOSPA assignment: cost min(d, c)^p
            ri, ci = _assign(np.minimum(Dm, GOSPA_C) ** GOSPA_P)
            dd = Dm[ri, ci]
            ok = dd < GOSPA_C
            loc = float((dd[ok] ** GOSPA_P).sum())
            k = int(ok.sum())
            fr["g_loc"][i], fr["g_miss"][i], fr["g_false"][i] = loc, len(T) - k, len(P) - k
            fr["gospa"][i] = (loc + GOSPA_C ** GOSPA_P / 2 * (len(T) + len(P) - 2 * k)) ** (1 / GOSPA_P)
            ri, ci = _assign(np.minimum(Dm, 2.0))
            dd = Dm[ri, ci]
            for j, c in enumerate(RINGS):
                fr[f"tp{int(c * 100)}"][i] = (dd <= c).sum()
            if len(dd):
                fr["match_err"][i] = float(np.median(dd[dd <= 0.5])) if (dd <= 0.5).any() else np.nan
            # greedy by score per ring (AP)
            o = np.argsort(-red["score"][a:b])
            for j, c in enumerate(RINGS):
                used = np.zeros(len(T), bool)
                for q in o:
                    dq = np.where(used, np.inf, Dm[:, q])
                    t = int(np.argmin(dq))
                    if dq[t] <= c:
                        used[t] = True
                        det_match[a + q, j] = True
            det_dist[a:b] = Dm.min(0)
        else:
            fr["g_miss"][i], fr["g_false"][i] = len(T), len(P)
            fr["gospa"][i] = (GOSPA_C ** GOSPA_P / 2 * (len(T) + len(P))) ** (1 / GOSPA_P)
    # false-positive causes (ring 0.5 m)
    j50 = RINGS.index(0.50)
    s = red["sector"]
    f = red["f"]
    o_r = obs_r[f, s] if len(f) else np.zeros(0)
    fl = floor_r[f, s] if len(f) else np.zeros(0)
    r = red["r"]
    ghost = np.zeros(len(f), bool)
    for ds in (-1, 0, 1):
        ss = np.clip(s + ds, 0, G.N_SECT - 1)
        ghost |= np.abs(r - 2 * obs_r[f, ss]) < 0.15 if len(f) else ghost
    cause = np.full(len(f), "", object)
    fp = ~det_match[:, j50]
    cause[fp] = "clutter"
    cause[fp & ~complete[f]] = "unlabelled"
    cause[fp & np.isfinite(fl) & (r >= fl - 0.15)] = "floor"
    cause[fp & ghost] = "ghost"
    cause[fp & np.isfinite(o_r) & (r > o_r + 0.5)] = "behind_obstacle"
    # range-only view (no angle needed): distinct truth ranges vs distinct detection ranges, all azimuths
    for c in RINGS:
        fr[f"rtp{int(c * 100)}"] = np.zeros(n)
    fr["n_truth_r"], fr["n_det_r"] = np.zeros(n), np.zeros(n)
    fr["near_truth"], fr["near_det"] = np.nanmin(np.where(np.isfinite(obs_r), obs_r, np.inf), 1), np.full(n, np.inf)
    fr["near_truth"][~np.isfinite(fr["near_truth"])] = np.nan
    s0 = np.searchsorted(d["f"], np.arange(n + 1))
    for i in range(n):
        tr = merge_ranges(obs_r[i][np.isfinite(obs_r[i])])
        dr = merge_ranges(d["r"][s0[i]:s0[i + 1]])
        fr["n_truth_r"][i], fr["n_det_r"][i] = len(tr), len(dr)
        if len(dr):
            fr["near_det"][i] = dr[0]
        if len(tr) and len(dr):
            Dm = np.abs(tr[:, None] - dr[None])
            ri, ci = _assign(np.minimum(Dm, 2.0))
            dd = Dm[ri, ci]
            for c in RINGS:
                fr[f"rtp{int(c * 100)}"][i] = (dd <= c).sum()
    fr["near_det"][~np.isfinite(fr["near_det"])] = np.nan
    cor_det = corridor_nearest(d, n)
    fr.update(complete=complete.astype(bool), n_det_fov_raw=n_in, n_det_out=n_out, cor_truth=cor_truth, cor_det=cor_det, covered=covered)
    dt = dict(f=f, sector=s, r=r, az=red["az_used"], score=red["score"], dist=det_dist, cause=cause,
              **{f"m{int(c * 100)}": det_match[:, j] for j, c in enumerate(RINGS)})
    return fr, dt


def summary(frames, dets=None, fps=None) -> dict:
    """Aggregate metrics over covered frames (dict of scalars). frames: DataFrame of score_rec frame rows."""
    F = frames[frames.covered.astype(bool)]
    out = {"frames": int(len(F))}
    if not len(F):
        return out
    # recall over every covered frame; precision / false positives / GOSPA only where the truth is complete
    # (static takes with tags only: anything that is not a tag is unlabelled, not 'false')
    Fc = F[F.complete.astype(bool)] if "complete" in F else F
    out["frames_complete"] = int(len(Fc))
    nt = F.n_truth.sum()
    ntc, ndc = Fc.n_truth.sum(), Fc.n_det.sum()
    for c in RINGS:
        tp = F[f"tp{int(c * 100)}"].sum()
        tpc = Fc[f"tp{int(c * 100)}"].sum()
        out[f"recall_{int(c * 100)}"] = tp / nt if nt else np.nan
        out[f"precision_{int(c * 100)}"] = tpc / ndc if ndc else np.nan
        out[f"f1_{int(c * 100)}"] = 2 * tpc / (ntc + ndc) if ntc + ndc else np.nan
        out[f"fp_per_frame_{int(c * 100)}"] = (ndc - tpc) / len(Fc) if len(Fc) else np.nan
    out["gospa"] = Fc.gospa.mean() if len(Fc) else np.nan
    out["gospa_loc"] = np.sqrt(F.g_loc.sum() / max(1, (F.n_truth - F.g_miss).sum()))
    out["miss_per_frame"] = F.g_miss.mean()
    out["false_per_frame"] = F.g_false.mean()
    out["match_err_m"] = F.match_err.median()
    out["det_out_fov_per_frame"] = F.n_det_out.mean()
    ct, cd = F.cor_truth.to_numpy(), F.cor_det.to_numpy()
    both = np.isfinite(ct) & np.isfinite(cd)
    out["cor_frames_truth"] = int(np.isfinite(ct).sum())
    out["cor_pd"] = float((both & (np.abs(cd - ct) <= 0.25)).sum() / max(1, np.isfinite(ct).sum()))
    out["cor_err_m"] = float(np.median(np.abs(cd - ct)[both])) if both.any() else np.nan
    out["cor_false_alarm_rate"] = float((np.isfinite(cd) & ~np.isfinite(ct)).sum() / max(1, (~np.isfinite(ct)).sum()))
    out["cor_early_rate"] = float((both & (cd < ct - 0.25)).sum() / max(1, len(F)))
    ntr, ndr = F.n_truth_r.sum(), F.n_det_r.sum()
    for c in RINGS:
        tp = F[f"rtp{int(c * 100)}"].sum()
        out[f"r_recall_{int(c * 100)}"] = tp / ntr if ntr else np.nan
        out[f"r_precision_{int(c * 100)}"] = tp / ndr if ndr else np.nan
        out[f"r_f1_{int(c * 100)}"] = 2 * tp / (ntr + ndr) if ntr + ndr else np.nan
    nt_, nd_ = F.near_truth.to_numpy(), F.near_det.to_numpy()
    b2 = np.isfinite(nt_) & np.isfinite(nd_)
    out["near_pd"] = float((b2 & (np.abs(nd_ - nt_) <= 0.25)).sum() / max(1, np.isfinite(nt_).sum()))
    out["near_early_rate"] = float((b2 & (nd_ < nt_ - 0.25)).sum() / max(1, len(F)))
    out["near_err_m"] = float(np.median(np.abs(nd_ - nt_)[b2])) if b2.any() else np.nan
    if fps:
        on = np.isfinite(cd) & ~np.isfinite(ct)
        out["cor_false_alarms_per_min"] = float((np.diff(on.astype(int)) == 1).sum() / (len(F) / fps / 60))
    if dets is not None and len(dets):
        D = dets
        for c in RINGS:
            out[f"ap_{int(c * 100)}"] = ap(D.score.to_numpy(), D[f"m{int(c * 100)}"].to_numpy(), nt)
        fp = D[D.cause != ""]
        for k, v in fp.cause.value_counts().items():
            out[f"fp_{k}_per_frame"] = v / len(F)
    return out


def ap(score, match, n_truth):
    """All-point interpolated average precision (pooled detections, greedy matching already done)."""
    if n_truth == 0 or len(score) == 0:
        return np.nan
    o = np.argsort(-score, kind="stable")
    m = match[o].astype(float)
    tp = np.cumsum(m)
    prec = tp / np.arange(1, len(m) + 1)
    rec = tp / n_truth
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r_prev = np.r_[0, rec[:-1]]
    return float(((rec - r_prev) * prec).sum())
