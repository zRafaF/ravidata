"""GT4-GT6: obstacle ground truth at every radar frame, in the radar frame.

Representation (same for every take): per radar frame and 4 deg azimuth sector inside the camera FOV, the nearest
obstacle point (slant range r, az) and the nearest floor point; plus per frame the nearest obstacle in the
walking corridor (|X| < 0.5 m, 0.3 < Z < 3 m, inside the FOV).

* carried takes: keyframe depth of the feed-forward map (default Pi3X), metric through the AprilTags, moved to the
  radar frame time with the fused trajectory when there is one (else the nearest keyframe within max_dt);
  'up' from the phone gravity; floor = densest level below the camera; obstacle = 0.12 .. 2.0 m above the floor.
  Tags on people / chairs / brooms seen at that moment are added as obstacle points.
* static takes: the AprilTags on the objects (median over the recording; floor tags are floor), plus the
  static-depth surfaces (derived/static-depth-v1) when that step has been run -> 'truth_complete'.

Tables: truth_sectors (frame x sector), truth_frames (per frame), map_scale (tag-based metric scale per window)."""

from __future__ import annotations

import numpy as np

from . import common as C
from . import geometry as G
from . import imu

H_OBS_MIN, H_OBS_MAX, H_FLOOR_TOL = 0.12, 2.0, 0.06
CORRIDOR = dict(half=0.5, z0=0.3, z1=3.0)


def t_rc():
    e = C.load_json("extrinsic", None)
    return np.array(e["t_rc"], float) if e else np.zeros(3)


def floor_plane(P, up, h, hf, iters=200, tol=0.03, max_tilt_deg=15.0, seed=0):
    """RANSAC plane through the points near the floor level whose normal is within max_tilt of 'up'.
    Returns (n, d) with height above floor = P @ n + d, or None."""
    cand = np.flatnonzero(np.abs(h - hf) < 0.15)
    if len(cand) < 50:
        return None
    rng = np.random.default_rng(seed)
    Q = P[cand]
    best, best_n = None, 0
    cmax = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        if n @ up < 0:
            n = -n
        if n @ up < cmax:
            continue
        d = -n @ a
        k = int((np.abs(Q @ n + d) < tol).sum())
        if k > best_n:
            best, best_n = (n, d), k
    if best is None:
        return None
    n, d = best
    inl = Q[np.abs(Q @ n + d) < tol]                       # refit on the inliers (least squares)
    c0 = inl.mean(0)
    _, _, vt = np.linalg.svd(inl - c0, full_matrices=False)
    n = vt[2] if vt[2] @ up > 0 else -vt[2]
    return n, -n @ c0


def classify(P_cam, up, h_floor=None):
    """Height above the floor (a RANSAC floor plane near the densest low level, normal close to the gravity 'up'),
    floor level below the camera, obstacle / floor masks."""
    h = P_cam @ up
    hf = G.floor_height(h) if h_floor is None else h_floor
    if not np.isfinite(hf):
        return h, hf, np.zeros(len(h), bool), np.zeros(len(h), bool)
    pl = floor_plane(P_cam, up, h, hf)
    if pl is not None:
        hh = P_cam @ pl[0] + pl[1]                          # height above the floor plane
        hf = -pl[1]                                         # camera height above it (negative = below the camera)
        hf = -abs(hf)
    else:
        hh = h - hf
    obs = (hh > H_OBS_MIN) & (hh < H_OBS_MAX)
    flo = np.abs(hh) < H_FLOOR_TOL
    return h, hf, obs, flo


def summarise(P_rad, obs, flo, extra_obs=None):
    """Sector table (N_SECT rows) + corridor nearest for one frame. extra_obs: (k, 3) radar-frame points."""
    r, az, el = G.polar(P_rad)
    if extra_obs is not None and len(extra_obs):
        re_, ae_, ee_ = G.polar(extra_obs)
        r, az, el = np.r_[r, re_], np.r_[az, ae_], np.r_[el, ee_]
        obs = np.r_[obs, np.ones(len(re_), bool)]
        flo = np.r_[flo, np.zeros(len(re_), bool)]
    inr = (r >= G.R_MIN) & (r <= G.R_MAX + 0.3)
    o_r, o_i = G.nearest_per_sector(r, az, obs & inr)
    f_r, _ = G.nearest_per_sector(r, az, flo & inr)
    sec = G.sector_of(az)
    cnt = np.bincount(sec[sec >= 0], minlength=G.N_SECT)
    X, Z = G.plane(r, az)
    cor = obs & (np.abs(X) < CORRIDOR["half"]) & (Z > CORRIDOR["z0"]) & (Z < CORRIDOR["z1"]) & (sec >= 0)
    cor_r = float(r[cor].min()) if cor.any() else np.nan
    if len(r):
        o_az = np.where(o_i >= 0, az[np.maximum(o_i, 0)], np.nan)
        o_el = np.where(o_i >= 0, el[np.maximum(o_i, 0)], np.nan)
    else:
        o_az = o_el = np.full(G.N_SECT, np.nan)
    return o_r, o_az, o_el, f_r, cnt, cor_r


def _gravity_up(take, t):
    g = imu.at(take, t, "Gravity", smooth_s=0.2)
    return None if g is None else G.up_from_gravity(g)


def _tag_points(tags, take, t0, t1, kinds_excl=("floor_tag",)):
    if tags is None or len(tags) == 0:
        return None
    q = tags[(tags["take"] == take) & (tags.t_host >= t0) & (tags.t_host <= t1) & (tags.reproj_px < 2.0)
             & (tags.view_deg < 70)]
    return q[~q.kind.isin(kinds_excl)], q[q.kind.isin(kinds_excl)]


SD_PREFERENCE = ("depthpro", "da2l", "da2s")      # static-depth models, best first


def static_depth_dir(take):
    import os
    root = C.DATA / "derived" / "static-depth-v1"
    want = os.environ.get("RAVIDATA_STATIC_DEPTH")      # force one model (short name)
    for m in ([want] if want else SD_PREFERENCE):
        if (root / m / take / "summary.json").exists():      # a finished run, not an empty folder
            return root / m / take
    return None


def static_depth_points(take, part):
    d = static_depth_dir(take)
    fs = sorted(d.glob(f"{part}_*.npz")) if d is not None else []
    if not fs:
        return None
    import cv2
    deps, K, valid = [], None, None
    for f in fs:
        z = np.load(f, allow_pickle=True)
        if not np.isfinite(float(z["a"])):
            continue
        dep = z["depth"].astype(np.float32)
        deps.append(dep)
        K = z["K"]
        jpg = f.with_suffix(".jpg")                       # the undistortion border is black: no depth there
        if jpg.exists():
            g = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
            if g is not None and g.shape == dep.shape:
                v = cv2.erode((g > 12).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
                valid = v if valid is None else (valid & v)
    if not deps:
        return None
    dm = np.nanmedian(np.stack(deps), 0)
    H = dm.shape[0]
    b = int(0.03 * H)                                     # and a 3 % margin on every side
    vv, uu = np.mgrid[b:H - b:2, b:H - b:2]
    z = dm[vv, uu]
    ok = np.isfinite(z) & (z > 0)
    if valid is not None:
        ok &= valid[vv, uu]
    return np.stack([(uu[ok] - K[0, 2]) / K[0, 0] * z[ok], (vv[ok] - K[1, 2]) / K[1, 1] * z[ok], z[ok]], 1)


def run_static(take, tags, rows_s, rows_f, trc):
    for rec in C.recordings(take):
        t = C.radar_times(rec, C.default_max_frames())
        if len(t) == 0:
            continue
        up = _gravity_up(take, np.median(t))
        up = up[0] if up is not None else np.array([0.0, -1.0, 0.0])
        sel = _tag_points(tags, take, t[0], t[-1])
        pts, src = [], "tags"
        hf = np.nan
        if sel is not None:
            ob, fl = sel
            for _, g in ob.groupby("id"):
                pts.append(np.median(g[["tx", "ty", "tz"]].to_numpy(), 0))
            if len(fl):
                hf = float(np.median(fl[["tx", "ty", "tz"]].to_numpy() @ up))
        P_obs = np.array(pts).reshape(-1, 3)
        sd = static_depth_points(take, rec.part)
        if sd is not None:
            h, hf2, obs, flo = classify(sd, up, hf if np.isfinite(hf) else None)
            hf = hf2
            P = np.r_[sd, P_obs]
            obs = np.r_[obs, np.ones(len(P_obs), bool)]
            flo = np.r_[flo, np.zeros(len(P_obs), bool)]
            src = f"static-depth:{static_depth_dir(take).parent.name}+tags"
        else:
            P, obs, flo = P_obs, np.ones(len(P_obs), bool), np.zeros(len(P_obs), bool)
        o_r, o_az, o_el, f_r, cnt, cor_r = summarise(G.to_radar(P, trc), obs, flo)
        if sd is None:
            cnt = np.full(G.N_SECT, -1)                      # coverage unknown with tags only
        for f, tt in enumerate(t):
            _emit(rows_s, rows_f, rec, f, tt, o_r, o_az, o_el, f_r, cnt, cor_r, src, 0.0, hf, sd is not None)


def _emit(rows_s, rows_f, rec, f, t, o_r, o_az, o_el, f_r, cnt, cor_r, src, dt, hf, complete):
    sc = G.sector_centres()
    for s in range(G.N_SECT):
        rows_s.append((rec.key, f, s, sc[s], o_r[s], o_az[s], o_el[s], f_r[s], int(cnt[s])))
    rows_f.append(dict(rec=rec.key, take=rec.take, config=rec.config, part=rec.part, f=f, t=t, source=src,
                       dt_kf=dt, covered=bool(np.isfinite(dt)), complete=complete,
                       n_obj=int(np.isfinite(o_r).sum()), corridor_r=cor_r, floor_h=hf))


def run_carried(take, tags, rows_s, rows_f, rows_scale, trc, model="pi3x", max_dt=0.25):
    from .maps import TakeMaps, interp_poses
    tm = TakeMaps(take, model)
    rows_scale += tm.fit_scale(tags)
    tr = tm.trajectory()
    if tr is not None:
        C.log(f"  {take}: fused trajectory, metric factor {tr[2]:.3f}")
    kf = tm.kf
    t_kf = tm.t[kf]
    cache = {}

    def kf_points(i):
        if i not in cache:
            if len(cache) > 6:
                cache.pop(next(iter(cache)))
            X, k = tm.points(int(i))
            X = X * tm.metric(k)
            up = _gravity_up(take, tm.t[i])
            up = up[0] if up is not None else np.array([0.0, -1.0, 0.0])
            cache[i] = (X, up)
        return cache[i]

    hf_hist = []
    for rec in C.recordings(take):
        t = C.radar_times(rec, C.default_max_frames())
        Tt = interp_poses(tr[0], tr[1], t) if tr is not None else None
        sel = _tag_points(tags, take, t[0] - 1, t[-1] + 1)
        tob = sel[0] if sel is not None else None
        for f, tt in enumerate(C.bar(t, desc=f"truth {rec.key}", unit="frame")):
            j = int(np.argmin(np.abs(t_kf - tt)))
            dt = float(t_kf[j] - tt)
            i = int(kf[j])
            if abs(dt) > max_dt and (Tt is None or not np.isfinite(Tt[f]).all()):
                _emit(rows_s, rows_f, rec, f, tt, *[np.full(G.N_SECT, np.nan)] * 4, np.zeros(G.N_SECT), np.nan,
                      "none", np.nan, np.nan, False)
                continue
            X, up_k = kf_points(i)
            if Tt is not None and np.isfinite(Tt[f]).all():
                Tk = tm.pose_metric(i)
                if Tk is not None:
                    Rk, pk = Tk[:3, :3], Tk[:3, 3]
                    R, p = Tt[f][:3, :3], Tt[f][:3, 3]
                    X = (X @ Rk.T + pk - p) @ R
                    up_k = R.T @ (Rk @ up_k)
            _, hf, obs, flo = classify(X, up_k)
            hf_hist.append(hf)
            extra = None
            if tob is not None and len(tob):
                q = tob[np.abs(tob.t_host - tt) < 0.1]
                if len(q):
                    extra = G.to_radar(q.groupby("id")[["tx", "ty", "tz"]].median().to_numpy(), trc)
            o_r, o_az, o_el, f_r, cnt, cor_r = summarise(G.to_radar(X, trc), obs, flo, extra)
            _emit(rows_s, rows_f, rec, f, tt, o_r, o_az, o_el, f_r, cnt, cor_r, f"map:{model}", dt, hf, True)


def run(takes=None, model="pi3x", max_dt=0.25):
    import pandas as pd
    tags = C.load_table("tags", required=False)
    trc = t_rc()
    rows_s, rows_f, rows_scale = [], [], []
    for take in C.available_takes(takes):
        C.log(f"truth {take}")
        if take in C.STATIC:
            run_static(take, tags, rows_s, rows_f, trc)
        else:
            if not (C.DATA / "derived" / "maps-v1" / model / take).exists():
                C.log(f"  no maps for {take} ({model}): skipped")
                continue
            run_carried(take, tags, rows_s, rows_f, rows_scale, trc, model, max_dt)
    S = pd.DataFrame(rows_s, columns=["rec", "f", "sector", "az_c", "obs_r", "obs_az", "obs_el", "floor_r", "n_pts"])
    Fm = pd.DataFrame(rows_f)
    done = C.available_takes(takes)
    S["take"] = S.rec.str.split("/").str[0]
    C.save_table(S, "truth_sectors", takes=done)
    C.save_table(Fm, "truth_frames", takes=done)
    C.save_table(pd.DataFrame(rows_scale), "map_scale", takes=done)
    if len(Fm):
        g = Fm.groupby("take").agg(frames=("f", "size"), covered=("covered", "mean"), n_obj=("n_obj", "mean"),
                                   corridor=("corridor_r", lambda x: np.isfinite(x).mean()))
        C.log("\n" + g.round(2).to_string())
    return S, Fm
