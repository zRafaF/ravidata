"""F08: the camera-built 3D map (one self-consistent map window of indoor_course) with the walk, the AprilTags and
the radar returns placed in it. Exports text-free rasters (oblique orthographic view and bird's-eye view of the
coloured point cloud) and JSON sidecars with every overlay already projected to the view planes (metres).

Radar returns: per subsampled radar frame, the strongest range bins (local maxima, dB over the frame's median),
each at its interferometric azimuth and ZERO elevation in the radar frame (the radar is taken to be at the camera),
moved to the map frame by the camera pose interpolated at the radar frame's time (corrected sync via TakeMaps).

run: PYTHONPATH=src TCC_FIGURES=<tcc1>/figures/data python3 -m analysis.paper.f08_mapa3d"""
from __future__ import annotations

import numpy as np

from .. import common as C, radar as R
from .. import truth as TR
from .. import geometry as G
from ..maps import TakeMaps, interp_poses, MAPS
from . import common as P

TAKE, REC_PREFIX, WINDOW = "indoor_course", "01_", 1
STRIDE = 2                  # pixel stride of the keyframe depth maps
H_CUT = 1.95                # points higher than this above the floor are not drawn (ceiling would hide the room)
VIEW_AZ, VIEW_EL = 205.0, 38.0        # oblique view: viewer direction (deg) in the map frame (x = walk direction)
PX_PER_M = 150.0
FRAME_STEP = 3              # every 3rd radar frame
N_PEAKS, THR_DB = 3, 12.0   # strongest local maxima per frame, at least THR_DB over the frame's median
HIT_M = 0.15                # a return 'lands on the map' when a map point lies within this distance


def window_points(tm, k, stride):
    """Points of every keyframe of window k from the window's OWN depth (window units, window frame), with the
    keyframe's RGB. Same confidence / border filter as TakeMaps.points."""
    import cv2
    w = tm.window(k)
    K = tm.K
    H = w["depth"].shape[1]
    b = int(0.03 * H)
    vv, uu = np.mgrid[b:H - b:stride, b:H - b:stride]
    pts, cols = [], []
    for j, i in enumerate(w["idx"]):
        z = w["depth"][j][vv, uu].astype(np.float64)
        cf = w["conf"][j][vv, uu]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() > 10:
            ok &= cf >= np.percentile(cf[ok], 30.0)
        # depth edges smear between foreground and background: drop steep depth steps
        gy, gx = np.gradient(np.where(np.isfinite(z), z, 0.0))
        ok &= np.hypot(gx, gy) < 0.04 * np.maximum(z, 1e-6) * stride
        u, v, zz = uu[ok], vv[ok], z[ok]
        X = np.stack([(u - K[0, 2]) / K[0, 0] * zz, (v - K[1, 2]) / K[1, 1] * zz, zz], 1)
        Tw = w["c2w"][j]
        pts.append(X @ Tw[:3, :3].T + Tw[:3, 3])
        im = cv2.imread(str(MAPS / "frames" / tm.take / "rgb" / f"{int(i):06d}.jpg"))
        if im is None:
            cols.append(np.full((len(X), 3), 128, np.uint8))
        else:
            if im.shape[0] != H:
                im = cv2.resize(im, (H, H), interpolation=cv2.INTER_AREA)
            cols.append(im[v, u, ::-1])
    return np.concatenate(pts), np.concatenate(cols), w


def map_frame(tm, w, Pm, s_w):
    """Rotation / origin of the drawing frame: z = floor normal (gravity + RANSAC floor plane, as truth.window_cloud),
    x = horizontal direction of the walk, origin on the floor below the first camera. Returns (A, o) with
    q = (p - o) @ A.T in metres."""
    up = TR._gravity_up(tm.take, tm.t[w["idx"]])
    u = np.einsum("nij,nj->i", w["c2w"][:, :3, :3], up) if up is not None else -w["c2w"][0, :3, 1]
    u /= np.linalg.norm(u)
    h = Pm @ u
    pl = TR.floor_plane(Pm, u, h, G.floor_height(h - np.median(h), lo=-3, hi=1) + np.median(h))
    n, d = pl if pl is not None else (u, -np.percentile(h, 5))
    cam = w["c2w"][:, :3, 3] * s_w
    o = cam[0] - (cam[0] @ n + d) * n
    x = cam[-1] - cam[0]
    x = x - (x @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n]), o


def view_basis(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    v = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])      # towards the viewer
    r = np.array([-np.sin(a), np.cos(a), 0.0])
    return np.stack([r, np.cross(v, r), v])                                       # screen right, screen up, depth


def splat(S, col, xlim, ylim, size=2):
    """Orthographic z-buffer splat. S: (n, 3) screen right / up / depth (larger = nearer). White background."""
    W = int(round((xlim[1] - xlim[0]) * PX_PER_M)); H = int(round((ylim[1] - ylim[0]) * PX_PER_M))
    x = ((S[:, 0] - xlim[0]) * PX_PER_M).astype(int)
    y = ((ylim[1] - S[:, 1]) * PX_PER_M).astype(int)
    zb = np.full((H, W), -np.inf)
    img = np.full((H, W, 3), 255, np.uint8)
    for dy in range(size):
        for dx in range(size):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            flat = yy[ok] * W + xx[ok]
            o = np.argsort(S[ok, 2], kind="stable")                # far first, so the nearest is written last
            f, z, c = flat[o], S[ok, 2][o], col[ok][o]
            best = np.full(H * W, -np.inf); best[f] = z            # last write = nearest
            win = z >= np.maximum(best[f], zb.ravel()[f])
            zb.ravel()[f[win]] = z[win]
            img.reshape(-1, 3)[f[win]] = c[win]
    return img


def radar_returns(rd, t0, t1):
    """(frame time, range m, azimuth deg, dB over median) of the strongest local maxima of subsampled frames."""
    sel = np.flatnonzero((rd.t >= t0) & (rd.t <= t1))[::FRAME_STEP]
    x = rd.cube[sel].astype(np.complex128)
    Pw = (np.abs(x) ** 2).sum(1).mean(1)
    L = 10 * np.log10(Pw + 1e-15)
    L = L - np.median(L[:, 8:110], 1, keepdims=True)
    a_, b_ = R.PAIRS["az"]
    S = (x[:, :, b_] * np.conj(x[:, :, a_])).sum(1)
    S[:, 1:-1] = S[:, :-2] + S[:, 1:-1] + S[:, 2:]
    cal = R.angle_calib()["az"]
    az = R.phase_to_angle(np.angle(S), cal["sign"], cal["offset_rad"])
    r = rd.range_m - R.range_bias()
    out = []
    for f in range(len(sel)):
        l = L[f]
        pk = np.flatnonzero((l[1:-1] >= l[:-2]) & (l[1:-1] > l[2:]) & (l[1:-1] > THR_DB)) + 1
        pk = pk[(r[pk] >= G.R_MIN) & (r[pk] <= G.R_MAX)]
        for b in pk[np.argsort(-l[pk])][:N_PEAKS]:
            out.append((rd.t[sel[f]], r[b], az[f, b], l[b]))
    return np.array(out), len(sel)


def run():
    from scipy.spatial import cKDTree
    rec = [r for r in C.recordings(TAKE) if r.file.stem.startswith(REC_PREFIX)][0]
    rd = C.load_radar(rec)
    tags = C.load_tags()
    tm = TakeMaps(TAKE)
    tm.fit_scale(tags)
    s_w = tm.metric(WINDOW)
    Pw, col, w = window_points(tm, WINDOW, STRIDE)
    A, o = map_frame(tm, w, Pw[::7] * s_w, s_w)
    Q = (Pw * s_w - o) @ A.T                                   # map frame, metres: x walk, z up
    t_kf = tm.t[w["idx"]]
    c2w = w["c2w"].copy(); c2w[:, :3, 3] *= s_w
    t0, t1 = max(t_kf.min(), rd.t[0]), min(t_kf.max(), rd.t[-1])
    to_map = lambda p: (p - o) @ A.T

    # trajectory (camera centres of the keyframes) and heading (optical axis) at a few poses
    traj = to_map(c2w[:, :3, 3])
    head = c2w[:, :3, 2] @ A.T

    # radar returns in the map frame
    det, n_frames = radar_returns(rd, t0, t1)
    Tt = interp_poses(t_kf, c2w, det[:, 0])
    a = np.radians(det[:, 2])
    p_rad = np.stack([det[:, 1] * np.sin(a), np.zeros(len(det)), det[:, 1] * np.cos(a)], 1) + TR.t_rc()
    p_w = np.einsum("nij,nj->ni", Tt[:, :3, :3], p_rad) + Tt[:, :3, 3]
    ret = to_map(p_w)
    org = to_map(Tt[:, :3, 3])
    tree = cKDTree(Q[::3])
    dist = tree.query(ret)[0]
    hit = dist <= HIT_M
    # is the measured range explained by SOME map surface in the beam (any elevation)? nearest map point to the
    # sphere of that range, within +-10 deg of the measured azimuth and +-45 deg of elevation
    Qs = Q[::9]
    hs = Qs[:, 2]
    expl = np.zeros(len(det), bool); expl_floor = np.zeros(len(det), bool)
    Ainv_o = A, o
    for tt in np.unique(det[:, 0]):
        m = det[:, 0] == tt
        T1 = Tt[np.flatnonzero(m)[0]]
        Xc = ((Qs @ A + o) - T1[:3, 3]) @ T1[:3, :3]            # map points in the camera / radar frame
        rr, aa, ee = G.polar(Xc)
        for j in np.flatnonzero(m):
            c = (np.abs(aa - det[j, 2]) < 10) & (np.abs(rr - det[j, 1]) < 0.08)
            expl[j] = (c & (np.abs(ee) < 45)).sum() >= 5
            cf = c & (hs < 0.08) if True else c
            expl_floor[j] = (not hit[j]) and (c & (ee < -30) & (hs < 0.08)).sum() >= 5 and not (c & (np.abs(ee) < 30)).sum() >= 5

    # AprilTags seen in the stretch (camera frame -> map frame with the pose at the sighting time), median per id
    tg = tags[(tags["take"] == TAKE) & (tags.t_host >= t0) & (tags.t_host <= t1) & (tags.reproj_px < 2.0)
              & tags.id.isin(list(C.TAG_KIND))]
    tag_rows = []
    if len(tg):
        Tg = interp_poses(t_kf, c2w, tg.t_host.to_numpy())
        pg = to_map(np.einsum("nij,nj->ni", Tg[:, :3, :3], tg[["tx", "ty", "tz"]].to_numpy(float)) + Tg[:, :3, 3])
        for i in sorted(tg.id.unique()):
            m = (tg.id == i).to_numpy()
            if m.sum() >= 3:
                tag_rows.append(dict(id=int(i), kind=C.TAG_KIND[int(i)], n=int(m.sum()), xyz=np.median(pg[m], 0),
                                     spread_m=float(np.median(np.linalg.norm(pg[m] - np.median(pg[m], 0), axis=1)))))

    keep = (Q[:, 2] < H_CUT) & (Q[:, 2] > -0.3)
    Qk, ck = Q[keep], col[keep]
    stats = dict(n_frames=int(n_frames), n_returns=int(len(det)), hit_m=HIT_M, frac_hit=float(hit.mean()),
                 frac_range_explained=float(expl.mean()), frac_floor_below=float(expl_floor.mean()),
                 median_dist_m=float(np.median(dist)), thr_db=THR_DB, n_peaks=N_PEAKS, frame_step=FRAME_STEP,
                 t0_s=float(t0 - rd.t[0]), t1_s=float(t1 - rd.t[0]), duration_s=float(t1 - t0),
                 path_m=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()), n_points=int(len(Q)),
                 n_keyframes=int(len(t_kf)), metric_scale=float(s_w), h_cut_m=H_CUT,
                 camera_height_m=float(np.median(traj[:, 2])), window=WINDOW, rec=rec.key)
    C.log(f"  f08: {stats}")

    for name, V, size in (("f08_iso", view_basis(VIEW_AZ, VIEW_EL), 2),
                          ("f08_topo", np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]), 2)):
        S = Qk @ V.T
        pr = lambda p: (np.atleast_2d(p) @ V.T)[:, :2]
        allp = np.concatenate([S[:, :2], pr(ret), pr(traj)])
        lo, hi = np.percentile(S[:, :2], [0.3, 99.7], axis=0)
        lo = np.minimum(lo, allp[len(S):].min(0)) - 0.15; hi = np.maximum(hi, allp[len(S):].max(0)) + 0.15
        xlim, ylim = (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))
        img = splat(S, ck, xlim, ylim, size)
        P.save_png(name, img)
        order = np.argsort(det[:, 0])
        P.save_json(name, dict(
            units="m (plano da vista, projeção ortográfica)", xlim=xlim, ylim=ylim, px=[img.shape[1], img.shape[0]],
            view=dict(az_deg=VIEW_AZ, el_deg=VIEW_EL) if name == "f08_iso" else "top",
            axes=dict(origin=pr([0, 0, 0])[0], x=pr([1, 0, 0])[0], y=pr([0, 1, 0])[0], z=pr([0, 0, 1])[0]),
            traj=np.round(pr(traj), 3), traj_floor=np.round(pr(traj * [1, 1, 0]), 3), traj_t=np.round(t_kf - rd.t[0], 2),
            head=np.round(pr(traj + 0.4 * head), 3),
            returns=[dict(t=round(float(det[j, 0] - rd.t[0]), 2), p=np.round(pr(ret[j])[0], 3),
                          o=np.round(pr(org[j])[0], 3), db=round(float(det[j, 3]), 1), hit=bool(hit[j]),
                          expl=bool(expl[j]), floor=bool(expl_floor[j]), d=round(float(dist[j]), 3)) for j in order],
            tags=[dict(id=g["id"], kind=g["kind"], n=g["n"], p=np.round(pr(g["xyz"])[0], 3),
                       h=round(float(g["xyz"][2]), 2)) for g in tag_rows],
            stats=stats))


if __name__ == "__main__":
    run()
