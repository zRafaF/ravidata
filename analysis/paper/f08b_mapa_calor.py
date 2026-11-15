"""F08b: the camera-built 3D map of F08 with the radar evidence laid over it as a HEAT MAP instead of dots.

One self-consistent Pi3X map window of indoor_course inside the 2D (range-Doppler) recording 03. Exports
  f08b_iso.png / .json    oblique view of the coloured cloud, walk projected to the view plane (no radar);
  f08b_topo.png / .json   bird's-eye view (lightened, so the heat reads over it), walk, tags, statistics;
  f08b_calor_bin.png      evidence map (i): one azimuth per range bin (power summed over Doppler);
  f08b_calor_rd.png       evidence map (ii): one azimuth per range-Doppler cell (grid_doppler.detections);
both RGBA, transparent where there is no evidence, same extent as f08b_topo.png.

Evidence map: horizontal grid of CELL m in the map frame. Per radar frame, the evidence in polar form (range,
azimuth, ZERO elevation in the radar frame; the radar is taken to be at the camera) is spread with a Gaussian
(SIG_R, SIG_AZ), carried to the map frame by the camera pose interpolated at the frame's time and added to the grid.
Weight = dB over the frame's median (median over the range bins of the Doppler-summed power, the same reference
for both maps), clip((L - 3) / 27, 0, 1); returns measured beyond +-FOV_DEG are not placed. The sum is divided by the number of frames that could
have seen each cell (range R_MIN..R_MAX, |azimuth| <= FOV_DEG at that frame), so a cell holds a MEAN evidence.
Control (chance level): the same, with each frame's evidence turned by a random angle about the vertical (N_CTRL
draws); a second control mirrors the azimuth about the boresight. Statistic: share of the evidence within NEAR_M
(horizontally) of an obstacle surface of the map (points in the height band around the radar, in the floor plan).

run: PYTHONPATH=src TCC_FIGURES=<tcc1>/figures/data python3 -m analysis.paper.f08b_mapa_calor"""
from __future__ import annotations

import json

import numpy as np

from .. import common as C, radar as R
from .. import geometry as G
from .. import grid_doppler as GD
from .. import pipeline as PL
from ..maps import TakeMaps, interp_poses
from . import common as P
from .f08_mapa3d import window_points, map_frame, view_basis          # same map machinery as F08 (not modified)

TAKE, REC_PREFIX, WINDOW = "indoor_course", "03_", 17
STRIDE = 2
H_CUT = 1.95                          # points higher than this above the floor are not drawn
VIEW_AZ, VIEW_EL = 205.0, 38.0
PX_PER_M = 160.0                      # 8 px per evidence cell
CELL = 0.05
SIG_R, SIG_AZ = 0.05, 6.0             # spread of one contribution (m, deg)
FOV_DEG = 60.0
DB_LO, DB_SPAN = 3.0, 27.0
N_MIN = 10                            # frames: a cell seen in fewer frames is averaged as if seen in N_MIN
OBST_H_MIN, OBST_BAND = 0.12, 0.6     # obstacle surface: > 0.12 m above the floor, within +-0.6 m of the radar height
OBST_MIN_PTS = 6                      # map points (stride 2) per 5 cm cell
NEAR_M = (0.15, 0.30)
N_CTRL, SEED = 8, 7
LIGHTEN = 0.42                        # base raster of the heat panels: blend towards white
V_PCT, GAMMA, A_MAX, V_FLOOR = 99.5, 0.75, 0.88, 0.06
EXT_LEVEL, EXT_MAX = 0.45, 0.9        # the top view is widened (at most EXT_MAX m per side) to hold evidence above this level
STOPS = np.array([[255, 214, 92], [246, 140, 36], [214, 56, 30], [128, 8, 44]], float)    # low -> high evidence

RG = np.arange(G.R_MIN, G.R_MAX + 1e-9, 0.02)
AG = np.arange(-FOV_DEG, FOV_DEG + 1e-9, 1.0)


def splat(S, col, xlim, ylim, size=2, px=PX_PER_M):
    """Orthographic z-buffer splat (as F08, pixel density as a parameter). White background."""
    W = int(round((xlim[1] - xlim[0]) * px)); H = int(round((ylim[1] - ylim[0]) * px))
    x = ((S[:, 0] - xlim[0]) * px).astype(int)
    y = ((ylim[1] - S[:, 1]) * px).astype(int)
    zb = np.full(H * W, -np.inf)
    img = np.full((H * W, 3), 255, np.uint8)
    for dy in range(size):
        for dx in range(size):
            xx, yy = x + dx, y + dy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            flat = yy[ok] * W + xx[ok]
            o = np.argsort(S[ok, 2], kind="stable")
            f, z, c = flat[o], S[ok, 2][o], col[ok][o]
            best = np.full(H * W, -np.inf); best[f] = z
            win = z >= np.maximum(best[f], zb[f])
            zb[f[win]] = z[win]
            img[f[win]] = c[win]
    return img.reshape(H, W, 3)


def polar_image(r, az, w):
    """Contributions (range m, azimuth deg, weight) -> smooth polar image on (RG, AG)."""
    if not len(r):
        return np.zeros((len(RG), len(AG)))
    gr = np.exp(-0.5 * ((RG[:, None] - r[None]) / SIG_R) ** 2)
    ga = np.exp(-0.5 * ((AG[None] - az[:, None]) / SIG_AZ) ** 2) * w[:, None]
    return gr @ ga


class Acc:
    """Evidence grid in top-view coordinates (sx to the right, sy up, metres); row 0 = lowest sy."""

    def __init__(self, x0, y0, nx, ny, n_layers):
        self.x0, self.y0, self.nx, self.ny = x0, y0, nx, ny
        self.acc = [np.zeros((n, ny, nx)) for n in n_layers]
        self.cnt = np.zeros((ny, nx))

    def add(self, c2, ex2, ez2, images):
        """c2: radar position; ex2 / ez2: horizontal projection of the radar's x (right) / z (boresight) axes.
        A cell at c2 + u ex2 + w ez2 holds the radar-plane point (u, 0, w): range hypot(u, w), azimuth atan2(u, w)."""
        from scipy.ndimage import map_coordinates
        reach = G.R_MAX + CELL
        i0 = max(int(np.floor((c2[0] - reach - self.x0) / CELL)), 0); i1 = min(int(np.ceil((c2[0] + reach - self.x0) / CELL)), self.nx)
        j0 = max(int(np.floor((c2[1] - reach - self.y0) / CELL)), 0); j1 = min(int(np.ceil((c2[1] + reach - self.y0) / CELL)), self.ny)
        X = self.x0 + (np.arange(i0, i1) + 0.5) * CELL - c2[0]
        Y = self.y0 + (np.arange(j0, j1) + 0.5) * CELL - c2[1]
        dx, dy = np.meshgrid(X, Y)
        det = ex2[0] * ez2[1] - ex2[1] * ez2[0]
        u = (dx * ez2[1] - dy * ez2[0]) / det
        w = (ex2[0] * dy - ex2[1] * dx) / det
        r = np.hypot(u, w); az = np.degrees(np.arctan2(u, w))
        vis = (r >= G.R_MIN) & (r <= G.R_MAX) & (np.abs(az) <= FOV_DEG)
        co = np.stack([(r[vis] - RG[0]) / (RG[1] - RG[0]), (az[vis] - AG[0]) / (AG[1] - AG[0])])
        self.cnt[j0:j1, i0:i1] += vis
        for a, ims in zip(self.acc, images):
            for l, im in enumerate(ims):
                if im is not None:
                    a[l, j0:j1, i0:i1][vis] += map_coordinates(im, co, order=1, mode="nearest")

    def mean(self, k):
        """(layers, ny, nx) mean evidence per frame that could see the cell."""
        return self.acc[k] / np.maximum(self.cnt, N_MIN)[None]


def heat_rgba(M, vmax):
    """Mean evidence (ny, nx; row 0 = lowest sy) -> RGBA raster at PX_PER_M, row 0 = top."""
    import cv2
    k = int(round(PX_PER_M * CELL))
    v = cv2.resize(M[::-1].astype(np.float32), (M.shape[1] * k, M.shape[0] * k), interpolation=cv2.INTER_CUBIC)
    v = np.clip(v / vmax, 0, 1)
    v = np.clip((v - V_FLOOR) / (1 - V_FLOOR), 0, 1) ** GAMMA
    x = v * (len(STOPS) - 1)
    i = np.minimum(x.astype(int), len(STOPS) - 2)
    f = (x - i)[..., None]
    rgb = STOPS[i] * (1 - f) + STOPS[i + 1] * f
    alpha = A_MAX * np.minimum(v / 0.35, 1.0) ** 0.8 * (v > 0)          # opaque enough from 35 % on; colour carries the rest
    return np.dstack([rgb, 255 * alpha]).round().astype(np.uint8)


def near_fracs(M, dist, seen):
    tot = float(M[seen].sum())
    return [float(M[seen & (dist <= d)].sum() / tot) if tot > 0 else float("nan") for d in NEAR_M]


def run():
    from scipy.ndimage import distance_transform_edt
    rec = [r for r in C.recordings(TAKE) if r.file.stem.startswith(REC_PREFIX)][0]
    tags = C.load_tags()
    tm = TakeMaps(TAKE)
    tm.fit_scale(tags)
    s_w = tm.metric(WINDOW)
    Pw, col, w = window_points(tm, WINDOW, STRIDE)
    A, o = map_frame(tm, w, Pw[::7] * s_w, s_w)
    Q = (Pw * s_w - o) @ A.T                                   # map frame, metres: x walk, z up
    t_kf = tm.t[w["idx"]]
    c2w = w["c2w"].copy(); c2w[:, :3, 3] *= s_w
    to_map = lambda p: (p - o) @ A.T
    VT = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])     # top view: screen right = -y, screen up = x (walk)
    VI = view_basis(VIEW_AZ, VIEW_EL)

    # ---- radar: (i) per range bin, (ii) per range-Doppler cell --------------------------------------------------
    F1 = R.features(rec)                                        # power summed over Doppler, receivers averaged
    t = F1.t
    t0, t1 = max(t_kf.min(), t[0]), min(t_kf.max(), t[-1])
    sel = np.flatnonzero((t >= t0) & (t <= t1))
    L = 10 * np.log10(np.maximum(F1.P[sel], 1e-15))
    L = L - np.median(L[:, 8:110], 1, keepdims=True)
    W1 = np.clip((L - DB_LO) / DB_SPAN, 0, 1)
    S = F1.S_az[sel].copy()
    S[:, 1:-1] = S[:, :-2] + S[:, 1:-1] + S[:, 2:]              # +-1 range bin, as work/audit/align_videos.py
    cal = R.angle_calib()["az"]
    AZ1 = R.phase_to_angle(np.angle(S), cal["sign"], cal["offset_rad"])
    use = (F1.r >= G.R_MIN) & (F1.r <= G.R_MAX)                  # bins below R_MIN hold the leakage, not the scene

    tuned = json.loads((C.RESULTS / "analysis-v2" / "tables" / "opt_best_params_pooled_arc.json").read_text())["grid_doppler"]
    p = dict(PL.DEFAULTS["grid_doppler"], **tuned)
    F2 = R.features(rec, {k: p[k] for k in R.PRE_DEFAULT if k in p})
    d = GD.detections(rec, F2, p)
    m = (d["f"] >= sel[0]) & (d["f"] <= sel[-1])
    d = {k: v[m] for k, v in d.items()}
    assert (np.diff(d["f"]) >= 0).all()
    # weight of a range-Doppler cell: dB over the SAME reference as map (i), the frame's median (over the range
    # bins) of the Doppler-summed power. (Over the median range-Doppler cell, which is noise, 94 % of the cells
    # would saturate the 3..30 dB scale and the map would be a count of detections.)
    ref = np.median(F1.P[:, 8:110], 1)
    L2 = 10 * np.log10(np.maximum(F2.RD[d["f"], d["k"], d["b"]], 1e-300) / ref[d["f"]])
    W2 = np.clip((L2 - DB_LO) / DB_SPAN, 0, 1)
    C.log(f"  f08b: range-Doppler cells {len(W2)}, weight > 0: {(W2 > 0).mean():.2f}, saturated: {(W2 >= 1).mean():.3f}; "
          f"dB over the bin median p10/50/90: {np.round(np.percentile(L2, [10, 50, 90]), 1)}; "
          f"over the cell median (snr_db): {np.round(np.percentile(d['snr_db'], [10, 50, 90]), 1)}")
    starts = np.searchsorted(d["f"], np.r_[sel, sel[-1] + 1])

    # ---- poses, grid ------------------------------------------------------------------------------------------------
    Tt = interp_poses(t_kf, c2w, t[sel])
    cam = to_map(Tt[:, :3, 3]) @ VT.T
    ex = (Tt[:, :3, 0] @ A.T) @ VT.T                             # radar x (right) and z (boresight) in top-view coords
    ez = (Tt[:, :3, 2] @ A.T) @ VT.T
    pitch = np.degrees(np.arcsin(np.clip(ez[:, 2], -1, 1)))      # boresight above (+) / below (-) the horizontal
    radar_h = float(np.median(cam[:, 2]))
    pad = G.R_MAX + 0.2
    gx0 = np.floor((cam[:, 0].min() - pad) / CELL) * CELL; gy0 = np.floor((cam[:, 1].min() - pad) / CELL) * CELL
    nx = int(np.ceil((cam[:, 0].max() + pad - gx0) / CELL)); ny = int(np.ceil((cam[:, 1].max() + pad - gy0) / CELL))
    real = Acc(gx0, gy0, nx, ny, (1, 3))
    ctrl = [Acc(gx0, gy0, nx, ny, (1, 3)) for _ in range(N_CTRL)]
    mirr = Acc(gx0, gy0, nx, ny, (1, 3))                         # second control: azimuth mirrored about the boresight
    rng = np.random.default_rng(SEED)
    for i in range(len(sel)):
        im1 = polar_image(F1.r[use], AZ1[i, use], W1[i, use])
        a, b = starts[i], starts[i + 1]
        im2 = [polar_image(d["r"][a:b][d["layer"][a:b] == l], d["az"][a:b][d["layer"][a:b] == l],
                           W2[a:b][d["layer"][a:b] == l]) if (d["layer"][a:b] == l).any() else None for l in range(3)]
        real.add(cam[i, :2], ex[i, :2], ez[i, :2], ([im1], im2))
        mirr.add(cam[i, :2], -ex[i, :2], ez[i, :2], ([im1], im2))
        for c in ctrl:
            ph = rng.uniform(0, 2 * np.pi)
            Rz = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
            c.add(cam[i, :2], Rz @ ex[i, :2], Rz @ ez[i, :2], ([im1], im2))

    # ---- obstacle surfaces of the map on the same grid -------------------------------------------------------
    St = Q @ VT.T
    ob = (Q[:, 2] > max(OBST_H_MIN, radar_h - OBST_BAND)) & (Q[:, 2] < radar_h + OBST_BAND)
    ix = np.floor((St[ob, 0] - gx0) / CELL).astype(int); iy = np.floor((St[ob, 1] - gy0) / CELL).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    occ = np.bincount(iy[ok] * nx + ix[ok], minlength=nx * ny).reshape(ny, nx) >= OBST_MIN_PTS
    dist = distance_transform_edt(~occ) * CELL
    seen = real.cnt > 0

    def summary(k, layer=None):
        pick = (lambda a: a.mean(k).sum(0)) if layer is None else (lambda a: a.mean(k)[layer])
        fr = near_fracs(pick(real), dist, seen)
        fc = np.array([near_fracs(pick(c), dist, c.cnt > 0) for c in ctrl])
        v = np.sort(pick(real)[seen])[::-1]
        half = float((np.searchsorted(np.cumsum(v), 0.5 * v.sum()) + 1) * CELL ** 2)      # area holding half the evidence
        return dict(near_m=list(NEAR_M), frac=fr, chance=fc.mean(0), chance_sd=fc.std(0), area_half_m2=half,
                    mirror=near_fracs(pick(mirr), dist, mirr.cnt > 0),
                    total=float(pick(real)[seen].sum()))

    area = [float((seen & (dist <= dm)).sum() / seen.sum()) for dm in NEAR_M]
    vl = [None, float(p["gd_v1"]), float(p["gd_v2"])]
    maps = dict(bin=dict(summary(0), n_contrib=int((W1[:, use] > 0).sum())),
                rd=dict(summary(1), n_contrib=int((W2 > 0).sum()), n_det=int(len(W2)),
                        layers=[dict(summary(1, l), layer=l, closing_min_mps=vl[l],
                                     n_det=int((d["layer"] == l).sum()), n_contrib=int(((d["layer"] == l) & (W2 > 0)).sum()))
                                for l in range(3)]))
    for k_, m_ in maps.items():
        C.log(f"  f08b {k_}: near {np.round(m_['frac'], 3)} chance {np.round(m_['chance'], 3)} mirror {np.round(m_['mirror'], 3)} "
              f"area with half the evidence {m_['area_half_m2']:.2f} m2")
    for m_ in maps["rd"]["layers"]:
        C.log(f"  f08b rd layer {m_['layer']}: near {np.round(m_['frac'], 3)} chance {np.round(m_['chance'], 3)} mirror {np.round(m_['mirror'], 3)} n {m_['n_contrib']} total {m_['total']:.1f}")

    # ---- views ----------------------------------------------------------------------------------------------------
    keep = (Q[:, 2] < H_CUT) & (Q[:, 2] > -0.3)
    Qk, ck = Q[keep], col[keep]
    traj = to_map(c2w[:, :3, 3])
    traj = traj[(t_kf >= t0) & (t_kf <= t1)]
    tg = tags[(tags["take"] == TAKE) & (tags.t_host >= t0) & (tags.t_host <= t1) & (tags.reproj_px < 2.0)
              & tags.id.isin(list(C.TAG_KIND))]
    tag_rows = []
    if len(tg):
        Tg = interp_poses(t_kf, c2w, tg.t_host.to_numpy())
        pg = to_map(np.einsum("nij,nj->ni", Tg[:, :3, :3], tg[["tx", "ty", "tz"]].to_numpy(float)) + Tg[:, :3, 3])
        for i in sorted(tg.id.unique()):
            mm = (tg.id == i).to_numpy()
            if mm.sum() >= 3:
                tag_rows.append(dict(id=int(i), kind=C.TAG_KIND[int(i)], n=int(mm.sum()), xyz=np.median(pg[mm], 0)))
    stats = dict(rec=rec.key, window=WINDOW, t0_s=float(t0 - t[0]), t1_s=float(t1 - t[0]), duration_s=float(t1 - t0),
                 n_frames=int(len(sel)), fps=float((len(sel) - 1) / (t[sel[-1]] - t[sel[0]])),
                 path_m=float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()), radar_height_m=radar_h,
                 pitch_deg_median=float(np.median(pitch)), pitch_deg_p10_p90=np.percentile(pitch, [10, 90]),
                 cell_m=CELL, sigma_r_m=SIG_R, sigma_az_deg=SIG_AZ, fov_deg=FOV_DEG, r_min_m=G.R_MIN, r_max_m=G.R_MAX,
                 db_lo=DB_LO, db_hi=DB_LO + DB_SPAN, n_min_frames=N_MIN, n_ctrl=N_CTRL,
                 obstacle=dict(h_min_m=max(OBST_H_MIN, radar_h - OBST_BAND), h_max_m=radar_h + OBST_BAND,
                               min_pts=OBST_MIN_PTS, n_cells=int(occ.sum())),
                 area_frac_near=area, metric_scale=float(s_w), n_points=int(len(Q)), n_keyframes=int(len(t_kf)),
                 cfar=dict({k: p[k] for k in ("guard_d", "guard_r", "train_d", "train_r", "k_db", "gd_v1", "gd_v2", "gd_max",
                                              "r_min", "r_max", "combine")}), maps=maps)
    C.log(f"  f08b: {json.dumps(stats, default=lambda v: np.asarray(v).tolist())}")

    def overlays(V, floor=False):
        pr = lambda q: (np.atleast_2d(q) @ V.T)[:, :2]
        return dict(traj=np.round(pr(traj), 3), traj_floor=np.round(pr(traj * [1, 1, 0]), 3),
                    axes=dict(origin=pr([0, 0, 0])[0], x=pr([1, 0, 0])[0], y=pr([0, 1, 0])[0], z=pr([0, 0, 1])[0]),
                    tags=[dict(id=g["id"], kind=g["kind"], n=g["n"], p=np.round(pr(g["xyz"])[0], 3),
                               h=round(float(g["xyz"][2]), 2)) for g in tag_rows])

    # oblique
    Si = Qk @ VI.T
    ti = (traj @ VI.T)[:, :2]; tf = ((traj * [1, 1, 0]) @ VI.T)[:, :2]
    lo, hi = np.percentile(Si[:, :2], [0.3, 99.7], axis=0)
    lo = np.minimum(lo, np.minimum(ti.min(0), tf.min(0))) - 0.15; hi = np.maximum(hi, np.maximum(ti.max(0), tf.max(0))) + 0.15
    xlim, ylim = (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))
    img = splat(Si, ck, xlim, ylim, 2, 150.0)
    P.save_png("f08b_iso", img)
    P.save_json("f08b_iso", dict(units="m (plano da vista, projeção ortográfica)", xlim=xlim, ylim=ylim,
                                 px=[img.shape[1], img.shape[0]], view=dict(az_deg=VIEW_AZ, el_deg=VIEW_EL),
                                 **overlays(VI), stats=stats))

    # top view: extent snapped to the evidence grid
    Sk = Qk @ VT.T
    lo, hi = np.percentile(Sk[:, :2], [0.3, 99.7], axis=0)
    lo = np.minimum(lo, cam[:, :2].min(0)) - 0.25; hi = np.maximum(hi, cam[:, :2].max(0)) + 0.25
    for k in (0, 1):                                              # widen (at most EXT_MAX) to hold the hot evidence
        M = real.mean(k).sum(0)
        jj, ii = np.nonzero(M >= EXT_LEVEL * np.percentile(M[M > 0], V_PCT))
        lo = np.maximum(np.minimum(lo, [gx0 + ii.min() * CELL - 0.15, gy0 + jj.min() * CELL - 0.15]), lo - EXT_MAX)
        hi = np.minimum(np.maximum(hi, [gx0 + (ii.max() + 1) * CELL + 0.15, gy0 + (jj.max() + 1) * CELL + 0.15]), hi + EXT_MAX)
    ia = max(int(np.floor((lo[0] - gx0) / CELL)), 0); ib = min(int(np.ceil((hi[0] - gx0) / CELL)), nx)
    ja = max(int(np.floor((lo[1] - gy0) / CELL)), 0); jb = min(int(np.ceil((hi[1] - gy0) / CELL)), ny)
    xlim = (float(gx0 + ia * CELL), float(gx0 + ib * CELL)); ylim = (float(gy0 + ja * CELL), float(gy0 + jb * CELL))
    base = splat(Sk, ck, xlim, ylim, 2)
    P.save_png("f08b_topo", (255 - (255 - base.astype(float)) * (1 - LIGHTEN)).round().astype(np.uint8))
    heat = {}
    for key, k in (("bin", 0), ("rd", 1)):
        M = real.mean(k).sum(0)
        crop = M[ja:jb, ia:ib]
        vmax = float(np.percentile(crop[crop > 0], V_PCT))
        rgba = heat_rgba(crop, vmax)
        assert rgba.shape[:2] == base.shape[:2], (rgba.shape, base.shape)
        P.save_png(f"f08b_calor_{key}", rgba)
        heat[key] = dict(image=f"f08b_calor_{key}", vmax=vmax, frac_inside_view=float(crop.sum() / M.sum()),
                         **maps[key])
    # outline of the area the radar could see in at least one frame (explains where the heat stops)
    import cv2
    cs, _ = cv2.findContours((real.cnt > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cover = []
    for c in cs:
        if len(c) >= 40:
            q = cv2.approxPolyDP(c.astype(np.float32), 0.7, True).reshape(-1, 2)
            q = np.stack([gx0 + (q[:, 0] + 0.5) * CELL, gy0 + (q[:, 1] + 0.5) * CELL], 1)
            cover.append(np.round(np.r_[q, q[:1]], 3))
    # colour key: the same ramp as heat_rgba, sampled
    vv = np.linspace(0, 1, 9)
    kk = heat_rgba(np.tile(vv, (1, 1)), 1.0)
    step = kk.shape[1] // len(vv)
    ramp = [[int(c) for c in kk[0, j * step + step // 2]] for j in range(len(vv))]
    P.save_json("f08b_topo", dict(units="m (planta, vista superior)", xlim=xlim, ylim=ylim,
                                  px=[base.shape[1], base.shape[0]], view="top", **overlays(VT), cover=cover,
                                  heat=heat, ramp=dict(v=vv, rgba=ramp, v_pct=V_PCT, lighten=LIGHTEN), stats=stats))
    # scratch check (not a figure file): obstacle cells of the displayed extent
    try:
        dbg = C.ROOT / "work" / "f08b"; dbg.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg / "occ.png"), cv2.resize((~occ[ja:jb, ia:ib][::-1]).astype(np.uint8) * 255, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
        np.savez(dbg / "maps.npz", bin=real.mean(0), rd=real.mean(1), cnt=real.cnt, dist=dist, crop=[ja, jb, ia, ib])
    except Exception as e:                                       # noqa: BLE001
        C.log(f"  f08b: debug output skipped ({e})")


if __name__ == "__main__":
    run()
