"""Global map from the pod's windows, on the PC (no GPU): pose graph with loop closures + weighted TSDF fusion.

    uv run --group viz python scripts/map_fuse.py --model pi3x --take indoor_course
    make fuse MODEL=pi3x TAKE=indoor_course

Steps (each window = one batch of keyframes the model reconstructed in one pass; they are consistent inside):

1. Join consecutive windows robustly (``map_viz.chain_windows``: every shared keyframe proposes a similarity, the
   one most others agree with is refitted on its inliers).
2. Loop closures: keyframes of windows at least two apart that see the same place (thumbnail similarity, then ORB
   matches), verified geometrically: the matched pixels of keyframe A, lifted to 3D with A's depth, give keyframe
   B's pose by PnP + RANSAC; the depth of the inliers gives the scale. Each closure is a similarity between the two
   windows, the same kind of constraint as a join.
3. Pose graph over the windows (one similarity each, window 0 fixed): joins + closures, Huber loss, so a wrong
   closure cannot pull the map apart. Feed-forward reconstructions with known intrinsics only need a similarity per
   submap (VGGT-SLAM's SL(4) is for the uncalibrated case).
4. Weighted TSDF fusion of all keyframes with the optimised poses, sparse voxels: each depth sample counts with
   weight = (model confidence / its frame median) x (1 m / depth)^2 — near and confident data dominate, overlaps
   average instead of piling up. Kept: voxels on the zero crossing seen by >= 3 frames. Pixels on the black
   border left by the undistortion are masked out (they were the "embossed" frame outlines).

Outputs in ``<maps>/fused/<model>/<take>/``: ``graph.json`` (closures, residuals before / after, settings),
``poses.npz`` (per window similarity; per keyframe camera-to-world and scale — the optimised "oracle" poses for the
analyses), ``fused.npz`` + ``fused.ply`` (points, colours, weights). ``scripts/map_viz.py`` shows them
(View: fused map; Chaining: pose graph).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import map_viz as mv  # noqa: E402


# ------------------------------------------------------------------------------------------------ similarity
def sim3(s: float, R: np.ndarray, t: np.ndarray) -> dict:
    return {"s": float(s), "R": np.asarray(R, float), "t": np.asarray(t, float)}


def sim3_inv(a: dict) -> dict:
    Ri = a["R"].T
    return sim3(1 / a["s"], Ri, -Ri @ a["t"] / a["s"])


def sim3_mul(a: dict, b: dict) -> dict:
    """a ∘ b."""
    return sim3(a["s"] * b["s"], a["R"] @ b["R"], a["s"] * a["R"] @ b["t"] + a["t"])


def window_sim3(ch, k: int) -> dict:
    """Window k -> global, as chained: x_g = R (S x) + t."""
    G = ch.G[k]
    return sim3(ch.S[k], G[:3, :3], G[:3, 3])


def apply_pose(T: dict, c2w: np.ndarray) -> np.ndarray:
    """Global camera-to-world (rotation, centre) of a window camera; the depth must still be scaled by T['s']."""
    out = np.eye(4)
    out[:3, :3] = T["R"] @ c2w[:3, :3]
    out[:3, 3] = T["s"] * T["R"] @ c2w[:3, 3] + T["t"]
    return out


# ------------------------------------------------------------------------------------------------ border mask
def valid_mask(data: Path, res: int, erode: int = 3) -> np.ndarray:
    """Pixels of the res x res keyframe that came from inside the original image (pod/frames.py undistorts the
    1440 x 1440 video with the calibration; the rest is black), shrunk by `erode` pixels."""
    import cv2
    cal = json.loads((data / "calibration" / "camera.calib.json").read_text(encoding="utf-8"))["display"]
    W, H = cal["image_size"]
    K, D = np.array(cal["K"], float), np.array(cal["dist"], float)
    m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
    ones = cv2.remap(np.ones((H, W), np.float32), m1, m2, cv2.INTER_LINEAR, borderValue=0)
    small = cv2.resize(ones, (res, res), interpolation=cv2.INTER_AREA) > 0.999
    if erode:
        small = cv2.erode(small.astype(np.uint8), np.ones((2 * erode + 1, 2 * erode + 1), np.uint8)) > 0
    return small


def mask_for(mask518: np.ndarray, shape) -> np.ndarray:
    import cv2
    if mask518.shape == tuple(shape):
        return mask518
    return cv2.resize(mask518.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


# ------------------------------------------------------------------------------------------------ loop closures
def keyframe_features(rgb_dir: Path, frames: list[int], mask518: np.ndarray, n_feat: int = 1500):
    import cv2
    orb = cv2.ORB_create(n_feat)
    feats, thumbs = {}, {}
    m8 = mask518.astype(np.uint8) * 255
    for f in frames:
        im = cv2.imread(str(rgb_dir / f"{f:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        kp, des = orb.detectAndCompute(im, m8)
        if des is None or len(kp) < 50:
            continue
        feats[f] = (np.array([k.pt for k in kp], np.float32), des)
        t = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        thumbs[f] = (t - t.mean()) / (t.std() + 1e-6)
    return feats, thumbs


def find_closures(wins, ch, own, rgb_dir, mask518, K518, tvid, min_gap_s=20.0, min_win_gap=2, step=2,
                  per_frame=6, min_inliers=60, log=print):
    """Loop-closure similarities between windows (dicts with i, j, frames, inliers, T_i<-j)."""
    import cv2
    fs = [f for n, f in enumerate(sorted(own)) if n % step == 0]
    feats, thumbs = keyframe_features(rgb_dir, fs, mask518)
    fs = [f for f in fs if f in feats]
    if len(fs) < 4:
        return []
    T = np.stack([thumbs[f] for f in fs])
    sim = T @ T.T / T.shape[1]
    win_of = np.array([own[f][0] for f in fs])
    t_of = np.array([tvid.get(f, 0.0) for f in fs])
    ok = (np.abs(win_of[:, None] - win_of[None, :]) >= min_win_gap) & (np.abs(t_of[:, None] - t_of[None, :]) >= min_gap_s)
    ok &= np.triu(np.ones_like(ok), 1).astype(bool)
    sim = np.where(ok, sim, -np.inf)
    cand = set()
    for a in range(len(fs)):
        for b in np.argsort(sim[a])[::-1][:per_frame]:
            if np.isfinite(sim[a, b]):
                cand.add((a, b))
    log(f"  loop closures: {len(fs)} keyframes, {len(cand)} candidate pairs")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best: dict[tuple[int, int], dict] = {}
    for a, b in sorted(cand):
        fa, fb = fs[a], fs[b]
        ma = bf.match(feats[fa][1], feats[fb][1])
        ma = [m for m in ma if m.distance < 48]
        if len(ma) < min_inliers:
            continue
        r = _closure(wins, own, fa, fb, feats, ma, K518, min_inliers)
        if r is None:
            continue
        key = (min(r["i"], r["j"]), max(r["i"], r["j"]))
        if key not in best or r["inliers"] > best[key]["inliers"]:
            best[key] = r
    out = sorted(best.values(), key=lambda r: (r["i"], r["j"]))
    log(f"  loop closures: {len(out)} verified (window pairs)")
    return out


def _closure(wins, own, fa, fb, feats, matches, K518, min_inliers):
    import cv2
    ka, ja = own[fa]
    kb, jb = own[fb]
    wa, wb = wins[ka], wins[kb]
    pa = feats[fa][0][[m.queryIdx for m in matches]]
    pb = feats[fb][0][[m.trainIdx for m in matches]]
    Ha, Wa = wa.depth.shape[1:]
    sa = Ha / 518.0
    ua, va = np.clip((pa[:, 0] * sa).astype(int), 0, Wa - 1), np.clip((pa[:, 1] * sa).astype(int), 0, Ha - 1)
    za = wa.depth[ja][va, ua].astype(np.float64)
    good = za > 0
    if good.sum() < min_inliers:
        return None
    Kd = wa.K[ja]
    P = np.stack([(ua - Kd[0, 2]) / Kd[0, 0] * za, (va - Kd[1, 2]) / Kd[1, 1] * za, za], 1)[good]
    X = P @ wa.c2w[ja][:3, :3].T + wa.c2w[ja][:3, 3]                 # window a coordinates
    ok, rvec, tvec, inl = cv2.solvePnPRansac(X.astype(np.float64), pb[good].astype(np.float64), K518, None,
                                             reprojectionError=4.0, iterationsCount=300, confidence=0.999,
                                             flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl) < min_inliers:
        return None
    inl = inl.ravel()
    R, _ = cv2.Rodrigues(rvec)
    zi = (X[inl] @ R.T + tvec.ravel())[:, 2]                          # depth in b, window-a units
    Hb, Wb = wb.depth.shape[1:]
    sb = Hb / 518.0
    ub = np.clip((pb[good][inl, 0] * sb).astype(int), 0, Wb - 1)
    vb = np.clip((pb[good][inl, 1] * sb).astype(int), 0, Hb - 1)
    zj = wb.depth[jb][vb, ub].astype(np.float64)
    m = (zj > 0) & (zi > 0)
    if m.sum() < min_inliers // 2:
        return None
    s = float(np.median(zi[m] / zj[m]))
    c2w_bi = np.eye(4); c2w_bi[:3, :3] = R.T; c2w_bi[:3, 3] = -R.T @ tvec.ravel()
    Rbi, tbi = c2w_bi[:3, :3], c2w_bi[:3, 3]
    Rbj, tbj = wb.c2w[jb][:3, :3], wb.c2w[jb][:3, 3]
    Rr = Rbi @ Rbj.T
    T_ab = sim3(s, Rr, tbi - s * Rr @ tbj)                            # window b -> window a
    return {"i": int(ka), "j": int(kb), "fa": int(fa), "fb": int(fb), "inliers": int(len(inl)), "T": T_ab,
            "depth": float(np.median(za[good]))}


# ------------------------------------------------------------------------------------------------ pose graph
def optimise(n: int, init: list[dict], edges: list[dict], depth_scale: list[float], log=print) -> list[dict]:
    """Similarity per window (window 0 fixed) from relative constraints {i, j, T (j -> i), w}."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    def unpack(x):
        out = [init[0]]
        for k in range(1, n):
            p = x[7 * (k - 1): 7 * k]
            out.append(sim3(np.exp(p[6]), Rot.from_rotvec(p[:3]).as_matrix(), p[3:6]))
        return out

    x0 = np.concatenate([np.r_[Rot.from_matrix(T["R"]).as_rotvec(), T["t"], np.log(T["s"])] for T in init[1:]])
    sig_r, sig_s = np.radians(1.0), 0.02

    def res(x):
        Ts = unpack(x)
        r = []
        for e in edges:
            P = sim3_mul(sim3_inv(Ts[e["i"]]), Ts[e["j"]])
            M = e["T"]
            sig_t = 0.05 * depth_scale[e["i"]]
            w = np.sqrt(e["w"])
            r.append(w * Rot.from_matrix(M["R"].T @ P["R"]).as_rotvec() / sig_r)
            r.append(w * (P["t"] - M["t"]) / sig_t)
            r.append([w * np.log(P["s"] / M["s"]) / sig_s])
        return np.concatenate(r)

    r0 = res(x0)
    sol = least_squares(res, x0, loss="huber", f_scale=3.0, max_nfev=200)
    log(f"  pose graph: {len(edges)} edges, cost {0.5 * (r0 ** 2).sum():.1f} -> {0.5 * (sol.fun ** 2).sum():.1f}")
    return unpack(sol.x)


def edge_errors(Ts, edges) -> list[dict]:
    from scipy.spatial.transform import Rotation as Rot
    out = []
    for e in edges:
        P = sim3_mul(sim3_inv(Ts[e["i"]]), Ts[e["j"]])
        M = e["T"]
        out.append({"i": e["i"], "j": e["j"], "kind": e["kind"],
                    "rot_deg": float(np.degrees(np.linalg.norm(Rot.from_matrix(M["R"].T @ P["R"]).as_rotvec()))),
                    "trans": float(np.linalg.norm(P["t"] - M["t"])), "scale": float(P["s"] / M["s"])})
    return out


# ------------------------------------------------------------------------------------------------ ICP refinement
def window_points(w, T: dict, mask518, stride: int = 6, zmax: float = np.inf, conf_drop: float = 30.0,
                  every: int = 2) -> np.ndarray:
    """Points of a window in global coordinates (with the window similarity T), for ICP."""
    out = []
    for j in range(0, len(w.idx), every):
        d = w.depth[j].astype(np.float32)
        H, W = d.shape
        ok = (d > 0) & np.isfinite(d) & (d * T["s"] <= zmax) & mask_for(mask518, (H, W))
        cf = w.conf[j].astype(np.float32)
        if conf_drop > 0:
            ok &= cf >= np.percentile(cf, conf_drop)
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        sel = ok[vv, uu]
        u, v, z = uu[sel], vv[sel], d[vv, uu][sel]
        K = w.K[j]
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
        x = pc @ w.c2w[j][:3, :3].T + w.c2w[j][:3, 3]
        out.append(T["s"] * x @ T["R"].T + T["t"])
    return np.concatenate(out) if out else np.zeros((0, 3))


def voxel_down(P: np.ndarray, v: float) -> np.ndarray:
    if len(P) == 0:
        return P
    k = np.floor(P / v).astype(np.int64)
    _, inv = np.unique(k, axis=0, return_inverse=True)
    inv = inv.ravel()
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, P[:, i]) / cnt for i in range(3)], 1)


def normals(P: np.ndarray, k: int = 16) -> np.ndarray:
    from scipy.spatial import cKDTree
    _, idx = cKDTree(P).query(P, k=min(k, len(P)))
    nb = P[idx] - P[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vec = np.linalg.eigh(cov)
    return vec[:, :, 0]                                          # smallest eigenvalue: the normal


def icp(P: np.ndarray, Q: np.ndarray, NQ: np.ndarray, max_d: float, iters: int = 30):
    """Point-to-plane ICP moving P onto Q (rigid). Returns 4x4, fitness (fraction of P matched), rmse."""
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot
    tree = cKDTree(Q)
    T = np.eye(4)
    X = P.copy()
    for it in range(iters):
        d, i = tree.query(X, distance_upper_bound=max_d)
        m = np.isfinite(d)
        if m.sum() < 100:
            return T, 0.0, np.inf
        x, q, n = X[m], Q[i[m]], NQ[i[m]]
        A = np.hstack([np.cross(x, n), n])                       # small-angle linearisation
        b = ((q - x) * n).sum(1)
        r = np.abs(b)
        wgt = 1.0 / np.maximum(1.0, r / (0.3 * max_d))           # Huber-like
        sol, *_ = np.linalg.lstsq(A * wgt[:, None], b * wgt, rcond=None)
        dT = np.eye(4)
        dT[:3, :3] = Rot.from_rotvec(sol[:3]).as_matrix()
        dT[:3, 3] = sol[3:]
        X = X @ dT[:3, :3].T + dT[:3, 3]
        T = dT @ T
        if it in (10, 20):
            max_d *= 0.6                                         # tighten
        if np.linalg.norm(sol) < 1e-6:
            break
    d, i = tree.query(X, distance_upper_bound=max_d)
    m = np.isfinite(d)
    return T, float(m.mean()), float(np.sqrt((d[m] ** 2).mean())) if m.any() else np.inf


def refine_edges(wins, Ts, edges, mask518, voxel: float, zmax: float, log=print) -> int:
    """Dense geometric check of every join / closure: point-to-plane ICP between the two windows' points as the
    graph currently places them; a good fit (enough overlap, small residual, small correction) replaces the edge
    measurement. Returns the number of refined edges."""
    cache: dict[int, tuple] = {}

    def cloud(k):
        if k not in cache:
            P = voxel_down(window_points(wins[k], Ts[k], mask518, zmax=zmax), 2 * voxel)
            cache[k] = (P, normals(P) if len(P) > 50 else None)
        return cache[k]

    n_ok, corr = 0, []
    for e in edges:
        Q, NQ = cloud(e["i"])
        P, _ = cloud(e["j"])
        if NQ is None or len(P) < 100:
            continue
        D, fit, rmse = icp(P, Q, NQ, max_d=6 * voxel)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))
        mv_ = float(np.linalg.norm(D[:3, 3]))
        e["icp"] = {"fitness": round(fit, 3), "rmse": round(rmse, 4), "deg": round(ang, 2), "move": round(mv_, 4)}
        if fit >= 0.3 and rmse <= 2.5 * voxel and ang <= 10 and mv_ <= 10 * voxel:
            Dsim = sim3(1.0, D[:3, :3], D[:3, 3])
            e["T"] = sim3_mul(sim3_inv(Ts[e["i"]]), sim3_mul(Dsim, Ts[e["j"]]))   # Ti^-1 ∘ D ∘ Tj
            e["w"] = e["w"] * (1.0 + 2.0 * fit)
            n_ok += 1
            corr.append((mv_, ang))
        if len(cache) > 6:                                        # keep memory small
            for k in list(cache)[:-4]:
                cache.pop(k)
    if corr:
        c = np.array(corr)
        log(f"  ICP: {n_ok}/{len(edges)} edges refined (median correction {np.median(c[:, 0]):.3f}, "
            f"{np.median(c[:, 1]):.2f}°)")
    else:
        log(f"  ICP: no edge refined")
    return n_ok


# ------------------------------------------------------------------------------------------------ TSDF
class SparseTSDF:
    """Sparse voxel TSDF with per-sample weights (sum w, sum w*sdf, sum w*rgb, frames seen)."""

    def __init__(self, voxel: float, trunc_vox: int = 4):
        self.v, self.nt = voxel, trunc_vox
        self.keys = np.zeros(0, np.int64)
        self.W = np.zeros(0, np.float32); self.WD = np.zeros(0, np.float32)
        self.WC = np.zeros((0, 3), np.float32); self.NF = np.zeros(0, np.int32)
        self._buf: list[tuple] = []

    @staticmethod
    def _key(ijk: np.ndarray) -> np.ndarray:
        o = ijk.astype(np.int64) + (1 << 20)
        return (o[:, 0] << 42) | (o[:, 1] << 21) | o[:, 2]

    @staticmethod
    def _unkey(k: np.ndarray) -> np.ndarray:
        m = (1 << 21) - 1
        return np.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1) - (1 << 20)

    def integrate(self, pc: np.ndarray, w: np.ndarray, rgb: np.ndarray, T: np.ndarray):
        """pc: (n, 3) camera-frame surface points (already scaled), w: (n,), rgb: (n, 3), T: camera-to-world."""
        z = pc[:, 2:3]
        dirs = pc / z                                                    # depth-parametrised ray
        tau = self.nt * self.v
        keys, ww, wd, wc = [], [], [], []
        for d in np.arange(-self.nt, self.nt + 1) * self.v:
            X = dirs * (z + d)
            Xw = X @ T[:3, :3].T + T[:3, 3]
            keys.append(self._key(np.floor(Xw / self.v)))
            ww.append(w); wd.append(w * np.clip(-d / tau, -1, 1)); wc.append(w[:, None] * rgb)
        k = np.concatenate(keys)
        u, inv = np.unique(k, return_inverse=True)                       # one entry per voxel per frame
        W = np.bincount(inv, np.concatenate(ww)).astype(np.float32)
        D = np.bincount(inv, np.concatenate(wd)).astype(np.float32)
        C = np.stack([np.bincount(inv, np.concatenate([c[:, i] for c in wc])) for i in range(3)], 1).astype(np.float32)
        self._buf.append((u, W, D, C, np.ones(len(u), np.int32)))
        if sum(len(b[0]) for b in self._buf) > 8_000_000:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        k = np.concatenate([self.keys] + [b[0] for b in self._buf])
        u, inv = np.unique(k, return_inverse=True)
        def acc(old, new_idx, dim=None):
            vals = np.concatenate([old] + [b[new_idx] for b in self._buf])
            if vals.ndim == 1:
                return np.bincount(inv, vals, minlength=len(u)).astype(old.dtype)
            return np.stack([np.bincount(inv, vals[:, i], minlength=len(u)) for i in range(vals.shape[1])], 1).astype(old.dtype)
        self.W, self.WD, self.WC, self.NF = acc(self.W, 1), acc(self.WD, 2), acc(self.WC, 3), acc(self.NF, 4)
        self.keys = u
        self._buf = []

    def extract(self, min_frames: int = 3, band: float = 0.5):
        self.flush()
        ok = (self.W > 0) & (self.NF >= min_frames)
        sdf = np.zeros_like(self.W); sdf[ok] = self.WD[ok] / self.W[ok]
        ok &= np.abs(sdf) <= band / self.nt                             # within half a voxel of the surface
        xyz = (self._unkey(self.keys[ok]) + 0.5) * self.v
        rgb = np.clip(self.WC[ok] / self.W[ok, None], 0, 255).astype(np.uint8)
        return xyz.astype(np.float32), rgb, self.W[ok], self.NF[ok]


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
            f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii"))
        fh.write(rec.tobytes())


# ------------------------------------------------------------------------------------------------ main
def run(maps: Path, data: Path, model: str, take: str, voxel="auto", zmax="auto", stride: int = 2,
        conf_drop: float = 30.0, kstep: int = 1, closures: bool = True, use_icp: bool = True, wins=None,
        log=print) -> Path:
    import cv2
    t0 = time.time()
    out = maps / "fused" / model / take
    out.mkdir(parents=True, exist_ok=True)
    log(f"{model} / {take}: loading windows ...")
    if wins is None:
        wins = mv.load_windows(maps / model / take, True)
    for w in wins:
        w.load_maps()
    ch = mv.chain_windows(wins, method="poses")
    own = mv.owners(wins)
    rgb_dir = mv.ensure_frames(maps, data, take)
    meta = json.loads((maps / f"frames_{take}.json").read_text(encoding="utf-8"))
    tvid = {f["i"]: f.get("t_video", 0.0) for f in meta["frames"]}
    K518 = np.array(meta["K"], float)
    mask518 = valid_mask(data, int(meta.get("res", 518)))
    n = len(wins)
    init = [window_sim3(ch, k) for k in range(n)]
    # typical depth of each window in its own units (sets the translation tolerance of its edges)
    depth_scale = [float(np.median(w.depth[:, ::16, ::16][w.depth[:, ::16, ::16] > 0])) for w in wins]

    # ---- edges: joins + loop closures
    edges = []
    for k in range(n - 1):
        T = sim3_mul(sim3_inv(init[k]), init[k + 1])
        edges.append({"i": k, "j": k + 1, "T": T, "w": max(ch.inliers[k], 1) / 12.0, "kind": "join"})
    lcs = []
    if closures and rgb_dir is not None:
        lcs = find_closures(wins, ch, own, rgb_dir, mask518, K518, tvid, log=log)
        for c in lcs:
            edges.append({"i": c["i"], "j": c["j"], "T": c["T"], "w": min(c["inliers"] / 100.0, 2.0), "kind": "loop"})
    before = edge_errors(init, edges)
    Ts = optimise(n, init, edges, depth_scale, log=log) if lcs else init
    med = float(np.median([np.median(w.depth[:, ::16, ::16][w.depth[:, ::16, ::16] > 0]) * T["s"]
                           for w, T in zip(wins, Ts)]))
    voxel = round(0.02 * med, 4) if voxel == "auto" else float(voxel)
    zmax = round(3.5 * med, 2) if zmax == "auto" else float(zmax)
    n_icp = 0
    if use_icp:
        log("  ICP refinement of the joins and closures ...")
        n_icp = refine_edges(wins, Ts, edges, mask518, voxel, zmax, log=log)
        if n_icp:
            Ts = optimise(n, Ts, edges, depth_scale, log=log)
    after = edge_errors(Ts, edges)

    # ---- poses per keyframe
    kf = sorted(own)
    c2w = np.stack([apply_pose(Ts[own[f][0]], wins[own[f][0]].c2w[own[f][1]]) for f in kf])
    scale = np.array([Ts[own[f][0]]["s"] for f in kf])
    np.savez_compressed(out / "poses.npz", keyframes=np.array(kf), c2w=c2w, depth_scale=scale,
                        window_s=np.array([T["s"] for T in Ts]), window_R=np.stack([T["R"] for T in Ts]),
                        window_t=np.stack([T["t"] for T in Ts]),
                        chain_s=np.array([T["s"] for T in init]), chain_R=np.stack([T["R"] for T in init]),
                        chain_t=np.stack([T["t"] for T in init]))
    log(f"  poses: {len(kf)} keyframes ({time.time() - t0:.0f} s)")

    # ---- TSDF
    log(f"  tsdf: voxel {voxel}, max depth {zmax} (median depth {med:.2f})")
    tsdf = SparseTSDF(voxel)
    for n_, f in enumerate(kf):
        if n_ % kstep:
            continue
        k, j = own[f]
        w = wins[k]
        d = w.depth[j].astype(np.float32)
        H, W = d.shape
        s = Ts[k]["s"]
        ok = (d > 0) & np.isfinite(d) & (d * s <= zmax) & mask_for(mask518, (H, W))
        cf = w.conf[j].astype(np.float32)
        if conf_drop > 0:
            ok &= cf >= np.percentile(cf, conf_drop)
        gx = np.zeros_like(d); gy = np.zeros_like(d)
        gx[:, 1:] = np.abs(np.diff(d, axis=1)); gy[1:, :] = np.abs(np.diff(d, axis=0))
        ok &= np.maximum(gx, gy) <= 0.04 * np.maximum(d, 1e-6) + 1e-6
        vv, uu = np.mgrid[0:H:stride, 0:W:stride]
        sel = ok[vv, uu]
        u, v, z = uu[sel], vv[sel], d[vv, uu][sel] * s
        Kd = w.K[j]
        pc = np.stack([(u - Kd[0, 2]) / Kd[0, 0] * z, (v - Kd[1, 2]) / Kd[1, 1] * z, z], 1)
        cfn = np.clip(cf[vv, uu][sel] / (np.median(cf[ok]) + 1e-6), 0, 3)
        wt = (cfn * (1.0 / np.maximum(z, 0.3)) ** 2).astype(np.float32)
        im = cv2.imread(str(rgb_dir / f"{f:06d}.jpg")) if rgb_dir is not None else None
        if im is not None:
            if im.shape[:2] != (H, W):
                im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            rgb = im[v, u, ::-1].astype(np.float32)
        else:
            rgb = np.full((len(z), 3), 160, np.float32)
        tsdf.integrate(pc, wt, rgb, c2w[n_])
        if n_ % 100 == 0:
            log(f"  tsdf: keyframe {n_}/{len(kf)}, {len(tsdf.keys):,} voxels ({time.time() - t0:.0f} s)")
    xyz, rgb, wts, nf = tsdf.extract()
    np.savez_compressed(out / "fused.npz", xyz=xyz, rgb=rgb, weight=wts, frames=nf, voxel=voxel)
    write_ply(out / "fused.ply", xyz, rgb)
    info = {"model": model, "take": take, "windows": n, "keyframes": len(kf), "voxel_m": voxel, "zmax": zmax,
            "stride": stride, "conf_drop_pct": conf_drop, "keyframe_step": kstep, "points": int(len(xyz)),
            "weights": "conf/median(conf) x (1/depth)^2; voxels seen by >= 3 frames; border mask from calibration",
            "closures": [{k: v for k, v in c.items() if k != "T"} for c in lcs],
            "icp_refined_edges": n_icp,
            "icp": [{"i": e["i"], "j": e["j"], "kind": e["kind"], **e.get("icp", {})} for e in edges],
            "edges_before": before, "edges_after": after, "seconds": round(time.time() - t0, 1)}
    (out / "graph.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    lb = [e for e in before if e["kind"] == "loop"]
    la = [e for e in after if e["kind"] == "loop"]
    if lb:
        log(f"  loop closures, median disagreement before / after: "
            f"{np.median([e['trans'] for e in lb]):.2f} / {np.median([e['trans'] for e in la]):.2f}, "
            f"{np.median([e['rot_deg'] for e in lb]):.1f}° / {np.median([e['rot_deg'] for e in la]):.1f}°")
    log(f"  fused: {len(xyz):,} points -> {out} ({time.time() - t0:.0f} s)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(mv.ROOT / "data" / "ravidata"))
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--model", default="pi3x", help="a model, or 'all'")
    ap.add_argument("--take", default="all", help="a take, or 'all'")
    ap.add_argument("--voxel", default="auto",
                    help="TSDF voxel size in model units (m for metric models); auto = 2 %% of the median depth "
                         "(~3 cm indoors), which also works for VGGT's arbitrary units")
    ap.add_argument("--zmax", default="auto", help="ignore depth beyond this; auto = 3.5 x the median depth")
    ap.add_argument("--stride", type=int, default=2, help="pixel stride of each depth map")
    ap.add_argument("--conf-drop", type=float, default=30.0, help="drop this percentile of confidence per frame")
    ap.add_argument("--keyframe-step", type=int, default=1)
    ap.add_argument("--no-closures", action="store_true")
    ap.add_argument("--no-icp", action="store_true", help="skip the point-to-plane ICP refinement of the edges")
    a = ap.parse_args(argv)
    data = Path(a.data)
    maps = data / "derived" / f"maps-{a.tag}"
    models = mv.MODELS if a.model == "all" else [a.model]
    for m in models:
        takes = sorted(p.name for p in (maps / m).glob("*") if p.is_dir()) if a.take == "all" else [a.take]
        for t in takes:
            run(maps, data, m, t, a.voxel, a.zmax, a.stride, a.conf_drop, a.keyframe_step, not a.no_closures,
                not a.no_icp)


if __name__ == "__main__":
    main()
