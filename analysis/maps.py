"""Feed-forward 3D maps (derived/maps-v1) as ground truth: keyframe depth, poses and metric scale.

Each keyframe is taken from its owner window (where it is most central). Depth is in the window's own units;
the metric scale per window comes from the AprilTags seen in its keyframes (tag distance from solvePnP over the
map depth at the tag centre). Fused poses (derived/maps-v1/fused/<model>/<take>/poses.npz, from map_fuse) give
one trajectory per take (global units = window units x window_s)."""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import common as C

MAPS = C.DATA / "derived" / "maps-v1"


class TakeMaps:
    def __init__(self, take: str, model: str = "pi3x"):
        self.take, self.model = take, model
        meta = json.loads((MAPS / f"frames_{take}.json").read_text(encoding="utf-8"))
        self.res = int(meta.get("res", 518))
        self.K = np.array(meta["K"], float)
        fr = meta["frames"]
        self.vf = np.array([f["video_frame"] for f in fr])
        sync = C.sync_of(take)
        self.t = sync.video_to_host(np.array([f["t_video"] for f in fr], float))
        wdir = MAPS / model / take
        self.wfiles = sorted(wdir.glob("win_*.npz"))
        best = {}
        for k, p in enumerate(self.wfiles):
            idx = np.load(p)["idx"]
            n = len(idx)
            for j, f in enumerate(idx):
                c = min(j, n - 1 - j)
                if f not in best or c > best[f][2]:
                    best[int(f)] = (k, j, c)
        self.owner = {f: (k, j) for f, (k, j, _) in best.items()}
        self.kf = np.array(sorted(self.owner))
        fp = MAPS / "fused" / model / take / "poses.npz"
        self.fused = dict(np.load(fp)) if fp.exists() else None
        self.scale_w = None              # metric scale per window (set by fit_scale)

    @lru_cache(maxsize=3)
    def window(self, k: int):
        z = np.load(self.wfiles[k])
        return {"idx": z["idx"], "c2w": z["c2w"].astype(np.float64), "depth": z["depth"], "conf": z["conf"]}

    def depth(self, i: int):
        k, j = self.owner[i]
        w = self.window(k)
        return w["depth"][j].astype(np.float32), w["conf"][j].astype(np.float32), k

    def points(self, i: int, stride: int = 4, conf_drop: float = 30.0, border: float = 0.03):
        """Camera-frame points (window units) of keyframe i, confidence-filtered, image border dropped."""
        d, c, k = self.depth(i)
        H = d.shape[0]
        b = int(border * H)
        vv, uu = np.mgrid[b:H - b:stride, b:H - b:stride]
        z = d[vv, uu]
        cf = c[vv, uu]
        ok = np.isfinite(z) & (z > 0)
        if ok.sum() > 10 and conf_drop > 0:
            ok &= cf >= np.percentile(cf[ok], conf_drop)
        u, v, z = uu[ok], vv[ok], z[ok]
        X = np.stack([(u - self.K[0, 2]) / self.K[0, 0] * z, (v - self.K[1, 2]) / self.K[1, 1] * z, z], 1)
        return X.astype(np.float64), k

    def depth_at(self, i: int, uv: np.ndarray):
        import cv2
        d, _, _ = self.depth(i)
        m = cv2.remap(d, uv[:, 0].astype(np.float32).reshape(-1, 1), uv[:, 1].astype(np.float32).reshape(-1, 1),
                      cv2.INTER_NEAREST, borderValue=np.nan)
        return m.ravel()

    def fit_scale(self, tags_df):
        """Metric scale per window from the tags in its keyframes; rows for the table map_scale."""
        import cv2
        Kd, Dd, W = C.camera_K()
        rows = []
        if tags_df is None or len(tags_df) == 0:
            self.scale_w = np.full(len(self.wfiles), np.nan)
            return rows
        T = tags_df[(tags_df["take"] == self.take) & (tags_df.reproj_px < 2.0) & (tags_df.view_deg < 70)]
        vf2i = {int(v): i for i, v in enumerate(self.vf)}
        for _, r in T.iterrows():
            i = vf2i.get(int(r.frame))
            if i is None or i not in self.owner:
                continue
            uv = cv2.undistortPoints(np.array([[[r.u, r.v]]], np.float64), Kd, Dd, P=Kd).reshape(1, 2)
            uv = uv * self.res / W
            dz = float(self.depth_at(i, uv)[0])
            if np.isfinite(dz) and dz > 0:
                rows.append(dict(take=self.take, model=self.model, window=self.owner[i][0], keyframe=i,
                                 id=int(r.id), tag_z=float(r.tz), map_z=dz, ratio=float(r.tz) / dz))
        s = np.full(len(self.wfiles), np.nan)
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            glob = float(df.ratio.median())
            for k in range(len(self.wfiles)):
                q = df[df.window == k].ratio
                s[k] = float(q.median()) if len(q) >= 3 else glob
        self.scale_w = s
        return rows

    def metric(self, k: int) -> float:
        s = self.scale_w[k] if self.scale_w is not None else np.nan
        return float(s) if np.isfinite(s) else 1.0

    def pose_metric(self, i: int):
        """Camera->world (fused trajectory) in metres for keyframe i, or None without fused poses."""
        if self.fused is None:
            return None
        kf = list(self.fused["keyframes"])
        if i not in kf:
            return None
        n = kf.index(i)
        k = self.owner[i][0]
        kappa = self.metric(k) / float(self.fused["window_s"][k])        # global units -> m
        T = self.fused["c2w"][n].copy()
        T[:3, 3] *= kappa
        return T

    @lru_cache(maxsize=1)
    def trajectory(self):
        """(t_host, c2w metric) of the keyframes with fused poses, using one global metric factor (median)."""
        if self.fused is None:
            return None
        kf = self.fused["keyframes"]
        kap = np.array([self.metric(self.owner[int(i)][0]) / float(self.fused["window_s"][self.owner[int(i)][0]])
                        for i in kf])
        kappa = float(np.median(kap))
        T = self.fused["c2w"].copy()
        T[:, :3, 3] *= kappa
        return self.t[kf], T, kappa


def interp_poses(t_kf, T_kf, t):
    """Camera->world at times t (slerp rotation, linear translation); nan outside the keyframe span."""
    from scipy.spatial.transform import Rotation, Slerp
    t = np.atleast_1d(t)
    o = np.argsort(t_kf)
    t_kf, T_kf = t_kf[o], T_kf[o]
    keep = np.r_[True, np.diff(t_kf) > 0]
    t_kf, T_kf = t_kf[keep], T_kf[keep]
    sl = Slerp(t_kf, Rotation.from_matrix(T_kf[:, :3, :3]))
    tc = np.clip(t, t_kf[0], t_kf[-1])
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3] = sl(tc).as_matrix()
    for j in range(3):
        out[:, j, 3] = np.interp(tc, t_kf, T_kf[:, j, 3])
    out[(t < t_kf[0]) | (t > t_kf[-1])] = np.nan
    return out
