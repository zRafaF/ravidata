"""The earlier firmware's energy grid (TCC1 / the legacy C pipeline), re-implemented as a candidate family.

Per frame, without any detector in front:
  1. local polar grid: every range bin (8 bins cropped at each end) puts its energy (power of one RX) into the 2 deg
     azimuth cell its interferometric angle falls in (+-60 deg); energies in the same cell add up;
  2. the cells are written into a Cartesian map (5 cm) at r (sin az, cos az): energy x gain, saturating, optionally
     with a 3 x 3 'diffusion' (neighbours get half);
  3. the map decays by decay_rate_per_sec ** dt (0.05 / s: 5 % is left after one second);
  4. cells above a threshold are clustered with DBSCAN (eps 0.15 m, min 2 cells); per cluster, the nearest cell to
     the sensor among the cluster and its 8-neighbours above max(peak / 8, threshold / 2) is the target.
Energies are in units of the frame's noise (SNR, linear) instead of q31 full scale; the threshold is in dB of that.
The C version moved the map with the IMU orientation (rotation only, fixed position): pose 'imu' here.
Differences from `grid`: no CFAR front end, no log-odds / free space, one target per cluster."""

from __future__ import annotations

import numpy as np

from . import geometry as G
from .cluster import dbscan

LEGACY_DEFAULT = dict(az_bin_deg=2.0, az_fov_deg=120.0, crop_near=8, crop_far=8, rx=1, gain=1.0,
                      decay_rate_per_sec=0.05, thr_db=15.0, eps=0.15, min_pts=2, diffusion=False, edge_frac=0.125,
                      snr_min_db=0.0, clamp_db=60.0, cell=0.05, half=3.2)


class LegacyGrid:
    def __init__(self, p=None, trig=None):
        self.p = dict(LEGACY_DEFAULT, **(p or {}))
        self.n = int(round(2 * self.p["half"] / self.p["cell"]))
        self.E = np.zeros((self.n, self.n), np.float64)          # rows: Z (0 .. 2 half), cols: X (-half .. half)
        self.trig = trig or (lambda a: (np.sin(a), np.cos(a)))

    def move(self, dx, dz, dyaw):
        if dx == 0 and dz == 0 and dyaw == 0:
            return
        import cv2
        c, s = np.cos(dyaw), np.sin(dyaw)
        cell, h = self.p["cell"], self.p["half"]
        A = np.array([[cell, 0, 0.5 * cell - h], [0, cell, 0.5 * cell - h], [0, 0, 1]])   # z centred too
        T = np.eye(3); T[:2, :2] = [[c, s], [-s, c]]; T[:2, 2] = [dx, dz]
        M = (np.linalg.inv(A) @ T @ A)[:2]
        self.E = cv2.warpAffine(self.E, M, (self.n, self.n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def update(self, r_axis, energy, az_deg, dt):
        """energy, az_deg: (B,) of this frame (energy in noise units)."""
        p = self.p
        self.E *= p["decay_rate_per_sec"] ** dt
        B = len(r_axis)
        idx = np.arange(int(p["crop_near"]), B - int(p["crop_far"]))
        e = energy[idx]
        az = az_deg[idx]
        ok = np.isfinite(az) & (np.abs(az) <= p["az_fov_deg"] / 2) & (e > 10 ** (p["snr_min_db"] / 10))
        if not ok.any():
            return
        idx, e, az = idx[ok], e[ok], az[ok]
        azq = (np.floor((az + p["az_fov_deg"] / 2) / p["az_bin_deg"]) + 0.5) * p["az_bin_deg"] - p["az_fov_deg"] / 2
        s, c = self.trig(np.radians(azq))                        # cell centre of the 2 deg polar cell
        r = r_axis[idx]
        X, Z = r * s, r * c
        cell, h = p["cell"], p["half"]
        ix = np.floor((X + h) / cell).astype(int)
        iz = np.floor((Z + h) / cell).astype(int)
        val = e * p["gain"]
        offs = [(0, 0, 1.0)] + ([(dy, dx, 0.5) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
                                if p["diffusion"] else [])
        for dy, dx, w in offs:
            jj, ii = ix + dx, iz + dy
            m = (jj >= 0) & (jj < self.n) & (ii >= 0) & (ii < self.n)
            np.add.at(self.E, (ii[m], jj[m]), val[m] * w)
        np.minimum(self.E, 10 ** (p["clamp_db"] / 10), out=self.E)

    def obstacles(self):
        p = self.p
        thr = 10 ** (p["thr_db"] / 10)
        iz, ix = np.nonzero(self.E > thr)
        if len(iz) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        cell, h = p["cell"], p["half"]
        X = (ix + 0.5) * cell - h
        Z = (iz + 0.5) * cell - h
        lab = dbscan(np.stack([X, Z], 1), p["eps"] + 1e-6, int(p["min_pts"]))
        out_r, out_a, out_s = [], [], []
        for c in np.unique(lab[lab >= 0]):
            m = lab == c
            peak = self.E[iz[m], ix[m]].max()
            t_edge = max(peak * p["edge_frac"], thr / 2)
            best = None
            for a_, b_ in zip(iz[m], ix[m]):
                sub = self.E[max(a_ - 1, 0):a_ + 2, max(b_ - 1, 0):b_ + 2]
                zz, xx = np.nonzero(sub >= t_edge)
                zz, xx = zz + max(a_ - 1, 0), xx + max(b_ - 1, 0)
                Xs, Zs = (xx + 0.5) * cell - h, (zz + 0.5) * cell - h
                if len(zz) == 0:
                    continue
                d = Xs * Xs + Zs * Zs
                k = int(np.argmin(d))
                if best is None or d[k] < best[0]:
                    best = (d[k], Xs[k], Zs[k])
            if best is not None:
                out_r.append(np.sqrt(best[0])); out_a.append(np.degrees(np.arctan2(best[1], best[2])))
                out_s.append(10 * np.log10(peak))
        return np.array(out_r), np.array(out_a), np.array(out_s)


def run_legacy(F, p, trig=None, poses=None):
    """Detections of the legacy grid over a recording (features F: power per RX not kept, so the combined power and
    the calibrated per-bin azimuth are used)."""
    from . import poses as PS
    from .radar import angles_at
    g = LegacyGrid({k[2:]: v for k, v in p.items() if k.startswith("L_")}, trig)
    n, B = F.P.shape
    f = np.repeat(np.arange(n), B)
    b = np.tile(np.arange(B), n)
    az, el, _ = angles_at(F, f, b, span=0)
    E = F.P / np.maximum(F.noise[:, None], 1e-300)
    if p.get("hgate"):                                       # the firmware also kept a height per cell
        from .detectors import heights
        h = heights(F, f, np.tile(F.r, n), az, el + p.get("el_off", 0.0), p.get("radar_h", 1.1))
        bad = np.isfinite(h) & ((h <= p.get("h_min", 0.15)) | (h >= p.get("h_max", 2.0)))
        E = np.where(bad.reshape(n, B), 0.0, E)
    az = az.reshape(n, B)
    deltas = None
    if p.get("pose", "none") != "none":
        deltas = poses if poses is not None else PS.deltas(F, p["pose"], p.get("speed", 1.0))
    dt = np.r_[np.median(np.diff(F.t)) if n > 1 else 0.03, np.diff(F.t)]
    out = {k: [] for k in ("f", "r", "az", "el", "score", "snr_db")}
    for i in range(n):
        if deltas is not None and i > 0 and np.all(np.isfinite(deltas[i])):
            g.move(*deltas[i])
        g.update(F.r, E[i], az[i], float(np.clip(dt[i], 0.001, 1.0)))
        r, a, s = g.obstacles()
        out["f"].append(np.full(len(r), i, np.int32)); out["r"].append(r); out["az"].append(a)
        out["el"].append(np.zeros(len(r))); out["score"].append(s); out["snr_db"].append(s)
    return {k: np.concatenate(v) for k, v in out.items()}
