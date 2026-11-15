"""Radar-centric occupancy grid (bird's-eye view of the radar's range-azimuth plane), log-odds.

Per frame: (1) move the grid by the pose change (none / IMU yaw / IMU yaw + assumed speed / oracle map pose),
(2) decay toward 0, (3) inverse sensor model: a Gaussian 'occupied' stamp at every detection (sigma along range
and across azimuth) and 'free' along its ray (ray marching from r_min to r - margin), (4) clamp. Obstacles are
read from the grid after optional blur / morphology, hysteresis thresholds and a minimum component size: per
component and 4 deg azimuth sector, its nearest cell. sin/cos come from `trig` (float64 by default)."""

from __future__ import annotations

import numpy as np

from . import geometry as G

GRID_DEFAULT = dict(cell=0.05, x_half=2.5, z_max=4.5, l_occ=0.85, l_free=0.4, l_min=-2.0, l_max=3.5,
                    decay=0.9, sigma_r=0.06, sigma_az=8.0, free_margin=0.15, free=True, snr_weight=True,
                    blur=0.0, morph="none", morph_k=3, thr_hi=1.5, thr_lo=0.8, min_cells=3, extract="component")


def float_trig(a_rad):
    return np.sin(a_rad), np.cos(a_rad)


class Grid:
    def __init__(self, p: dict | None = None, trig=float_trig):
        self.p = dict(GRID_DEFAULT, **(p or {}))
        c = self.p["cell"]
        self.nx = int(round(2 * self.p["x_half"] / c))
        self.nz = int(round(self.p["z_max"] / c))
        self.L = np.zeros((self.nz, self.nx), np.float32)
        self.trig = trig
        xs = (np.arange(self.nx) + 0.5) * c - self.p["x_half"]
        zs = (np.arange(self.nz) + 0.5) * c
        self.Xc, self.Zc = np.meshgrid(xs, zs)
        self.Rc = np.hypot(self.Xc, self.Zc)
        self.Ac = np.degrees(np.arctan2(self.Xc, self.Zc))
        self.sect = np.floor((self.Ac + 90) / G.SECTOR_DEG).astype(int)   # sectors over +-90 deg

    def ij(self, X, Z):
        c = self.p["cell"]
        return np.floor(Z / c).astype(int), np.floor((X + self.p["x_half"]) / c).astype(int)

    def move(self, dx, dz, dyaw_rad):
        """New radar pose = old pose moved by (dx, dz) and turned by dyaw (in the old radar plane)."""
        if dx == 0 and dz == 0 and dyaw_rad == 0:
            return
        import cv2
        c, s = np.cos(dyaw_rad), np.sin(dyaw_rad)
        R = np.array([[c, s], [-s, c]])        # p_old = R p_new + d  (x, z) with yaw + to the right
        d = np.array([dx, dz])
        cell, xh = self.p["cell"], self.p["x_half"]
        # pixel (col, row) -> metric: X = (col + .5) c - xh, Z = (row + .5) c
        A = np.array([[cell, 0, 0.5 * cell - xh], [0, cell, 0.5 * cell], [0, 0, 1]])
        Ai = np.linalg.inv(A)
        T = np.eye(3); T[:2, :2] = R; T[:2, 2] = d
        M = (Ai @ T @ A)[:2]
        self.L = cv2.warpAffine(self.L, M, (self.nx, self.nz), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def update(self, r, az_deg, snr_db):
        p = self.p
        self.L *= p["decay"]
        if len(r) == 0:
            return
        a = np.radians(np.nan_to_num(az_deg))
        s, c = self.trig(a)
        X, Z = r * s, r * c
        cell = p["cell"]
        w = np.clip((snr_db - 6.0) / 20.0, 0.2, 1.0) if p["snr_weight"] else np.ones(len(r))
        if p["free"]:                                      # ray marching, one 'free' per cell and frame
            free_idx = []
            step = cell * 0.5
            for k in range(len(r)):
                rr = np.arange(G.R_MIN, r[k] - p["free_margin"], step)
                if len(rr) == 0:
                    continue
                i, j = self.ij(rr * s[k], rr * c[k])
                ok = (i >= 0) & (i < self.nz) & (j >= 0) & (j < self.nx)
                free_idx.append(i[ok] * self.nx + j[ok])
            if free_idx:
                u = np.unique(np.concatenate(free_idx))
                self.L.ravel()[u] -= p["l_free"]
        sl = np.maximum(r * np.radians(p["sigma_az"]), cell)
        sr = max(p["sigma_r"], cell / 2)
        kr = int(min(12, np.ceil(3 * max(sl.max(), sr) / cell)))
        off = np.arange(-kr, kr + 1)
        oi, oj = np.meshgrid(off, off, indexing="ij")
        ci, cj = self.ij(X, Z)
        I = ci[:, None, None] + oi[None]
        J = cj[:, None, None] + oj[None]
        Xg = (J + 0.5) * cell - p["x_half"]
        Zg = (I + 0.5) * cell
        dX, dZ = Xg - X[:, None, None], Zg - Z[:, None, None]
        dr = dX * s[:, None, None] + dZ * c[:, None, None]
        dl = dX * c[:, None, None] - dZ * s[:, None, None]
        g = np.exp(-0.5 * ((dr / sr) ** 2 + (dl / sl[:, None, None]) ** 2)) * (p["l_occ"] * w)[:, None, None]
        ok = (I >= 0) & (I < self.nz) & (J >= 0) & (J < self.nx) & (g > 0.02 * p["l_occ"])
        np.add.at(self.L.ravel(), (I[ok] * self.nx + J[ok]), g[ok].astype(np.float32))
        np.clip(self.L, p["l_min"], p["l_max"], out=self.L)

    def obstacles(self):
        """(r, az_deg, score) of the nearest cell per component (extract='component') or per component and 4 deg
        azimuth sector (extract='sector': a wide object gives one point per sector)."""
        import cv2
        p = self.p
        L = self.L
        if p["blur"] > 0:
            L = cv2.GaussianBlur(L, (0, 0), p["blur"])
        lo = (L > p["thr_lo"]).astype(np.uint8)
        if p["morph"] != "none":
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(p["morph_k"]), int(p["morph_k"])))
            lo = cv2.morphologyEx(lo, {"open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
                                       "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE}[p["morph"]], k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(lo, connectivity=8)
        if n <= 1:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        mx = np.full(n, -np.inf)
        np.maximum.at(mx, lab.ravel(), L.ravel())
        good = (mx >= p["thr_hi"]) & (stats[:, cv2.CC_STAT_AREA] >= p["min_cells"])
        good[0] = False
        m = good[lab] & (self.Rc >= G.R_MIN)
        if not m.any():
            return np.zeros(0), np.zeros(0), np.zeros(0)
        key = lab[m].astype(np.int64) * 1000 + (self.sect[m] if p["extract"] == "sector" else 0)
        r, a, sc = self.Rc[m], self.Ac[m], mx[lab[m]]
        o = np.lexsort((r, key))
        k2 = key[o]
        first = np.r_[True, k2[1:] != k2[:-1]]
        return r[o][first], a[o][first], sc[o][first]
