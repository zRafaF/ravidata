"""Oblique (isometric-like) orthographic view of a coloured point cloud, as a text-free raster, plus a projector for
overlays (so Typst can draw marks at 3D positions). Coordinates in: X lateral (right), Y forward, Z up (metres)."""
from __future__ import annotations

import numpy as np


class Iso:
    def __init__(self, P, yaw_deg=-35.0, elev_deg=32.0, W=820, H=560, pad=0.04, extent=None):
        a, e = np.radians(yaw_deg), np.radians(elev_deg)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        # after Rz (turn about the vertical) and Rx (tilt), screen x = x, screen y = z, depth = y (far = large)
        self.R = Rx @ Rz
        S = (extent if extent is not None else P) @ self.R.T
        lo, hi = np.percentile(S[:, [0, 2]], 0.5, 0), np.percentile(S[:, [0, 2]], 99.5, 0)
        span = hi - lo
        self.lo = lo - pad * span; self.hi = hi + pad * span
        sx = W / (self.hi[0] - self.lo[0]); sy = H / (self.hi[1] - self.lo[1]); self.s = min(sx, sy)
        self.W = int(round((self.hi[0] - self.lo[0]) * self.s)); self.H = int(round((self.hi[1] - self.lo[1]) * self.s))

    def project(self, P):
        """(n, 3) -> (n, 2) image coordinates normalised to 0..1 (x right, y down) and depth."""
        S = np.atleast_2d(P) @ self.R.T
        u = (S[:, 0] - self.lo[0]) / (self.hi[0] - self.lo[0])
        v = 1 - (S[:, 2] - self.lo[1]) / (self.hi[1] - self.lo[1])
        return np.c_[u, v], S[:, 1]

    def render(self, P, rgb, splat=2):
        uv, depth = self.project(P)
        x = (uv[:, 0] * (self.W - 1)).astype(int); y = (uv[:, 1] * (self.H - 1)).astype(int)
        m = (x >= 0) & (x < self.W) & (y >= 0) & (y < self.H)
        o = np.argsort(-depth[m])                      # far first, near drawn last
        x, y, c = x[m][o], y[m][o], rgb[m][o]
        img = np.full((self.H, self.W, 3), 255, np.uint8)
        for dx in range(splat):
            for dy in range(splat):
                img[np.clip(y + dy, 0, self.H - 1), np.clip(x + dx, 0, self.W - 1)] = c
        return img
