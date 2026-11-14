"""Frames and conventions.

Camera frame (OpenCV): x right, y down, z forward (optical axis). The radar frame uses the same axes (the module
faces the same way as the lens) with its origin at ``t_rc`` in camera coordinates (lens 4 cm from the chip; the
sign / dy / dz are fitted by the ``extrinsic`` step). Radar polar coordinates:

    r  = |p|                     slant range (what the radar measures)
    az = atan2(x, z)             azimuth, + to the right
    el = atan2(-y, hypot(x, z))  elevation, + up

The scoring plane is the radar's own range-azimuth plane projected flat: X = r sin(az), Z = r cos(az) (the slant
range is kept, elevation is not; a 1 TX / 3 RX radar measures r and az best, el poorly)."""

from __future__ import annotations

import numpy as np

HFOV_HALF_DEG = 28.0          # camera half FOV used for truth (display HFOV 58.7 deg, minus a margin)
SECTOR_DEG = 4.0              # azimuth sector for "nearest obstacle per direction"
N_SECT = int(round(2 * HFOV_HALF_DEG / SECTOR_DEG))
R_MIN, R_MAX = 0.30, 4.10     # HPF corner ~0.26 m; usable range 4.10 m


def to_radar(p_cam: np.ndarray, t_rc) -> np.ndarray:
    return np.asarray(p_cam, float) - np.asarray(t_rc, float)


def polar(p: np.ndarray):
    p = np.asarray(p, float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    az = np.degrees(np.arctan2(x, z))
    el = np.degrees(np.arctan2(-y, np.hypot(x, z)))
    return r, az, el


def plane(r, az_deg):
    a = np.radians(np.nan_to_num(np.asarray(az_deg, float)))
    r = np.asarray(r, float)
    return r * np.sin(a), r * np.cos(a)


def sector_of(az_deg) -> np.ndarray:
    """Sector index 0..N_SECT-1, -1 outside the camera FOV."""
    az = np.asarray(az_deg, float)
    s = np.floor((az + HFOV_HALF_DEG) / SECTOR_DEG).astype(int)
    s[(s < 0) | (s >= N_SECT) | ~np.isfinite(az)] = -1
    return s


def sector_centres() -> np.ndarray:
    return -HFOV_HALF_DEG + SECTOR_DEG * (np.arange(N_SECT) + 0.5)


def up_from_gravity(g_dev) -> np.ndarray:
    """Phone gravity sensor (device x right, y up the screen, z out of the screen; reads +9.8 along 'up')
    -> unit 'up' vector in camera coordinates (x_cam = x_dev, y_cam = -y_dev, z_cam = -z_dev)."""
    g = np.asarray(g_dev, float)
    u = np.stack([g[..., 0], -g[..., 1], -g[..., 2]], -1)
    return u / np.linalg.norm(u, axis=-1, keepdims=True)


def floor_height(h: np.ndarray, w: np.ndarray | None = None, lo=-2.5, hi=-0.25, bin_m=0.04) -> float:
    """Height of the floor below the camera (negative) = densest level among points below the camera."""
    m = np.isfinite(h) & (h > lo) & (h < hi)
    if m.sum() < 50:
        return np.nan
    hist, edges = np.histogram(h[m], bins=np.arange(lo, hi + bin_m, bin_m), weights=None if w is None else w[m])
    hist = np.convolve(hist, [0.25, 0.5, 0.25], "same")
    k = int(np.argmax(hist))
    sel = m & (np.abs(h - (edges[k] + bin_m / 2)) < 1.5 * bin_m)
    return float(np.median(h[sel]))


def nearest_per_sector(r, az, kind_mask=None):
    """min r per sector (nan where empty) and the index of that point."""
    s = sector_of(az)
    out = np.full(N_SECT, np.nan)
    arg = np.full(N_SECT, -1)
    m = s >= 0
    if kind_mask is not None:
        m &= kind_mask
    if not m.any():
        return out, arg
    idx = np.flatnonzero(m)
    order = idx[np.lexsort((r[idx], s[idx]))]
    ss = s[order]
    first = np.r_[True, ss[1:] != ss[:-1]]
    out[ss[first]] = r[order[first]]
    arg[ss[first]] = order[first]
    return out, arg
