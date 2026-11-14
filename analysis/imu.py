"""Phone IMU (Sensor Logger CSVs) on the host clock."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .common import DATA, sync_of


@lru_cache(maxsize=8)
def load(take: str, sensor: str = "Gravity"):
    """(t_host, values (N, k), columns). Columns follow the CSV (x, y, z re-ordered for vectors)."""
    import pandas as pd
    p = DATA / "takes" / take / "imu" / f"{sensor}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    s = sync_of(take)
    t = s.phone_to_host(df["time"].to_numpy(np.float64) / 1e9)
    cols = [c for c in ("x", "y", "z") if c in df] or [c for c in df.columns if c not in ("time", "seconds_elapsed")]
    v = df[cols].to_numpy(np.float64)
    o = np.argsort(t)
    return t[o], v[o], cols


def at(take: str, t_host, sensor: str = "Gravity", smooth_s: float = 0.0):
    """Sensor values interpolated at t_host (optionally box-smoothed over smooth_s first)."""
    d = load(take, sensor)
    if d is None:
        return None
    t, v, _ = d
    if smooth_s > 0:
        dt = np.median(np.diff(t))
        k = max(1, int(round(smooth_s / dt)))
        ker = np.ones(k) / k
        v = np.stack([np.convolve(v[:, j], ker, "same") for j in range(v.shape[1])], 1)
    t_host = np.atleast_1d(np.asarray(t_host, float))
    return np.stack([np.interp(t_host, t, v[:, j]) for j in range(v.shape[1])], 1)


def orientation_at(take: str, t_host):
    """Rotation (scipy) device->world (Android game rotation vector, z up) at t_host."""
    from scipy.spatial.transform import Rotation, Slerp
    import pandas as pd
    p = DATA / "takes" / take / "imu" / "Orientation.csv"
    df = pd.read_csv(p)
    t = sync_of(take).phone_to_host(df["time"].to_numpy(np.float64) / 1e9)
    o = np.argsort(t)
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(np.float64)[o]
    t = t[o]
    keep = np.r_[True, np.diff(t) > 0]
    sl = Slerp(t[keep], Rotation.from_quat(q[keep]))
    return sl(np.clip(np.atleast_1d(t_host), t[keep][0], t[keep][-1]))


# device -> camera axes (x_cam = x_dev, y_cam = -y_dev, z_cam = -z_dev)
DEV2CAM = np.diag([1.0, -1.0, -1.0])


def cam_rotations(take: str, t_host):
    """Camera->world rotation matrices (N, 3, 3) from the phone orientation (world z up)."""
    R = orientation_at(take, t_host).as_matrix()          # device -> world
    return R @ DEV2CAM.T                                   # camera -> device -> world


def moving(take: str, t_host, win_s: float = 1.0, thr: float = 0.35):
    """1 where the phone is being carried (walking): rolling std of |linear acceleration| over win_s > thr m/s^2."""
    d = load(take, "Accelerometer")
    if d is None:
        return np.zeros(len(np.atleast_1d(t_host)))
    t, v, _ = d
    a = np.linalg.norm(v, axis=1)
    dt = np.median(np.diff(t))
    k = max(3, int(round(win_s / dt)))
    ker = np.ones(k) / k
    m1 = np.convolve(a, ker, "same")
    m2 = np.convolve(a * a, ker, "same")
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    return (np.interp(np.atleast_1d(t_host), t, sd) > thr).astype(float)
