"""Per-frame radar motion (dx, dz, dyaw) in the previous radar plane for the grid: none / imu (phone orientation,
yaw only) / imu_speed (yaw + an assumed constant forward speed) / oracle (fused map trajectory, metric)."""

from __future__ import annotations

import numpy as np

from . import common as C


def _rel(Ra, Rb):
    """Yaw (+ right) of camera b relative to camera a (both camera->world)."""
    Rr = np.einsum("nji,njk->nik", Ra, Rb)            # Ra^T Rb : cam_b -> cam_a
    f = Rr[:, :, 2]
    return np.arctan2(f[:, 0], f[:, 2]), Rr


def deltas(F, mode: str, speed: float = 1.0):
    n = F.n
    out = np.zeros((n, 3))
    if n < 2 or mode == "none":
        return out
    if mode in ("imu", "imu_speed"):
        from . import imu
        try:
            R = imu.cam_rotations(F.take, F.t)
        except Exception:
            return np.full((n, 3), np.nan)
        yaw, _ = _rel(R[:-1], R[1:])
        out[1:, 2] = yaw
        if mode == "imu_speed":                         # assumed walking speed while the IMU says 'walking'
            out[1:, 1] = speed * np.diff(F.t) * imu.moving(F.take, F.t[1:])
        return out
    if mode == "oracle":
        tr = oracle_trajectory(F.take)
        if tr is None:
            return np.full((n, 3), np.nan)
        from .maps import interp_poses
        T = interp_poses(tr[0], tr[1], F.t)
        Ta, Tb = T[:-1], T[1:]
        yaw, _ = _rel(Ta[:, :3, :3], Tb[:, :3, :3])
        d = np.einsum("nji,nj->ni", Ta[:, :3, :3], Tb[:, :3, 3] - Ta[:, :3, 3])   # in camera a
        out[1:, 0], out[1:, 1], out[1:, 2] = d[:, 0], d[:, 2], yaw
        out[~np.isfinite(out).all(1)] = np.nan
        return out
    raise ValueError(mode)


_TRAJ = {}


def oracle_trajectory(take: str, model: str = "pi3x"):
    if take in _TRAJ:
        return _TRAJ[take]
    tr = None
    if take in C.CARRIED:
        from .maps import TakeMaps
        try:
            tm = TakeMaps(take, model)
            tags = C.load_table("tags", required=False)
            tm.fit_scale(tags)
            r = tm.trajectory()
            tr = None if r is None else (r[0], r[1])
        except FileNotFoundError:
            tr = None
    _TRAJ[take] = tr
    return tr
