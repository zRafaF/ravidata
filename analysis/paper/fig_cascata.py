"""Carried walk, range-Doppler recording: range-time maps with (a) power and (b) power coloured by the radial speed
(closing = blue, receding = red, ~0 = dark), and (c) the walking speed from the camera map. Overlays (JSON): the truth
(nearest map obstacle in the beam) and the radar's height above the floor. usage:
python -m analysis.paper.fig_cascata [take] [recording index] [start s] [duration s]"""
from __future__ import annotations

import sys

import numpy as np

from .. import common as C, radar as R
from ..maps import TakeMaps, interp_poses
from . import common as P

RMAX = 4.0
VMAX = 1.0           # |radial speed| giving the full colour (m/s)


def run(take="indoor_course", rec_i=2, start=10.0, dur=20.0, name="fc"):
    from scipy.ndimage import uniform_filter1d
    rec = C.recordings(take)[rec_i]; rd = C.load_radar(rec)
    t = rd.t; rng = rd.range_m - R.range_bias(); v = rd.v_axis
    w = (t >= t[0] + start) & (t < t[0] + start + dur)
    keep = (rng >= 0.2) & (rng <= RMAX)
    Pd = (np.abs(rd.cube[w].astype(np.complex128)) ** 2).mean(2)[:, :, keep]   # (n, D, B)
    Pt = Pd.sum(1)
    if v is None:                                                          # 1D config: no Doppler panel
        v = np.zeros(Pd.shape[1])
    vm = (Pd * v[None, :, None]).sum(1) / np.maximum(Pt, 1e-30)           # power-weighted radial speed
    a = P.db_over_median(Pt, lo=3, hi=25, sl=slice(None))                  # 0..1
    # (a) power, as in the other range-time figures
    P.save_png(f"{name}_pot", P.to_rgb(np.flipud(a.T)))
    # (b) colour = radial speed, opacity = power (on white)
    s = np.clip(vm / VMAX, -1, 1)
    dark = np.array([0.15, 0.15, 0.18]); blue = np.array([0.04, 0.42, 0.85]); red = np.array([0.80, 0.20, 0.12])
    col = np.where(s[..., None] < 0, dark + (-s[..., None]) * (blue - dark), dark + s[..., None] * (red - dark))
    rgb = 1 - (a[..., None] ** 0.7) * (1 - col)
    P.save_png(f"{name}_dop", (np.flipud(np.transpose(rgb, (1, 0, 2))) * 255).astype(np.uint8))
    # truth and radar height
    Ft = C.load_table("truth_frames"); Ft = Ft[Ft.rec == rec.key].sort_values("t")
    m = (Ft.t >= t[w][0]) & (Ft.t <= t[w][-1])
    truth = np.c_[Ft.t[m] - t[w][0], Ft.arc_r[m]]
    height = np.c_[Ft.t[m] - t[w][0], uniform_filter1d(-Ft.floor_h[m].to_numpy(), 9)]
    # walking speed from the fused map trajectory (forward component, 0.5 s smoothing)
    tm = TakeMaps(take, "pi3x"); tm.fit_scale(C.load_tags()); tr = tm.trajectory()
    ts = np.arange(t[w][0], t[w][-1], 0.1)
    Tt = interp_poses(tr[0], tr[1], ts)
    vel = uniform_filter1d(np.gradient(Tt[:, :3, 3], ts, axis=0), 5, axis=0)
    speed = np.c_[ts - t[w][0], (vel * Tt[:, :3, 2]).sum(1)]
    # how the radial speed splits by range while walking (> 0.4 m/s)
    sp = np.interp(t[w], ts, speed[:, 1])
    walk = sp > 0.4
    rr = rng[keep]
    bands = []
    for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)):
        b = (rr >= lo) & (rr < hi)
        W = Pt[walk][:, b]
        bands.append(dict(lo=lo, hi=hi, v_med=float(np.sum(W * vm[walk][:, b]) / W.sum())))
    P.save_json(f"{name}_dados", dict(rec=rec.key, start=start, dur=dur, xlim=[0, float(t[w][-1] - t[w][0])],
                                      ylim=[float(rr[0]), float(rr[-1])], vmax=VMAX, truth=truth, height=height,
                                      speed=speed[::2], bands=bands, walking_share=float(walk.mean())))
    print(rec.key, "frames", int(w.sum()), "walking", round(float(walk.mean()), 2), bands)


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "indoor_course", int(a[1]) if len(a) > 1 else 2, float(a[2]) if len(a) > 2 else 10.0,
        float(a[3]) if len(a) > 3 else 20.0)
