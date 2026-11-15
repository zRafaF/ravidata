"""F04: still radar, tagged person walking. Exports the moving-power range-time raster (text-free), the tag sightings
over it, the range-residual histogram and the azimuth scatter (JSON series for Typst)."""
from __future__ import annotations

import numpy as np

from .. import common as C, radar as R
from . import common as P

REC = ("static_pass2", "07_S2_tcc-1d-40mm")
DUR = 10.0
RMAX = 3.2


def run():
    rec = [r for r in C.recordings(REC[0]) if r.file.stem == REC[1]][0]
    rd = C.load_radar(rec)
    t, rng = rd.t, rd.range_m - R.range_bias()
    tg = C.load_tags()
    q = tg[(tg["take"] == REC[0]) & tg.id.isin([0, 1]) & (tg.kind == "person") & (tg.reproj_px < 2)
           & (tg.t_host >= t[0]) & (tg.t_host <= t[-1])].copy()
    ts = q.t_host.to_numpy()
    t0s = np.arange(t[0], t[-1] - DUR, 0.25)
    n = [((ts >= a) & (ts < a + DUR)).sum() for a in t0s]
    t0 = float(t0s[int(np.argmax(n))])
    x = rd.cube.astype(np.complex128)
    x = x - x.mean(0, keepdims=True)
    Pm = (np.abs(x) ** 2).sum(1).mean(1)
    w = (t >= t0) & (t < t0 + DUR)
    keep = (rng >= 0.2) & (rng <= RMAX)
    A = P.db_over_median(Pm[w], lo=3, hi=25)[:, keep]       # (frames, bins)
    img = np.flipud(A.T)                                     # row 0 = far range
    # upsample in time so a frame is at least a few pixels wide
    img = np.repeat(img, 2, axis=1)
    P.save_png("f04_raster", P.to_rgb(img))
    qw = q[(q.t_host >= t0) & (q.t_host < t0 + DUR)]
    r_tag = np.sqrt(qw.tx ** 2 + qw.ty ** 2 + qw.tz ** 2).to_numpy()
    P.save_json("f04_raster", dict(xlim=[0, DUR], ylim=[float(rng[keep][0]), float(rng[keep][-1])], xunit="s", yunit="m",
                                   rec=rec.key, t0_s_after_first_frame=t0 - t[0], n_tag=int(len(qw)),
                                   tags=np.c_[qw.t_host.to_numpy() - t0, r_tag]))
    # (b), (c): calib_moving (all recordings)
    m = C.load_table("calib_moving")
    res = m.range_resid_m.to_numpy() * 100
    med = float(np.median(res)); mad = float(np.median(np.abs(res - med)))
    edges = np.arange(-30, 30.1, 2.0)
    cnt, _ = np.histogram(res, edges)
    cal = R.angle_calib()["az"]
    az = R.phase_to_angle(m.ph_az.to_numpy(), cal["sign"], cal["offset_rad"])
    azt = m.az_true.to_numpy()
    mar = float(np.median(np.abs(az - azt)))
    rs = np.random.default_rng(0)
    idx = rs.choice(len(m), min(400, len(m)), replace=False)
    P.save_json("f04_hist", dict(edges_cm=edges, counts=cnt, n=int(len(m)), median_cm=med, mad_cm=mad,
                                 n_out=int(((res < edges[0]) | (res > edges[-1])).sum())))
    P.save_json("f04_az", dict(true=azt[idx], meas=az[idx], n=int(len(m)), n_shown=int(len(idx)), median_abs_resid_deg=mar,
                               bias_median_deg=float(np.median(az - azt))))
    print("window", t0 - t[0], "tags", len(qw), "med/MAD cm", med, mad, "az", mar, "n", len(m), "out", ((res < -30) | (res > 30)).sum())


if __name__ == "__main__":
    run()
