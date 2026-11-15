"""F09 (b): range-time raster of a floor-tag window (indoor_course/02, 0.5-10.5 s) and the camera height above the floor
tags (|tag normal . tag position|) as points in data coordinates. The sketch (a) is native Typst."""
from __future__ import annotations

import cv2
import numpy as np

from .. import common as C, radar as R
from . import common as P

KEY = "02_D1_tcc-2d-40mm-12fps"
T0, T1 = 0.5, 10.5
YMAX = 2.6


def run():
    rec = [r for r in C.recordings("indoor_course") if r.file.stem == KEY][0]
    rd = C.load_radar(rec)
    t, rng = rd.t, rd.range_m - R.range_bias()
    Pw = (np.abs(rd.cube.astype(np.complex64)) ** 2).sum(1).mean(1)
    w = (t >= t[0] + T0) & (t < t[0] + T1)
    keep = (rng >= 0.2) & (rng <= YMAX)
    A = P.db_over_median(Pw[w], lo=3, hi=30)[:, keep]
    P.save_png("f09_raster", P.to_rgb(np.repeat(np.flipud(A.T), 2, axis=1)))
    tg = C.load_tags()
    q = tg[(tg["take"] == "indoor_course") & (tg.kind == "floor_tag") & (tg.reproj_px < 2)
           & (tg.t_host >= t[0] + T0) & (tg.t_host < t[0] + T1)]
    n = np.array([cv2.Rodrigues(np.array([a, b, c]))[0][:, 2] for a, b, c in zip(q.rx, q.ry, q.rz)])
    h = np.abs((n * q[["tx", "ty", "tz"]].to_numpy()).sum(1))
    ok = (h > 0.3) & (h < YMAX)
    tt = q.t_host.to_numpy()[ok] - (t[0] + T0)
    # contrast of the radar at the camera height vs the median of the window (how well the line follows the return)
    fi = np.clip(np.searchsorted(t, q.t_host.to_numpy()[ok]), 0, len(t) - 1)
    bi = np.clip(np.round((h[ok] + R.range_bias()) / 0.04).astype(int), 1, 126)
    L = 10 * np.log10(Pw + 1e-15)
    lev = np.maximum.reduce([L[fi, bi - 1], L[fi, bi], L[fi, bi + 1]]) - np.median(L[:, 8:110], 1)[fi]
    P.save_json("f09_raster", dict(xlim=[0, T1 - T0], ylim=[float(rng[keep][0]), float(rng[keep][-1])], rec=rec.key,
                                   t0_after_first_frame=T0, n=int(ok.sum()), median_h_m=float(np.median(h[ok])),
                                   h_min_m=float(h[ok].min()), h_max_m=float(h[ok].max()),
                                   median_db_over_median=float(np.median(lev)), points=np.c_[tt, h[ok]]))
    print("n", ok.sum(), "h", h[ok].min(), np.median(h[ok]), h[ok].max(), "lev dB", np.median(lev))


if __name__ == "__main__":
    run()
