"""F13: outdoors on grass. Outdoor bench approach (outdoor_course/01_D2, 65-79 s after the first radar frame):
video frame, range-time raster with the camera truth (arc_r_fov), and the mean power profile against an indoor
approach stretch (indoor_course/01, 6-20 s). Text-free rasters + JSON."""
from __future__ import annotations

import cv2
import numpy as np

from .. import common as C, radar as R
from . import common as PC

OUT = ("outdoor_course", "outdoor_course/01_D2_tcc-1d-40mm", 65.0, 79.0)
IND = ("indoor_course", "indoor_course/01_D1_tcc-1d-40mm", 6.0, 20.0)
RLIM = (0.2, 5.0)


def load(spec):
    take, key, a, b = spec
    rec = [r for r in C.recordings(take) if r.key == key][0]
    rd = C.load_radar(rec)
    t = rd.t - rd.t[0]
    sel = np.flatnonzero((t >= a) & (t < b))
    x = rd.cube[sel[0]:sel[-1] + 1].astype(np.complex128)
    P = (np.abs(x) ** 2).sum(1).mean(1)
    return rec, rd, t, sel, P


def run():
    rec, rd, t, sel, P = load(OUT)
    r_ax = rd.range_m - R.range_bias()
    ts = t[sel]
    dt = float(np.median(np.diff(t)))
    use = np.flatnonzero((r_ax >= RLIM[0]) & (r_ax <= RLIM[1]))
    a01 = PC.db_over_median(P)[:, use]
    rgb = PC.to_rgb(a01.T[::-1])
    rgb = cv2.resize(rgb, (rgb.shape[1], rgb.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    PC.save_png("f13_raster", rgb)
    db = float(np.median(np.diff(r_ax)))
    T = C.load_table("truth_frames")
    g = T[T.rec == rec.key].sort_values("f")
    tr = np.full(len(t), np.nan)
    m = min(len(t), len(g))
    tr[:m] = g.arc_r_fov.to_numpy()[:m]
    tr = tr[sel]

    # video frame in the middle of the stretch
    tmid = float(rd.t[0] + (OUT[2] + OUT[3]) / 2)
    tv = float(C.sync_of(OUT[0]).host_to_video(tmid))
    cap = cv2.VideoCapture(str(C.DATA / "takes" / OUT[0] / "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, tv * 1000)
    ok, fr = cap.read()
    cap.release()
    assert ok
    h, w = fr.shape[:2]
    s = 360 / max(h, w)
    fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    PC.save_png("f13_video", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))

    # profiles: mean over the stretch of the frame's dB over its own median
    def prof(spec):
        _, rd2, _, sel2, P2 = load(spec)
        L = 10 * np.log10(np.maximum(P2, 1e-15))
        Lm = L - np.median(L[:, 8:110], 1, keepdims=True)
        rr = rd2.range_m - R.range_bias()
        k = (rr >= 0.2) & (rr <= 4.1)
        return rr[k], Lm.mean(0)[k]
    ro, po = prof(OUT)
    ri, pi = prof(IND)
    half = po.max() / 2
    inb = po >= half
    i0 = int(np.argmax(po))
    lo = i0
    while lo > 0 and inb[lo - 1]:
        lo -= 1
    hi = i0
    while hi < len(po) - 1 and inb[hi + 1]:
        hi += 1
    PC.save_json("f13_dados", dict(
        rec_out=OUT[1], rec_in=IND[1], t_out=[OUT[2], OUT[3]], t_in=[IND[2], IND[3]],
        raster="f13_raster", video="f13_video", video_time_s=float((tmid - rd.t[0])),
        video_px=[fr.shape[1], fr.shape[0]],
        xlim=[float(ts[0] - dt / 2), float(ts[-1] + dt / 2)],
        ylim=[float(r_ax[use[0]] - db / 2), float(r_ax[use[-1]] + db / 2)],
        truth=[[float(a), float(b)] for a, b in zip(ts, tr) if np.isfinite(b)],
        profile_out=[[float(a), float(b)] for a, b in zip(ro, po)],
        profile_in=[[float(a), float(b)] for a, b in zip(ri, pi)],
        out_peak_db=float(po.max()), out_peak_r=float(ro[i0]), in_max_db=float(pi.max()),
        out_band=[float(ro[lo]), float(ro[hi])], unit="dB over frame median"))
    print(ro[i0], po.max(), pi.max(), ro[lo], ro[hi], (tr > 5).sum(), np.isfinite(tr).sum())


if __name__ == "__main__":
    run()
