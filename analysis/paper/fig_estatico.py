"""Still radar, tagged person walking (static_pass2 / S2, 1D config): what the radar measures on its own against
what the camera sees. Exports the moving-power range-time raster (text-free) and, as JSON, per frame the radar's own
strongest moving return (range, interferometric azimuth; no tag is used to find it) and the tag's range / azimuth.
Panels in the thesis: (a) range-time, (b) azimuth-time, (c) top view of both tracks."""
from __future__ import annotations

import numpy as np

from .. import common as C, geometry as G, radar as R
from . import common as P

REC = ("static_pass2", "07_S2_tcc-1d-40mm")
DUR = 10.0
RMIN, RMAX = 0.35, 3.2
SNR_DB = 12.0                 # radar point kept when the moving power is this far over the frame's median
LINE_DB = 6.0                 # constant lines: bins whose median moving power over the window is this far over the median
LINE_COLS = [(10, 108, 191), (194, 87, 26), (26, 143, 90), (106, 79, 179)]   # = lib.typ BLUE, ORANGE, GREEN, PURPLE


def scene(rd_t0, lines, rng_bias):
    """The still scene from the camera (static-depth frame of the same part), with the pixels whose distance to the
    radar lies within +-4 cm of each constant line tinted in that line's colour. Returns the PNG name."""
    import cv2
    import glob
    d = C.DATA / "derived" / "static-depth-v1" / "depthpro" / REC[0]
    part = REC[1].split("_")[1]
    fs = sorted(glob.glob(str(d / f"{part}_*.npz")))
    if not fs:
        return None
    z = np.load(fs[0])
    D = z["depth"].astype(np.float64); K = z["K"]
    im = cv2.cvtColor(cv2.imread(fs[0][:-4] + ".jpg"), cv2.COLOR_BGR2RGB).astype(np.float64)
    if im.shape[:2] != D.shape:
        im = cv2.resize(im, D.shape[::-1])
    v, u = np.mgrid[0:D.shape[0], 0:D.shape[1]]
    X = (u - K[0, 2]) / K[0, 0] * D; Y = (v - K[1, 2]) / K[1, 1] * D
    r = np.sqrt(X ** 2 + Y ** 2 + D ** 2)
    out = 0.55 * im + 0.45 * 255                                         # washed out, so the tints read
    for i, lr in enumerate(lines):
        m = np.abs(r - lr) <= 0.04
        out[m] = 0.25 * im[m] + 0.75 * np.array(LINE_COLS[i % len(LINE_COLS)])
    P.save_png("fe_cena", np.clip(out, 0, 255).astype(np.uint8))
    return dict(frame=pathlib_name(fs[0]), t_video=float(z["t_video"]))


def pathlib_name(f):
    from pathlib import Path
    return Path(f).stem


def run():
    rec = [r for r in C.recordings(REC[0]) if r.file.stem == REC[1]][0]
    rd = C.load_radar(rec)
    t, rng = rd.t, rd.range_m - R.range_bias()
    tg = C.load_tags()
    q = tg[(tg["take"] == REC[0]) & tg.id.isin([0, 1]) & (tg.kind == "person") & (tg.reproj_px < 2)
           & (tg.t_host >= t[0]) & (tg.t_host <= t[-1])].copy()
    ts = q.t_host.to_numpy()
    t0s = np.arange(t[0], t[-1] - DUR, 0.25)
    t0 = float(t0s[int(np.argmax([((ts >= a) & (ts < a + DUR)).sum() for a in t0s]))])
    x = rd.cube.astype(np.complex128)
    x = x - x.mean(0, keepdims=True)                                # the still scene removed
    Pm = (np.abs(x) ** 2).sum(1).mean(1)                            # (n, B) moving power
    w = (t >= t0) & (t < t0 + DUR)
    keep = (rng >= 0.2) & (rng <= RMAX)
    A = P.db_over_median(Pm[w], lo=3, hi=25)[:, keep]
    P.save_png("fe_raster", P.to_rgb(np.repeat(np.flipud(A.T), 2, axis=1)))
    # radar on its own: strongest moving bin per frame, its azimuth from the RX2-RX1 phase (calibrated)
    cal = R.angle_calib()["az"]
    a_, b_ = R.PAIRS["az"]
    ok_b = np.flatnonzero((rng >= RMIN) & (rng <= RMAX))
    pts = []
    for f in np.flatnonzero(w):
        b = int(ok_b[np.argmax(Pm[f, ok_b])])
        snr = 10 * np.log10(Pm[f, b] / max(np.median(Pm[f, ok_b]), 1e-300))
        if snr < SNR_DB:
            continue
        seg = x[f][:, :, b - 1:b + 2]
        az = float(R.phase_to_angle(np.angle((seg[:, b_] * np.conj(seg[:, a_])).sum()), cal["sign"], cal["offset_rad"]))
        pts.append((t[f] - t0, float(rng[b]), az, float(snr)))
    pts = np.array(pts)
    # constant lines of the moving-power map: still things the mean subtraction does not remove completely
    # (how often a range bin stands out in the raster of panel (a), averaged over the window; peaks >= 0.15 m apart)
    Am = P.db_over_median(Pm[w], lo=3, hi=25).mean(0)
    pk = [b for b in ok_b[1:-1] if Am[b] >= Am[b - 1] and Am[b] >= Am[b + 1]]
    lines = []
    for b in sorted(pk, key=lambda b: -Am[b]):
        if all(abs(rng[b] - x) >= 0.15 for x in lines):
            lines.append(float(rng[b]))
        if len(lines) == 3:
            break
    lines = sorted(lines)
    Ls = 10 * np.log10((np.abs(rd.cube.astype(np.complex128).mean(0)) ** 2).sum((0, 1)) + 1e-30)
    Ls = Ls - np.median(Ls[ok_b])
    line_static_db = [float(Ls[int(np.argmin(np.abs(rng - lr)))]) for lr in lines]
    sc = scene(t0, lines, R.range_bias())
    qw = q[(q.t_host >= t0) & (q.t_host < t0 + DUR)]
    r_t, az_t, _ = G.polar(qw[["tx", "ty", "tz"]].to_numpy())
    tags = np.c_[qw.t_host.to_numpy() - t0, r_t, az_t]
    # agreement at the tag times (nearest radar point within 60 ms)
    j = np.clip(np.searchsorted(pts[:, 0], tags[:, 0]), 1, len(pts) - 1)
    j = np.where(np.abs(pts[j - 1, 0] - tags[:, 0]) < np.abs(pts[j, 0] - tags[:, 0]), j - 1, j)
    m = np.abs(pts[j, 0] - tags[:, 0]) < 0.06
    dr = pts[j[m], 1] - tags[m, 1]
    da = pts[j[m], 2] - tags[m, 2]
    near = np.abs(dr) < 0.3
    P.save_json("fe_dados", dict(
        rec=rec.key, t0_s_after_first_frame=t0 - t[0], xlim=[0, DUR], ylim=[float(rng[keep][0]), float(rng[keep][-1])],
        radar=pts[:, :3], tags=tags, snr_db=SNR_DB, lines=lines, line_static_db=line_static_db, scene=sc,
        n_radar=int(len(pts)), n_frames=int(w.sum()), n_tags=int(len(tags)), n_pairs=int(m.sum()),
        share_radar_on_person=float(near.mean()), range_err_med_cm=float(np.median(np.abs(dr[near])) * 100),
        az_err_med_deg=float(np.median(np.abs(da[near])))))
    print(f"t0 {t0 - t[0]:.1f}s radar pts {len(pts)}/{w.sum()} tags {len(tags)} pairs {m.sum()} "
          f"on person {near.mean():.2f} |dr| {np.median(np.abs(dr[near])) * 100:.1f} cm |daz| {np.median(np.abs(da[near])):.1f} deg")


if __name__ == "__main__":
    run()
