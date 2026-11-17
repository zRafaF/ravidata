"""One radar frame of the carried indoor walk against the camera map, in the video's real colours.
Exports (text-free): the camera keyframe, a bird's-eye raster and a side-view raster of the local map (keyframes
within +-WIN s of the radar frame, metric, in the radar's frame at that instant); JSON: the radar's range profile of
that frame, its peaks (range, calibrated azimuth), the camera's height and pitch, the truth (nearest map obstacle in
the beam). Points at depth edges (flying pixels) and the least confident 30 % are dropped; nothing is coloured by
class. usage: python -m analysis.paper.fig_quadro [take] [recording index] [seconds into the recording]"""
from __future__ import annotations

import sys

import numpy as np

from .. import common as C, geometry as G, radar as R, truth as TR
from ..maps import TakeMaps, interp_poses
from . import common as P

WIN = 4.0
EDGE = 0.04          # |grad depth| / depth per pixel above this = depth edge, dropped
RES = 0.01           # m per raster pixel
BEV = dict(x=(-2.5, 2.5), z=(0.0, 4.5))
SIDE = dict(x=(0.0, 4.5), y=(-0.2, 2.4))
H_MAX = 2.0          # bird's-eye: points up to this height (no ceiling)
SLICE = 0.35         # side view: points within +-SLICE m of the vertical plane through the radar axis


def _splat(shape, ij, rgb, order):
    img = np.full(shape + (3,), 255, np.uint8)
    ij, rgb = ij[order], rgb[order]
    for di in (0, 1):
        for dj in (0, 1):
            a = np.clip(ij[:, 0] + di, 0, shape[0] - 1); b = np.clip(ij[:, 1] + dj, 0, shape[1] - 1)
            img[a, b] = rgb
    return img


def run(take="indoor_course", rec_i=0, t_rel=30.0, name="fq"):
    import cv2
    tm = TakeMaps(take, "pi3x"); tm.fit_scale(C.load_tags())
    rec = C.recordings(take)[rec_i]; rd = C.load_radar(rec)
    f = int(np.argmin(np.abs(rd.t - (rd.t[0] + t_rel)))); tt = float(rd.t[f])
    i = int(tm.kf[np.argmin(np.abs(tm.t[tm.kf] - tt))]); k = tm.owner[i][0]
    w = tm.window(k); s_w = tm.metric(k); t_w = tm.t[w["idx"]]
    Tw = interp_poses(t_w, w["c2w"], np.array([tt]))[0]
    frames_dir = C.DATA / "derived" / "maps-v1" / "frames" / take / "rgb"
    Q, RGB = [], []
    for j, idx in enumerate(w["idx"]):
        if abs(tm.t[int(idx)] - tt) > WIN:
            continue
        d = w["depth"][j].astype(np.float64); c = w["conf"][j].astype(np.float32)
        im = cv2.cvtColor(cv2.imread(str(frames_dir / f"{int(idx):06d}.jpg")), cv2.COLOR_BGR2RGB)
        gy, gx = np.gradient(d)
        ok = np.isfinite(d) & (d > 0) & (np.maximum(np.abs(gx), np.abs(gy)) / np.maximum(d, 1e-6) < EDGE)
        ok &= c >= np.percentile(c[ok], 30)
        b = int(0.03 * d.shape[0]); ok[:b] = ok[-b:] = False; ok[:, :b] = ok[:, -b:] = False
        v, u = np.nonzero(ok); z = d[v, u]
        X = np.stack([(u - tm.K[0, 2]) / tm.K[0, 0] * z, (v - tm.K[1, 2]) / tm.K[1, 1] * z, z], 1)
        T = w["c2w"][j]
        Xw = X @ T[:3, :3].T + T[:3, 3]
        Q.append((Xw - Tw[:3, 3]) @ Tw[:3, :3] * s_w); RGB.append(im[v, u])
        if int(idx) == i:
            P.save_png(f"{name}_camera", im[b:-b, b:-b])
    Q = np.concatenate(Q); RGB = np.concatenate(RGB)
    up = TR._gravity_up(take, np.array([tt]))[0]
    h = Q @ up
    hf = G.floor_height(h - np.median(h), lo=-3, hi=1) + np.median(h)
    pl = TR.floor_plane(Q, up, h, hf)
    n, d0 = pl if pl is not None else (up, -hf)
    hh = Q @ n + d0
    fw = np.array([0, 0, 1.0]) - n[2] * n; fw /= np.linalg.norm(fw)
    lat = np.cross(fw, n); lat *= np.sign(lat[0]) or 1.0
    xb, zb = Q @ lat, Q @ fw
    cam_h = float(d0); pitch = float(np.degrees(np.arcsin(np.clip(-n[2], -1, 1))))
    # bird's-eye: up to H_MAX, higher points drawn last
    m = (hh < H_MAX) & (xb > BEV["x"][0]) & (xb < BEV["x"][1]) & (zb > BEV["z"][0]) & (zb < BEV["z"][1])
    shp = (int((BEV["z"][1] - BEV["z"][0]) / RES), int((BEV["x"][1] - BEV["x"][0]) / RES))
    ij = np.c_[((BEV["z"][1] - zb[m]) / RES).astype(int), ((xb[m] - BEV["x"][0]) / RES).astype(int)]
    P.save_png(f"{name}_bev", _splat(shp, ij, RGB[m], np.argsort(hh[m])))
    # side view: a slice along the radar axis, points nearest the centre line drawn last
    m = (np.abs(xb) <= SLICE) & (zb > SIDE["x"][0]) & (zb < SIDE["x"][1]) & (hh > SIDE["y"][0]) & (hh < SIDE["y"][1])
    shp = (int((SIDE["y"][1] - SIDE["y"][0]) / RES), int((SIDE["x"][1] - SIDE["x"][0]) / RES))
    ij = np.c_[((SIDE["y"][1] - hh[m]) / RES).astype(int), ((zb[m] - SIDE["x"][0]) / RES).astype(int)]
    P.save_png(f"{name}_side", _splat(shp, ij, RGB[m], np.argsort(-np.abs(xb[m]))))
    # oblique 3D view (same points, real colours, no ceiling), with the radar's track over the window as an overlay
    from .iso import Iso
    keep3 = (hh < H_MAX) & (np.hypot(xb, zb) < 4.8) & (zb > -1.5)
    X3 = np.c_[xb, zb, hh][keep3]
    iso = Iso(X3)
    P.save_png(f"{name}_iso", iso.render(X3, RGB[keep3]))
    tw = (np.abs(t_w - tt) <= WIN)
    pos = (w["c2w"][tw, :3, 3] - Tw[:3, 3]) @ Tw[:3, :3] * s_w                 # radar track, radar axes at tt
    track = np.c_[pos @ lat, pos @ fw, pos @ n + d0]
    fan = [[0, 0, cam_h]] + [[1.0 * np.sin(np.radians(a)), 1.0 * np.cos(np.radians(a)), cam_h] for a in range(-60, 61, 10)]
    iso_overlay = dict(radar=iso.project(np.array([[0, 0, cam_h]]))[0][0], foot=iso.project(np.array([[0, 0, 0]]))[0][0],
                       track=iso.project(track)[0], fan=iso.project(np.array(fan))[0], aspect=iso.W / iso.H)
    # nearest obstacle in the beam, as the truth defines it (height > 0.12 m, |az| <= 60, |el| <= 30 deg)
    r_all, az_all, el_all = G.polar(Q)
    ob = (hh > TR.H_OBS_MIN) & (np.abs(az_all) <= TR.ARC_HALF_DEG) & (np.abs(el_all) <= TR.ARC_EL_HALF_DEG) & (r_all >= G.R_MIN)
    nearest = None
    if ob.any():
        q = int(np.flatnonzero(ob)[np.argmin(r_all[ob])])
        nearest = dict(r=float(r_all[q]), az=float(az_all[q]), el=float(el_all[q]), lateral=float(xb[q]),
                       ahead=float(zb[q]), height=float(hh[q]))
    if nearest is not None:                                       # where it is in the oblique view
        iso_overlay["nearest"] = iso.project(np.array([[nearest["lateral"], nearest["ahead"], nearest["height"]]]))[0][0]
    # radar: range profile of this frame, peaks with their azimuth
    x = rd.cube[f].astype(np.complex128)                                   # (D, 3, B)
    rng = rd.range_m - R.range_bias()
    Pw = (np.abs(x) ** 2).mean((0, 1))
    use = (rng >= G.R_MIN) & (rng <= G.R_MAX)
    L = 10 * np.log10(Pw + 1e-30); L = L - np.median(L[use])
    pk = [b for b in np.flatnonzero(use)[1:-1] if L[b] >= 10 and L[b] >= L[b - 1] and L[b] >= L[b + 1]]
    pk = sorted(pk, key=lambda b: -L[b])[:8]
    cal = R.angle_calib()["az"]; a_, b_ = R.PAIRS["az"]
    peaks = []
    for b in sorted(pk):
        s = (x[:, b_, b - 1:b + 2] * np.conj(x[:, a_, b - 1:b + 2])).sum()
        peaks.append((float(rng[b]), float(L[b]), float(R.phase_to_angle(np.angle(s), cal["sign"], cal["offset_rad"]))))
    Ft = C.load_table("truth_frames")
    row = Ft[Ft.rec == rec.key].iloc[(Ft[Ft.rec == rec.key].t - tt).abs().argmin()]
    P.save_json(f"{name}_dados", dict(
        rec=rec.key, t_rel=tt - rd.t[0], keyframe=i, window=k, window_len=int(len(w["idx"])), n_points=int(len(Q)), bev=BEV, side=SIDE,
        h_max=H_MAX, slice_m=SLICE, iso=iso_overlay, cam_h=cam_h, pitch_deg=pitch, profile=np.c_[rng[use], L[use]], peaks=peaks,
        truth_arc_m=float(row.arc_r), nearest_obstacle=nearest, floor_in_beam_m=float(row.arc_floor_r), hfov=G.HFOV_HALF_DEG))
    print(f"{rec.key} t {tt - rd.t[0]:.1f}s kf {i} win {k} pts {len(Q)} h {cam_h:.2f} pitch {pitch:.0f} "
          f"nearest {nearest} truth {row.arc_r:.2f} floor {row.arc_floor_r:.2f} peaks {[(round(p[0], 2), round(p[1]), round(p[2])) for p in peaks]}")


if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0] if a else "indoor_course", int(a[1]) if len(a) > 1 else 0, float(a[2]) if len(a) > 2 else 30.0)
