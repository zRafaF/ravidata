"""Explanatory figures (the 'story' of the evaluation), drawn from the data, for the reader and the paper.

    python -m analysis.story               # the curated set -> results/analysis-v1/figures/story/
    make story

* truth_<take>_<t>.png    how the ground truth is built at one radar frame: camera view with obstacle / floor
                          points, bird's-eye view of the local map with the sectors, the arc truth and the radar's
                          detections, and a side view (distance vs height) with the floor plane, the obstacle band
                          and the radar beam levelled by the IMU;
* waterfall_<rec>_<t>.png range-time map of the radar power with the truth on top (nearest obstacle in the beam,
                          nearest floor, nearest obstacle in the camera view) and each detector's nearest
                          detection, plus a strip per detector: found / early / missed, frame by frame.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import common as C
from . import geometry as G
from . import imu
from . import pipeline as PL
from . import truth as TR
from .plots import SERIES, INK, INK2, SURF, style

OBS_C, FLOOR_C, OTHER_C = "#eb6834", "#2a78d6", "#b9b8b3"


def _out(name, plt, fig, pdf=False):
    p = C.out_dir("figures", "story") / f"{name}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    if pdf:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    C.log(f"  figure {C.rel(p)}")


def _rec(key):
    take = key.split("/")[0]
    for r in C.recordings(take):
        if r.key == key or key.split("/")[-1] in r.file.stem:
            return r
    raise SystemExit(f"{key}: no such recording")


def _best(objective="arc"):
    return C.load_json(f"opt_best_params_pooled_{objective}", {}) or {}


def _video_frame(take, t_video):
    import av
    with av.open(str(C.DATA / "takes" / take / "video.mp4")) as c:
        vs = c.streams.video[0]
        c.seek(int(max(t_video - 0.5, 0) / vs.time_base), stream=vs, backward=True)
        img = None
        for fr in c.decode(vs):
            img = fr.to_ndarray(format="rgb24")
            if (fr.time or 0) >= t_video:
                break
    return img


def fig_truth_carried(key="indoor_course/01_D1", t_rel=30.0, pdf=False):
    import cv2
    from .maps import TakeMaps, interp_poses
    plt = style()
    rec = _rec(key)
    take = rec.take
    t_all = C.radar_times(rec)
    f = int(np.searchsorted(t_all, t_all[0] + t_rel))
    t = t_all[f]
    trc = TR.t_rc()
    tags = C.load_table("tags", required=False)
    tm = TakeMaps(take)
    tm.fit_scale(tags)
    j = int(np.argmin(np.abs(tm.t[tm.kf] - t)))
    i = int(tm.kf[j])
    X, k = tm.points(i, stride=6)
    X = X * tm.metric(k)
    up = TR._gravity_up(take, tm.t[i])[0]
    h, hf, obs, flo = TR.classify(X, up)
    # camera view (undistorted to the map's 518 px intrinsics)
    sync = C.sync_of(take)
    img = _video_frame(take, float(sync.host_to_video(tm.t[i])))
    Kd, Dd, W = C.camera_K()
    und = cv2.resize(cv2.undistort(img, Kd, Dd), (tm.res, tm.res), interpolation=cv2.INTER_AREA)
    uv = X[:, :2] / X[:, 2:3] * [tm.K[0, 0], tm.K[1, 1]] + [tm.K[0, 2], tm.K[1, 2]]
    o_r, o_az, o_el, f_r, _, cor_r = TR.summarise(G.to_radar(X, trc), obs, flo)
    # local map (+-4 s of keyframes in the same window) in the radar frame at t
    P, tk, hh, s_w, t_w, c2w_w = TR.window_cloud(tm, take, tm.owner[i][0], stride=4)
    Tw = interp_poses(t_w, c2w_w, np.array([t]))[0]
    sel = np.abs(tk - t) <= TR.ARC_WINDOW_S
    Q = G.to_radar((P[sel] - Tw[:3, 3]) @ Tw[:3, :3] * s_w, trc)
    hq = hh[sel]
    r, az, el = G.polar(Q)
    Xp, Zp = G.plane(r, az)
    o_m = (hq > TR.H_OBS_MIN) & (hq < TR.H_OBS_MAX)
    f_m = np.abs(hq) < TR.H_FLOOR_TOL
    arc, arc_fov, arc_floor = TR.arc_truth(Q, hq)
    # radar detections at that frame (tuned for the arc view, else defaults)
    best = _best()
    d, _, Fr = PL.detect(rec, "ca_cfar", best.get("ca_cfar", {}), None)
    m = d["f"] == f
    dX, dZ = G.plane(d["r"][m], np.nan_to_num(d["az"][m]))

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1.25], wspace=0.25)
    a0, a1, a2 = (fig.add_subplot(gs[0, q]) for q in range(3))
    a0.imshow(und)
    oth = ~(obs | flo)
    from matplotlib.lines import Line2D
    for msk, col, s in ((oth, OTHER_C, 2), (flo, FLOOR_C, 3), (obs, OBS_C, 3)):
        a0.scatter(uv[msk, 0], uv[msk, 1], s=s, c=col, alpha=0.45, lw=0)
    handles = [Line2D([], [], ls="", marker="o", ms=5, color=c, label=l) for c, l in
               ((OBS_C, "obstacle: 0.12-2.0 m above the floor"), (FLOOR_C, "floor"), (OTHER_C, "other (above 2 m)"))]
    handles.append(Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec="white", mew=1.6,
                          label="nearest obstacle per 4° sector"))
    sm = np.isfinite(o_r)
    if sm.any():
        a = np.radians(o_az[sm]); e = np.radians(o_el[sm])
        Pp = np.stack([o_r[sm] * np.cos(e) * np.sin(a), -o_r[sm] * np.sin(e), o_r[sm] * np.cos(e) * np.cos(a)], 1) + trc
        u2 = Pp[:, :2] / Pp[:, 2:3] * [tm.K[0, 0], tm.K[1, 1]] + [tm.K[0, 2], tm.K[1, 2]]
        a0.plot(u2[:, 0], u2[:, 1], "o", mfc="none", mec="white", ms=9, mew=1.6)
    a0.set_xlim(0, tm.res); a0.set_ylim(tm.res, 0); a0.axis("off")
    a0.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7, frameon=False)
    a0.set_title("a) camera view: map depth, labelled with the IMU 'up'", fontsize=9)

    a1.scatter(Xp[~(o_m | f_m)], Zp[~(o_m | f_m)], s=1, c=OTHER_C, alpha=0.3, lw=0)
    a1.scatter(Xp[f_m], Zp[f_m], s=1.5, c=FLOOR_C, alpha=0.35, lw=0)
    a1.scatter(Xp[o_m], Zp[o_m], s=2, c=OBS_C, alpha=0.6, lw=0)
    for half, ls, lab in ((G.HFOV_HALF_DEG, "--", "camera view ±28°"), (TR.ARC_HALF_DEG, ":", "radar beam ±60°")):
        for sg in (-1, 1):
            a1.plot([0, 5 * np.sin(np.radians(sg * half))], [0, 5 * np.cos(np.radians(sg * half))], ls, color=INK2,
                    lw=0.9, label=lab if sg > 0 else None)
    if sm.any():
        sx, sz = G.plane(o_r[sm], o_az[sm])
        a1.plot(sx, sz, "o", mfc="none", mec=INK, ms=7, mew=1.3, label="truth: nearest per sector")
    if np.isfinite(arc):
        tt = np.radians(np.linspace(-60, 60, 121))
        a1.plot(arc * np.sin(tt), arc * np.cos(tt), color=INK, lw=1.6, label=f"arc truth: {arc:.2f} m")
    a1.plot(dX, dZ, "D", color=SERIES[6], ms=6, mec=INK, mew=0.5, label="radar detections (CA-CFAR, tuned)")
    a1.plot([0], [0], "^", color=INK, ms=9)
    a1.set_xlim(-3, 3); a1.set_ylim(0, 4.5); a1.set_aspect("equal")
    a1.set_xlabel("X (m)"); a1.set_ylabel("Z (m), radar range-azimuth plane")
    a1.legend(loc="upper left", fontsize=6.5, framealpha=0.85)
    a1.set_title("b) local map (±4 s of keyframes), bird's-eye", fontsize=9)

    hor = np.hypot(Q[:, 0], Q[:, 2]) * np.sign(Q[:, 2])
    bm = (np.abs(az) <= TR.ARC_HALF_DEG)
    H = -np.median(hq[f_m] - (Q[f_m] @ (-np.array([0, 1.0, 0])))) if f_m.any() else 1.2
    a2.axhspan(TR.H_OBS_MIN, TR.H_OBS_MAX, color=OBS_C, alpha=0.07, lw=0)
    a2.axhline(0, color=FLOOR_C, lw=1)
    a2.scatter(hor[bm & ~o_m & ~f_m], hq[bm & ~o_m & ~f_m], s=1, c=OTHER_C, alpha=0.3, lw=0)
    a2.scatter(hor[bm & f_m], hq[bm & f_m], s=1.5, c=FLOOR_C, alpha=0.35, lw=0)
    a2.scatter(hor[bm & o_m], hq[bm & o_m], s=2, c=OBS_C, alpha=0.6, lw=0)
    # radar position and beam (+-30 deg about the boresight, levelled with the IMU pitch)
    Hr = float(np.median(hq[f_m] - 0)) if False else None
    cam_h = float(-TR.classify(X, up)[1])
    pitch = np.degrees(np.arcsin(np.clip(up[2], -1, 1)))       # boresight elevation above the horizon (- = down)
    a2.plot([0], [cam_h], "^", color=INK, ms=9)
    for e_ in (pitch - TR.ARC_EL_HALF_DEG, pitch, pitch + TR.ARC_EL_HALF_DEG):
        a2.plot([0, 5 * np.cos(np.radians(e_))], [cam_h, cam_h + 5 * np.sin(np.radians(e_))], ":" if e_ != pitch else "-",
                color=INK2, lw=0.9)
    if np.isfinite(arc):
        a2.axvline(arc, color=INK, lw=1.2, ls="--")
        a2.text(arc, 2.3, f" arc truth {arc:.2f} m", fontsize=7)
    if np.isfinite(arc_floor):
        a2.axvline(arc_floor, color=FLOOR_C, lw=1.0, ls="--")
        a2.text(arc_floor, 2.1, f" floor enters the beam {arc_floor:.2f} m", fontsize=7, color=FLOOR_C)
    a2.set_xlim(0, 4.5); a2.set_ylim(-0.3, 2.6)
    a2.set_xlabel("horizontal distance (m)"); a2.set_ylabel("height above the floor (m)")
    a2.set_title(f"c) side view inside the beam: obstacle band 0.12-2.0 m, radar {cam_h:.2f} m high, "
                 f"pitch {pitch:+.0f}°", fontsize=9)
    fig.suptitle(f"Ground truth at one radar frame: {rec.key}, t = {t_rel:.1f} s", fontsize=10)
    _out(f"truth_{take}_{t_rel:05.1f}s", plt, fig, pdf)


def fig_waterfall(key="indoor_course/01_D1", t0=20.0, dur=30.0, families=("ca_cfar", "grid", "grid_legacy"),
                  params="arc", pdf=False):
    plt = style()
    rec = _rec(key)
    best = _best(params) if params else {}
    Fd = C.load_table("truth_frames")
    q = Fd[Fd.rec == rec.key].sort_values("f")
    S = C.load_table("truth_sectors")
    s = S[S.rec == rec.key]
    near = s.groupby("f").obs_r.min().reindex(q.f).to_numpy()
    res = {}
    F = None
    for fam in families:
        d, _, Fx = PL.detect(rec, fam, best.get(fam, {}), None)
        res[fam] = d
        F = F or Fx
    F0 = __import__("analysis.radar", fromlist=["x"]).features(rec, {}, None)
    t = F0.t - F0.t[0]
    a, b = np.searchsorted(t, t0), np.searchsorted(t, t0 + dur)
    P = 10 * np.log10(np.maximum(F0.P[a:b], 1e-300) / F0.noise[a:b, None])
    arc = q.arc_r.to_numpy()[a:b] if "arc_r" in q else np.full(b - a, np.nan)
    flo = q.arc_floor_r.to_numpy()[a:b] if "arc_floor_r" in q else np.full(b - a, np.nan)
    nr = near[a:b]
    n_f = len(families)
    fig = plt.figure(figsize=(14, 3.2 + 2.4 * n_f))
    gs = fig.add_gridspec(1 + n_f, 1, height_ratios=[3] + [0.55] * n_f, hspace=0.12)
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(P.T, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=30,
                   extent=[t[a], t[b - 1], F0.r[0], F0.r[-1]])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="power over the frame's noise (dB)")
    tt = t[a:b]
    ax.plot(tt, arc, color="white", lw=1.8, label="truth: nearest obstacle in the beam (±60°)")
    ax.plot(tt, nr, color="white", lw=0.8, ls=":", label="truth: nearest obstacle in the camera view")
    ax.plot(tt, flo, color="#7fd4ff", lw=1.2, ls="--", label="truth: nearest floor in the beam")
    for k, (fam, d) in enumerate(res.items()):
        m = (d["f"] >= a) & (d["f"] < b)
        nd = np.full(b - a, np.nan)
        np.minimum.at(np.nan_to_num(nd, nan=np.inf), 0, 0)
        tmp = np.full(b - a, np.inf)
        np.minimum.at(tmp, d["f"][m] - a, d["r"][m])
        tmp[~np.isfinite(tmp)] = np.nan
        ax.plot(tt, tmp, ".", color=SERIES[[1, 2, 3, 6, 4][k]], ms=3.2, label=f"{fam}: nearest detection")
        res[fam] = tmp
    ax.set_ylim(0, G.R_MAX + 0.4); ax.set_ylabel("range (m)")
    lg = ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.9, facecolor="white", edgecolor="#b9b8b3")
    for txt in lg.get_texts():
        txt.set_color(INK)
    ax.set_xticklabels([])
    ax.set_title(f"{rec.key}: radar range-time map with the truth and each detector's nearest detection"
                 + (f" (settings tuned for the arc view)" if best else " (default settings)"), fontsize=10)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1baf7a", "#eda100", "#e34948", "#e4e3df"])
    for k, fam in enumerate(res):
        ax2 = fig.add_subplot(gs[1 + k], sharex=None)
        nd = res[fam]
        st = np.full(b - a, 3)
        ok = np.isfinite(arc)
        st[ok & np.isfinite(nd) & (np.abs(nd - arc) <= 0.25)] = 0
        st[np.isfinite(nd) & (~ok | (nd < arc - 0.25))] = 1
        st[ok & (~np.isfinite(nd) | (nd > arc + 0.25))] = 2
        ax2.imshow(st[None], aspect="auto", cmap=cmap, vmin=0, vmax=3, extent=[t[a], t[b - 1], 0, 1],
                   interpolation="nearest")
        ax2.set_yticks([]); ax2.set_ylabel(fam, rotation=0, ha="right", va="center", fontsize=8)
        fr = [(st == v).mean() for v in range(3)]
        ax2.text(1.005, 0.5, f"found {fr[0]:.0%}  early {fr[1]:.0%}  missed {fr[2]:.0%}", transform=ax2.transAxes,
                 fontsize=7, va="center")
        if k < len(res) - 1:
            ax2.set_xticklabels([])
    ax2.set_xlabel("time in the recording (s)   —   strips: green found (±0.25 m), amber early alarm, red missed")
    _out(f"waterfall_{rec.take}_{rec.file.stem.split('_')[0]}_{t0:05.1f}s", plt, fig, pdf)


CURATED = [
    ("truth", dict(key="indoor_course/01_D1", t_rel=30.0)),
    ("truth", dict(key="outdoor_course/01_D2", t_rel=30.0)),
    ("waterfall", dict(key="indoor_course/01_D1", t0=15.0, dur=40.0)),
    ("waterfall", dict(key="outdoor_course/01_D2", t0=15.0, dur=40.0)),
    ("waterfall", dict(key="indoor_course/03_D1", t0=10.0, dur=40.0, families=("ca_cfar", "rd_cfar", "grid"))),
]


def run(only=None, pdf=False):
    C.set_quick(False)
    for kind, kw in CURATED:
        if only and kind not in only:
            continue
        try:
            (fig_truth_carried if kind == "truth" else fig_waterfall)(pdf=pdf, **kw)
        except Exception as e:
            import traceback
            traceback.print_exc()
            C.log(f"  {kind} {kw}: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    run(a.only, a.pdf)
