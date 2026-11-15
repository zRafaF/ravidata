"""F01: overview, one indoor approach followed through the six steps (indoor_course/01, 6-20 s after the first radar
frame, chosen moment 14.0 s). Text-free rasters (video frame, range-time map, range-azimuth map of one radar frame)
and one JSON with limits, tag overlays, the beam-wide truth, the tuned threshold detector's nearest return and the
per-frame outcome (found / early / lost) with its rates over the stretch, and the constant answer."""
from __future__ import annotations

import cv2
import numpy as np

from .. import common as C, geometry as G, radar as R
from . import common as PC

TAKE, KEY = "indoor_course", "indoor_course/01_D1_tcc-1d-40mm"
T0, T1, TM = 6.0, 20.0, 14.0           # stretch and chosen moment, s after the first radar frame
RLIM = (0.2, 4.5)
AZ = np.linspace(-60, 60, 241)
TOL = 0.25                             # as analysis/score.py summary (arc_pd / arc_early_rate / arc_late_rate)
FAMILY = "thr"


def outcome(det, truth):
    """0 found, 1 early, 2 lost, -1 no truth (frames without truth are not scored)."""
    o = np.full(len(truth), -1)
    ok = np.isfinite(truth)
    has = np.isfinite(det)
    o[ok] = 2
    o[ok & has & (np.abs(det - truth) <= TOL)] = 0
    o[ok & has & (det < truth - TOL)] = 1
    return o


def rates(o):
    n = max(1, int((o >= 0).sum()))
    return dict(n=int((o >= 0).sum()), found=float((o == 0).sum() / n), early=float((o == 1).sum() / n),
                lost=float((o == 2).sum() / n))


def runs(ts, o, dt):
    """Per-frame classes -> [t_start, t_end, class] runs (for native rectangles)."""
    out = []
    for a, c in zip(ts, o):
        if out and out[-1][2] == int(c) and abs(out[-1][1] - (a - dt / 2)) < dt:
            out[-1][1] = float(a + dt / 2)
        else:
            out.append([float(a - dt / 2), float(a + dt / 2), int(c)])
    return out


def run():
    rec = [r for r in C.recordings(TAKE) if r.key == KEY][0]
    rd = C.load_radar(rec)
    t = rd.t - rd.t[0]
    dt = float(np.median(np.diff(t)))
    sel = np.flatnonzero((t >= T0) & (t < T1))
    x = rd.cube[sel[0]:sel[-1] + 1].astype(np.complex128)
    ts = t[sel[0]:sel[-1] + 1]
    P = (np.abs(x) ** 2).sum(1).mean(1)
    r_ax = rd.range_m - R.range_bias()
    db = float(np.median(np.diff(r_ax)))
    use = np.flatnonzero((r_ax >= RLIM[0]) & (r_ax <= RLIM[1]))
    lev = PC.db_over_median(P)
    ylim = [float(r_ax[use[0]] - db / 2), float(r_ax[use[-1]] + db / 2)]

    # (3) range-time raster
    rgb = PC.to_rgb(lev[:, use].T[::-1])
    rgb = cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    PC.save_png("f01_rt", rgb)

    # (4) range-azimuth map of the radar frame at the chosen moment (each bin at its interferometric azimuth)
    fm = int(np.argmin(np.abs(ts - TM)))
    a_, b_ = R.PAIRS["az"]
    S = (x[fm, :, b_] * np.conj(x[fm, :, a_])).sum(0)
    S[1:-1] = S[:-2] + S[1:-1] + S[2:]
    cal = R.angle_calib()["az"]
    az_b = R.phase_to_angle(np.angle(S), cal["sign"], cal["offset_rad"])
    g = lev[fm][:, None] * np.exp(-0.5 * ((AZ[None] - az_b[:, None]) / 6.0) ** 2)        # (B, az)
    m = g[use][::-1]
    m = cv2.resize(m, (m.shape[1], m.shape[0] * 4), interpolation=cv2.INTER_LINEAR)
    PC.save_png("f01_ra", PC.to_rgb(m))

    # (2) video frame at the moment + tags seen then
    th = float(rd.t[0] + ts[fm])
    tv = float(C.sync_of(TAKE).host_to_video(th))
    cap = cv2.VideoCapture(str(C.DATA / "takes" / TAKE / "video.mp4"))
    cap.set(cv2.CAP_PROP_POS_MSEC, tv * 1000)
    ok, fr = cap.read()
    cap.release()
    assert ok
    full = fr.shape[1]
    fr = cv2.resize(fr, (360, 360), interpolation=cv2.INTER_AREA)
    PC.save_png("f01_video", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    tg = C.load_tags()
    tg = tg[(tg["take"] == TAKE) & (tg.reproj_px < 2)]
    q = tg[np.abs(tg.t_host - th) < 0.07].drop_duplicates("id")
    tags = [dict(id=int(k.id), kind=str(k.kind), u=float(k.u / full), v=float(k.v / full),
                 r=float(np.sqrt(k.tx ** 2 + k.ty ** 2 + k.tz ** 2)), az=float(np.degrees(np.arctan2(k.tx, k.tz))))
            for k in q.itertuples()]

    # (5), (6) truth, detector, outcome: the same frames for the strip and the rates
    E = C.load_table("eval_frames_tuned_pooled_arc")
    E = E[(E.family == FAMILY) & (E.rec == KEY)].drop_duplicates("f").set_index("f")
    truth = np.full(len(ts), np.nan)
    det = np.full(len(ts), np.nan)
    fidx = np.arange(sel[0], sel[-1] + 1)
    have = np.isin(fidx, E.index.to_numpy())
    truth[have] = E.loc[fidx[have], "arc_truth"].to_numpy(float)
    det[have] = E.loc[fidx[have], "arc_det"].to_numpy(float)
    Sm = C.load_table("eval_summary_tuned_pooled_arc")
    Sm = Sm[(Sm.family == FAMILY) & (Sm["take"] == TAKE) & (Sm.config == rec.config)].iloc[0]
    const = float(Sm.arc_const_m)
    o_det = outcome(det, truth)
    o_const = outcome(np.full(len(ts), const), truth)
    pts = lambda v: [[float(a), float(b)] for a, b in zip(ts, v) if np.isfinite(b)]
    PC.save_json("f01_dados", dict(
        rec=KEY, family=FAMILY, variant="tuned_pooled_arc", t_stretch=[T0, T1], t_moment=float(ts[fm]),
        xlim=[float(ts[0] - dt / 2), float(ts[-1] + dt / 2)], ylim=ylim, az_lim=[float(AZ[0]), float(AZ[-1])],
        cam_half_fov_deg=G.HFOV_HALF_DEG, tol_m=TOL, video_time_s=tv, tags=tags,
        truth=pts(truth), det=pts(det), truth_at_moment=float(truth[fm]), det_at_moment=float(det[fm]),
        strip_det=runs(ts, o_det, dt), strip_const=runs(ts, o_const, dt),
        rates_det=rates(o_det), rates_const=rates(o_const), const_m=const,
        take_rates=dict(found=float(Sm.arc_pd), early=float(Sm.arc_early_rate), lost=float(Sm.arc_late_rate),
                        found_const=float(Sm.arc_pd_const)),
        classes=["found", "early", "lost"], unit="s after the first radar frame; m; deg"))
    print("moment", ts[fm], "video", tv, "tags", [(g["id"], g["kind"], round(g["r"], 2), round(g["az"], 1)) for g in tags])
    print("truth", truth[fm], "det", det[fm], "const", const, rates(o_det), rates(o_const), "nan truth", np.isnan(truth).sum(), len(ts))


if __name__ == "__main__":
    run()
