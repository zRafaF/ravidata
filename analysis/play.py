"""Replay a recording as a video: camera frame + instantaneous radar heatmap + detections + truth, frame by frame.

    python -m analysis.play --rec indoor_course/01_D1_tcc-1d-40mm --start 20 --seconds 15
    make play REC=indoor_course/01_D1_tcc-1d-40mm START=20 SECONDS=15 PLAY_FAMILIES="ca_cfar grid"

Panels: (left) the synchronised camera frame with the truth points (white rings) and the detections (coloured)
projected into it; (right, top) bird's-eye view of the radar's range-azimuth plane: the *instantaneous* power of
this frame (2-element Bartlett beam over the azimuth pair, dB over the frame's noise; no accumulation), the truth
(nearest obstacle per 4 deg sector, white), the camera FOV and the walking corridor, and the detections of each
family; (right, bottom) the range profile with the CFAR threshold of the first CFAR family, the detections and the
truth ranges. Output: results/analysis-v1/videos/<recording>_<start>s_<families>.mp4 (and .gif with --gif)."""

from __future__ import annotations

import argparse
import json

import numpy as np

from . import common as C
from . import geometry as G
from . import pipeline as PL
from . import radar as R

FAM_COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
MARK = ["o", "s", "^", "D", "v", "P"]


def find_rec(key: str) -> C.Recording:
    take = key.split("/")[0]
    for r in C.recordings(take):
        if r.key == key or r.file.stem == key.split("/")[-1] or key.split("/")[-1] in r.file.stem:
            return r
    raise SystemExit(f"recording {key} not found; e.g. " + ", ".join(r.key for r in C.recordings(take)))


def truth_for(rec: C.Recording):
    """(obs_r, obs_az, obs_el, floor_r) per frame from the full tables, else the quick ones (first frames only)."""
    import pandas as pd
    for quick in (False, True):
        C.set_quick(quick)
        S = C.load_table("truth_sectors", required=False)
        if S is not None and (S.rec == rec.key).any():
            s = S[S.rec == rec.key].sort_values(["f", "sector"])
            n = s.f.max() + 1
            piv = lambda c: s[c].to_numpy().reshape(n, G.N_SECT)
            Fr = C.load_table("truth_frames").query("rec == @rec.key").sort_values("f")
            arc = Fr.arc_r.to_numpy() if "arc_r" in Fr else np.full(n, np.nan)
            C.set_quick(False)
            return piv("obs_r"), piv("obs_az"), piv("obs_el"), piv("floor_r"), ("quick" if quick else "full"), arc
    C.set_quick(False)
    return None


def bartlett(x_a, x_b, thetas_deg, cal):
    """Power of a 2-element beam steered to each angle, per range bin: (B, n_theta), with the fitted sign/offset."""
    psi = cal["sign"] * (2 * np.pi * R.D_OVER_LAMBDA * np.sin(np.radians(thetas_deg)) + cal["offset_rad"])
    cross = x_b * np.conj(x_a)                                  # (D, B)
    p = (np.abs(x_a) ** 2 + np.abs(x_b) ** 2).sum(0)[:, None] + \
        2 * np.real(cross.sum(0)[:, None] * np.exp(-1j * psi)[None])
    return np.maximum(p / 2, 1e-300)


def project(K, D, trc, r, az, el):
    """Image pixels of radar-frame points (r, az, el) (camera = radar + t_rc)."""
    import cv2
    a, e = np.radians(az), np.radians(np.nan_to_num(el))
    P = np.stack([r * np.cos(e) * np.sin(a), -r * np.sin(e), r * np.cos(e) * np.cos(a)], 1) + trc
    ok = P[:, 2] > 0.1
    uv = np.full((len(r), 2), np.nan)
    if ok.any():
        pr, _ = cv2.projectPoints(P[ok], np.zeros(3), np.zeros(3), K, D)
        uv[ok] = pr.reshape(-1, 2)
    return uv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis.play", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec", required=True, help="take/recording, e.g. indoor_course/01_D1_tcc-1d-40mm")
    ap.add_argument("--start", type=float, default=0.0, help="seconds after the recording's first scene frame")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--families", nargs="*", default=["ca_cfar", "grid"])
    ap.add_argument("--params", default="default", help="default | tuned_pooled_plane | tuned_pooled_arc | ...")
    ap.add_argument("--bg", default="none", help="heatmap background removal: none | ema | empty")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--gif", action="store_true", help="also write a GIF (half size, 10 fps)")
    ap.add_argument("--db", type=float, nargs=2, default=[0.0, 35.0], help="heatmap colour range, dB over noise")
    a = ap.parse_args(argv)

    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = find_rec(a.rec)
    C.set_quick(False)
    best = {}
    if a.params != "default":
        for q in (False, True):
            C.set_quick(q)
            best = C.load_json(f"opt_best_params_{a.params.replace('tuned_', '')}", {}) or best   # e.g. tuned_pooled_arc
            if best:
                break
    C.set_quick(False)
    cal = None
    for q in (False, True):                                     # angle calibration: full run, else quick
        C.set_quick(q)
        cal = C.load_json("angle_calib", None)
        ext = C.load_json("extrinsic", None)
        if cal:
            break
    C.set_quick(False)
    R.angle_calib.cache_clear()
    if cal:
        R.angle_calib = lambda: {"az": cal["az"], "el": cal["el"]}      # same calibration for detections
    cal_az = (cal or {}).get("az", {"sign": 1.0, "offset_rad": 0.0})
    trc = np.array((ext or {}).get("t_rc", [0, 0, 0]), float)

    # ---- radar, detections, truth (whole recording: grids and M-of-N need their history)
    rd = C.load_radar(rec)
    x = rd.cube.astype(np.complex128)
    if a.bg == "empty":
        ref = R._empty_ref(rec)
        x = x - ref[None] if ref is not None else x
    elif a.bg == "ema":
        from scipy.signal import lfilter
        bg = lfilter([0.02], [1, -0.98], x, axis=0, zi=(0.98 * x[0])[None])[0]
        x = x - np.concatenate([x[:1], bg[:-1]], 0)
    i0 = int(np.searchsorted(rd.t, rd.t[0] + a.start))
    i1 = int(np.searchsorted(rd.t, rd.t[0] + a.start + a.seconds))
    dets, Fs = {}, {}
    for fam in a.families:
        d, ms, F = PL.detect(rec, fam, best.get(fam, {}), None)
        if d is None:
            C.log(f"{fam}: not applicable to {rec.config}")
            continue
        dets[fam] = d
        Fs[fam] = F
    tr = truth_for(rec)
    cfar_fam = next((f for f in dets if "cfar" in f), None)
    thr_map = None
    if cfar_fam:
        p = dict(PL.DEFAULTS[cfar_fam], **best.get(cfar_fam, {}))
        F = Fs[cfar_fam]
        if cfar_fam == "rd_cfar":
            thr_map = None
        else:
            from . import detectors as D
            kind = "os" if cfar_fam == "os_cfar" else p.get("kind", "ca")
            thr_map = D.cfar_1d(F.P, kind, p["guard"], p["train"], p["k_db"], p.get("os_q", 0.75))
    Fref = Fs[next(iter(Fs))] if Fs else R.features(rec)

    # ---- static parts of the figure
    K, Dd, W = C.camera_K()
    thetas = np.linspace(-75, 75, 151)
    rr = rd.range_m - R.range_bias()
    TH, RR = np.meshgrid(np.radians(thetas), rr)
    Xg, Zg = RR * np.sin(TH), RR * np.cos(TH)
    plt.rcParams.update({"font.size": 9, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    fig = plt.figure(figsize=(14, 7.2), dpi=100)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1, 0.75], wspace=0.12, hspace=0.35)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_bev = fig.add_subplot(gs[:2, 1])
    ax_rp = fig.add_subplot(gs[2, 1])
    ax_img.axis("off")
    im_img = ax_img.imshow(np.zeros((720, 720, 3), np.uint8))
    ax_img.set_xlim(0, 720); ax_img.set_ylim(720, 0)
    mesh = ax_bev.pcolormesh(Xg, Zg, np.zeros_like(Xg)[:-1, :-1], cmap="magma", vmin=a.db[0], vmax=a.db[1],
                             shading="flat")
    fig.colorbar(mesh, ax=ax_bev, fraction=0.04, pad=0.01, label="power over noise (dB)")
    for rr_ in (1, 2, 3, 4):
        t = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ax_bev.plot(rr_ * np.sin(t), rr_ * np.cos(t), color="#ffffff", lw=0.4, alpha=0.35)
    for s in (-G.HFOV_HALF_DEG, G.HFOV_HALF_DEG):
        ax_bev.plot([0, 4.5 * np.sin(np.radians(s))], [0, 4.5 * np.cos(np.radians(s))], color="#ffffff", lw=0.8,
                    ls="--", alpha=0.6)
    ax_bev.plot([-0.5, -0.5, 0.5, 0.5], [0.3, 3.0, 3.0, 0.3], color="#1baf7a", lw=0.8, alpha=0.8)
    ax_bev.set_xlim(-3, 3); ax_bev.set_ylim(0, 4.5); ax_bev.set_aspect("equal")
    ax_bev.set_xlabel("X (m), right +"); ax_bev.set_ylabel("Z (m), ahead")
    tr_bev, = ax_bev.plot([], [], "o", mfc="none", mec="#ffffff", ms=9, mew=1.5, label="truth (nearest per sector)")
    tr_img, = ax_img.plot([], [], "o", mfc="none", mec="#ffffff", ms=10, mew=1.5)
    arc_t = np.radians(np.linspace(-60, 60, 61))
    arc_tr, = ax_bev.plot([], [], color="#ffffff", lw=1.5, ls="--", label="arc truth (nearest, +-60 deg)")
    arc_det = {}
    det_bev, det_img, det_rp = {}, {}, {}
    for k, fam in enumerate(dets):
        det_bev[fam], = ax_bev.plot([], [], MARK[k], color=FAM_COL[k], ms=6, mec="#0b0b0b", mew=0.5, label=fam)
        det_img[fam], = ax_img.plot([], [], MARK[k], color=FAM_COL[k], ms=8, mec="#0b0b0b", mew=0.5)
        det_rp[fam] = ax_rp.vlines([], 0, 1, color=FAM_COL[k], lw=1.5, transform=ax_rp.get_xaxis_transform())
        arc_det[fam], = ax_bev.plot([], [], color=FAM_COL[k], lw=1.0, alpha=0.8)
    ax_bev.legend(loc="upper right", fontsize=7, framealpha=0.7)
    rp_line, = ax_rp.plot(rr, np.zeros_like(rr), color="#0b0b0b", lw=1.2, label="power")
    rp_thr, = ax_rp.plot(rr, np.full_like(rr, np.nan), color="#eb6834", lw=1, ls="--",
                         label=f"{cfar_fam} threshold" if cfar_fam else "")
    rp_tr = ax_rp.vlines([], 0, 1, color="#52514e", lw=1, ls=":", transform=ax_rp.get_xaxis_transform())
    ax_rp.set_xlim(0, G.R_MAX + 0.3); ax_rp.set_ylim(-5, 50)
    ax_rp.set_xlabel("range (m)"); ax_rp.set_ylabel("dB over noise"); ax_rp.legend(fontsize=7, loc="upper right")
    title = fig.suptitle("")

    # ---- video
    sync = C.sync_of(rec.take)
    cont = av.open(str(C.DATA / "takes" / rec.take / "video.mp4"))
    vs = cont.streams.video[0]
    vs.thread_type = "AUTO"
    tv0 = float(sync.host_to_video(rd.t[i0]))
    cont.seek(int(max(tv0 - 1.0, 0) / vs.time_base), stream=vs, backward=True)
    frames = cont.decode(vs)
    cur = None

    def video_at(tv):
        nonlocal cur
        for fr in frames:
            img = fr.to_ndarray(format="rgb24")
            cur = (float(fr.time or 0), img)
            if cur[0] >= tv - 1 / 60:
                break
        return cur[1] if cur else None

    out_dir = C.out_dir("videos")
    name = f"{rec.take}_{rec.file.stem}_{a.start:.0f}s_{'-'.join(dets)}"
    fps = (len(rd.t) - 1) / (rd.t[-1] - rd.t[0]) * a.speed
    oc = av.open(str(out_dir / f"{name}.mp4"), "w")
    ost = oc.add_stream("libx264", rate=int(round(fps)))
    ost.pix_fmt = "yuv420p"
    ost.options = {"crf": "20"}
    gif = []
    starts = {f: np.searchsorted(d["f"], np.arange(len(rd.t) + 1)) for f, d in dets.items()}
    a_pair, b_pair = R.PAIRS["az"]
    pb = C.bar(range(i0, i1), desc=f"play {rec.key}", unit="frame")
    for i in pb:
        # heatmap (instantaneous, this frame only)
        P = bartlett(x[i, :, a_pair], x[i, :, b_pair], thetas, cal_az)
        noise = np.median(Fref.noise[i]) if Fref is not None else np.median(P)
        mesh.set_array((10 * np.log10(P[:-1, :-1] / noise)).ravel())
        rp_line.set_ydata(10 * np.log10(np.maximum(Fref.P[i], 1e-300) / Fref.noise[i]))
        if thr_map is not None:
            rp_thr.set_ydata(10 * np.log10(np.maximum(thr_map[i], 1e-300) / Fs[cfar_fam].noise[i]))
        # truth
        if tr is not None and i < len(tr[0]):
            o_r, o_az, o_el = tr[0][i], tr[1][i], tr[2][i]
            m = np.isfinite(o_r)
            X, Z = G.plane(o_r[m], o_az[m])
            tr_bev.set_data(X, Z)
            uv = project(K, Dd, trc, o_r[m], o_az[m], o_el[m]) * 720 / W
            tr_img.set_data(uv[:, 0], uv[:, 1])
            rp_tr.set_segments([[[r_, 0], [r_, 1]] for r_ in np.unique(np.round(o_r[m], 2))])
            ar = tr[5][i] if i < len(tr[5]) else np.nan
            arc_tr.set_data(*((ar * np.sin(arc_t), ar * np.cos(arc_t)) if np.isfinite(ar) else ([], [])))
        else:
            tr_bev.set_data([], []); tr_img.set_data([], [])
        # detections
        txt = []
        for fam, d in dets.items():
            s0, s1 = starts[fam][i], starts[fam][i + 1]
            r_, az_, el_ = d["r"][s0:s1], np.nan_to_num(d["az"][s0:s1]), d["el"][s0:s1]
            X, Z = G.plane(r_, az_)
            det_bev[fam].set_data(X, Z)
            uv = project(K, Dd, trc, r_, az_, el_) * 720 / W
            det_img[fam].set_data(uv[:, 0], uv[:, 1])
            det_rp[fam].set_segments([[[v, 0], [v, 0.12]] for v in r_])
            nr = r_.min() if len(r_) else np.nan
            arc_det[fam].set_data(*((nr * np.sin(arc_t), nr * np.cos(arc_t)) if np.isfinite(nr) else ([], [])))
            txt.append(f"{fam}: {s1 - s0}")
        img = video_at(float(sync.host_to_video(rd.t[i])))
        if img is not None:
            im_img.set_data(cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA))
        title.set_text(f"{rec.key}   t = {rd.t[i] - rd.t[0]:6.2f} s   frame {i}   detections  " + ", ".join(txt)
                       + ("" if tr is not None else "   (no truth table: run the truth step)"))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        h, w = buf.shape[:2]
        buf = buf[: h - h % 2, : w - w % 2]
        if i == i0:
            ost.width, ost.height = buf.shape[1], buf.shape[0]
        for pkt in ost.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(buf), format="rgb24")):
            oc.mux(pkt)
        if a.gif and (i - i0) % max(1, int(round(fps / 10))) == 0:
            from PIL import Image
            gif.append(Image.fromarray(buf).resize((buf.shape[1] // 2, buf.shape[0] // 2)))
    for pkt in ost.encode():
        oc.mux(pkt)
    oc.close()
    cont.close()
    C.log(f"-> {C.rel(out_dir / f'{name}.mp4')}")
    if gif:
        gif[0].save(out_dir / f"{name}.gif", save_all=True, append_images=gif[1:], duration=100, loop=0)
        C.log(f"-> {C.rel(out_dir / f'{name}.gif')}")


if __name__ == "__main__":
    main()
