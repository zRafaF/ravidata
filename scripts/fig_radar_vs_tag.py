"""Radar range vs AprilTag range from the phone video, on one clock (thesis figure + numbers).

    uv run python scripts/fig_radar_vs_tag.py runs/packs/<pack>.tccpack.h5 --take <moving take id>
        [--static-takes ID,ID,...] [--person-tags 0,1,2] [--calib runs/recs/calib/<phone>.calib.json]
        [--out runs/recs/<folder>/radar_vs_tag.png]

Everything comes from files that are kept in the repo, so it re-runs without new measurements:

* the pack (`ravidata pack`): radar cubes, the embedded phone video, the clock alignment (beeps + manual offsets);
* the camera intrinsics (`ravidata calib`), scaled to the video's size (a WhatsApp copy is the same 1:1 picture,
  downscaled);
* `docs/protocols/targets/tags.json` (tag ID -> printed size).

What it does:

1. **Static tags** (e.g. on chairs): averages the video frames of the static takes (the picture does not move, so this
   removes most compression noise), finds the tags and their range / azimuth from the camera (``solvePnP``).
2. **Moving tag** (e.g. on a person's chest): finds it in every video frame of the moving take -> range over time on
   the radar clock.
3. **Radar**: the moving return per radar frame. Static clutter is removed by subtracting the complex mean over the
   take (valid only while the radar itself is still); the strongest remaining range bin in [r_min, r_max] is the
   moving target. Frames with little moving power are ignored.
4. **Time check**: shifts the video by -1.5..+1.5 s and finds the shift where radar and tag ranges agree best.
   ``extra_lag_s`` near 0 means the pack's alignment is right.
5. Figure: left, the static range profile of each static take with the static tags marked; right, the moving
   return (range-time, clutter removed) with the tag ranges on top, with the pack alignment and with beeps only.

Caveats: the camera is a few cm from the radar (session form: cam_dx); the tag range is to the tag centre, the radar
range is to the strongest scatterer (the torso for a chest tag). Both are within one 4 cm bin of each other here.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, YELLOW, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#2b2b2b", "#8a8a85"


# ------------------------------------------------------------------------------------------------ camera + tags
def camera_for(video: dict, calib_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """K, dist for the frames as decoded. Upright video (rotation 0, e.g. WhatsApp) -> the calib's 'display'
    block; a video stored sideways like the calibration clip -> its 'raw' block. Scaled to the video size."""
    cal = json.loads(calib_path.read_text(encoding="utf-8"))
    w, h, rot = int(video["width"]), int(video["height"]), int(video.get("rotation", 0))
    block = "raw" if rot != 0 and rot == cal["rotation_deg"] else "display"
    c = cal[block]
    cw, ch = c["image_size"]
    if abs(w / h - cw / ch) > 1e-3:
        raise SystemExit(f"video {w}x{h} has another aspect ratio than the calibration ({cw}x{ch}): cropped?")
    s = w / cw
    K = np.array(c["K"], float)
    K[:2] *= s
    return K, np.array(c["dist"], float), f"{calib_path.name} [{block}] x {s:.4f}"


def tag_sizes() -> dict[int, float]:
    t = json.loads((ROOT / "docs/protocols/targets/tags.json").read_text(encoding="utf-8"))["tags"]
    return {int(k): float(v["size_mm"]) for k, v in t.items()}


def _detector():
    import cv2
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                   cv2.aruco.DetectorParameters())


def _stretch(g: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(g, (1, 99.5))
    return np.clip((g.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)


def tag_poses(gray: np.ndarray, K, D, sizes, det, upscale: float) -> list[dict]:
    """[{id, range_m, az_deg (+ = right), el_deg (+ = up), x, y, z, px}] for every known tag in a grey frame."""
    import cv2
    g = _stretch(gray)
    if upscale != 1:
        g = cv2.resize(g, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    corners, ids, _ = det.detectMarkers(g)
    out = []
    if ids is None:
        return out
    for cc, i in zip(corners, ids.ravel()):
        L = sizes.get(int(i))
        if L is None:
            continue
        img = (cc.reshape(4, 2) / upscale).astype(np.float32)
        obj = np.array([[-L / 2, L / 2, 0], [L / 2, L / 2, 0], [L / 2, -L / 2, 0], [-L / 2, -L / 2, 0]], np.float32)
        ok, rv, tv = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        x, y, z = (tv.ravel() / 1000.0).tolist()
        out.append({"id": int(i), "range_m": float(np.sqrt(x * x + y * y + z * z)),
                    "az_deg": float(np.degrees(np.arctan2(x, z))), "el_deg": float(np.degrees(np.arctan2(-y, z))),
                    "x": x, "y": y, "z": z,
                    "px": float(np.mean([np.linalg.norm(img[k] - img[(k + 1) % 4]) for k in range(4)]))})
    return out


# ------------------------------------------------------------------------------------------------ video on the radar clock
def video_frames(P: dict, t0: float, t1: float, beep_only: bool = False):
    """(t_host, grey frame) for video frames whose host time is in [t0, t1]."""
    import av
    al = P["alignment"]
    man = 0.0 if beep_only else al.get("manual_video_ms", 0.0) / 1000.0
    with av.open(io.BytesIO(P["video"]["mp4"])) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for fr in c.decode(vs):
            th = al["video_a"] * float(fr.time) + al["video_b"] + man
            if th < t0:
                continue
            if th > t1:
                break
            yield th, fr.to_ndarray(format="gray")


def static_tags(P, takes, K, D, sizes, upscale) -> dict[int, dict]:
    """Average the frames of the static takes, detect once -> {id: pose}."""
    acc, n = None, 0
    for rid in takes:
        th = P["takes"][rid]["t_host"]
        for _, g in video_frames(P, th[0] + 1.0, th[-1] - 1.0):
            acc = g.astype(np.float64) if acc is None else acc + g
            n += 1
    if not n:
        return {}
    return {p["id"]: p | {"frames_averaged": n} for p in tag_poses(acc / n, K, D, sizes, _detector(), upscale)}


def moving_tags(P, rid, K, D, sizes, ids, upscale) -> np.ndarray:
    """(N, 4): t_host (pack alignment), id, range_m, az_deg, for the moving tags in the moving take."""
    th = P["takes"][rid]["t_host"]
    det, rows = _detector(), []
    for t, g in video_frames(P, th[0] - 0.5, th[-1] + 0.5):
        for p in tag_poses(g, K, D, sizes, det, upscale):
            if p["id"] in ids:
                rows.append((t, p["id"], p["range_m"], p["az_deg"]))
    return np.array(rows, float).reshape(-1, 4)


# ------------------------------------------------------------------------------------------------ radar
def static_profile_db(take: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean |cube| over frames and RX (zero-Doppler row for 2D), dB."""
    c = take["cube"]
    m = np.abs(c[:, c.shape[1] // 2]) if c.shape[1] > 1 else np.abs(c[:, 0])
    return take["range_m"], 20 * np.log10(m.mean(axis=(0, 1)) + 1e-12)


def moving_return(take: dict, r_min: float, r_max: float, smooth: int = 5):
    """Clutter-removed power (T, R) and the moving target's range per frame (NaN when weak)."""
    from scipy.ndimage import uniform_filter1d
    c = take["cube"]                                     # (T, D, RX, R)
    dyn = (np.abs(c - c.mean(0, keepdims=True)) ** 2).sum(axis=(1, 2))
    dyn = uniform_filter1d(dyn, smooth, axis=0)
    r = take["range_m"]
    m = (r >= r_min) & (r <= r_max)
    rr = r[m][np.argmax(dyn[:, m], axis=1)]
    pw = dyn[:, m].max(axis=1)
    rr[pw < np.percentile(pw, 30)] = np.nan
    return dyn, rr


def lag_scan(t_radar, r_radar, t_tag, r_tag, lags) -> np.ndarray:
    """(L, 3): lag, median |radar - tag| (m), n. Positive lag = the video is shifted later."""
    out = []
    for lag in lags:
        rr = np.interp(t_tag + lag, t_radar, r_radar, left=np.nan, right=np.nan)
        e = np.abs(rr - r_tag)
        e = e[np.isfinite(e)]
        out.append((lag, float(np.median(e)) if e.size >= 20 else np.nan, e.size))
    return np.array(out)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pack")
    ap.add_argument("--take", help="moving take id (default: the last take in the pack)")
    ap.add_argument("--static-takes", default="", help="comma list (default: every other take)")
    ap.add_argument("--person-tags", default="0,1,2", help="IDs of the moving tags (default: the 150 mm tags)")
    ap.add_argument("--calib", default=str(ROOT / "runs/recs/calib/samsung_2026-11-13.calib.json"))
    ap.add_argument("--r-min", type=float, default=0.4)
    ap.add_argument("--r-max", type=float, default=4.6)
    ap.add_argument("--upscale", type=float, default=0, help="tag search upscale (default: 2 below 1000 px wide)")
    ap.add_argument("--out", help="PNG (default: next to the pack, <pack>.radar_vs_tag.png)")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ravidata.align import load_pack

    pack = Path(a.pack)
    P = load_pack(pack)
    if "mp4" not in P.get("video", {}):
        raise SystemExit("the pack has no embedded video (packed with --no-video)")
    takes = list(P["takes"])
    rid = a.take or takes[-1]
    static = [s for s in a.static_takes.split(",") if s] or [t for t in takes if t != rid]
    K, D, cam = camera_for(P["video"], Path(a.calib))
    up = a.upscale or (2.0 if int(P["video"]["width"]) < 1000 else 1.0)
    sizes = tag_sizes()
    ids = {int(x) for x in a.person_tags.split(",") if x}
    al = P["alignment"]
    man = al.get("manual_video_ms", 0.0) / 1000.0

    st = static_tags(P, static, K, D, sizes, up)
    mv = moving_tags(P, rid, K, D, sizes, ids, up)
    tk = P["takes"][rid]
    dyn, rr = moving_return(tk, a.r_min, a.r_max)
    th = tk["t_host"]
    lags = np.round(np.arange(-1.5, 1.5001, 0.01), 3)
    scan = lag_scan(th, rr, mv[:, 0], mv[:, 2], lags) if len(mv) else np.zeros((0, 3))
    ok = np.isfinite(scan[:, 1]) if len(scan) else np.zeros(0, bool)
    best = scan[ok][np.argmin(scan[ok, 1])] if ok.any() else [np.nan, np.nan, 0]
    at = lambda lag: scan[np.argmin(np.abs(scan[:, 0] - lag))] if len(scan) else [lag, np.nan, 0]  # noqa: E731

    res = {
        "pack": pack.name, "moving_take": rid, "static_takes": static, "camera": cam, "upscale": up,
        "video": {"name": P["video"].get("name"), "size": [int(P["video"]["width"]), int(P["video"]["height"])]},
        "alignment": {"video_a": al["video_a"], "video_b": al["video_b"], "manual_video_ms": al.get("manual_video_ms", 0.0),
                      "beeps_matched": len(al.get("video_matches", []))},
        "static_tags": st,
        "moving_tag_frames": int(len(mv)), "moving_tag_ids": sorted({int(i) for i in mv[:, 1]}) if len(mv) else [],
        "moving_tag_range_m": [float(np.nanmin(mv[:, 2])), float(np.nanmax(mv[:, 2]))] if len(mv) else None,
        "median_abs_err_m": {"pack_alignment": float(at(0.0)[1]), "beeps_only": float(at(-man)[1]),
                             "best": float(best[1])},
        "extra_lag_s": float(best[0]), "n_compared": int(best[2]),
        "radar_frames": int(len(th)),
    }

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    cols = [BLUE, AQUA, ORANGE, YELLOW]
    for k, s in enumerate(static[:4]):
        r, db = static_profile_db(P["takes"][s])
        m = (r > 0.3) & (r < 4.9)
        is2d = P["takes"][s]["cube"].shape[1] > 1
        ax[0].plot(r[m], db[m] - db[m].max(), lw=1.6, color=cols[k],
                   label=f"{s.split('_spi_')[-1]}{' (zero Doppler)' if is2d else ''}")
    for p in st.values():
        ax[0].axvline(p["range_m"], color=MUTED, lw=1, ls="--")
    if st:
        x0 = max(p["range_m"] for p in st.values()) + 0.05
        ax[0].text(x0, 1.5, "\n".join(f"tag {i}: {p['range_m']:.2f} m, {p['az_deg']:+.0f}°" for i, p in sorted(st.items())),
                   color=INK, fontsize=8.5, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
    ax[0].set_xlim(0.3, 4.9)
    ax[0].set_xlabel("range (m)")
    ax[0].set_ylabel("mean |cube| (dB re. strongest)")
    ax[0].set_title("Static scene: range profile per take, static tags from video", loc="left", color=INK, fontsize=11)
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower left")

    Dd = 10 * np.log10(dyn + 1e-12)
    Dd -= np.percentile(Dd, 99.5)
    r = tk["range_m"]
    im = ax[1].imshow(Dd.T, origin="lower", aspect="auto", extent=[0, th[-1] - th[0], r[0], r[-1]], cmap="Blues",
                      vmin=-30, vmax=0)
    if len(mv):
        ax[1].scatter(mv[:, 0] - th[0] - man, mv[:, 2], s=6, facecolor="none", edgecolor=MUTED, linewidth=0.6,
                      label="tag range, beeps only", zorder=2)
        ax[1].scatter(mv[:, 0] - th[0], mv[:, 2], s=10, color=ORANGE, edgecolor="white", linewidth=0.4,
                      label=f"tag range, pack alignment (video {1000 * man:+.0f} ms)", zorder=3)
    ax[1].set_ylim(a.r_min - 0.1, a.r_max)
    ax[1].set_xlim(0, th[-1] - th[0])
    ax[1].grid(False)
    ax[1].set_xlabel("time in take (s)")
    ax[1].set_ylabel("range (m)")
    ax[1].set_title("Moving target: radar (static clutter removed) vs video tag", loc="left", color=INK, fontsize=11)
    ax[1].legend(frameon=True, facecolor="white", edgecolor="none", fontsize=8.5, loc="upper left")
    cb = fig.colorbar(im, ax=ax[1], pad=0.01)
    cb.set_label("moving power (dB)", color=INK)
    cb.outline.set_visible(False)
    fig.tight_layout()
    out = Path(a.out) if a.out else pack.with_suffix("").with_suffix(".radar_vs_tag.png")
    fig.savefig(out, dpi=150)
    res["figure"] = out.name
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    e = res["median_abs_err_m"]
    print("static tags: " + ", ".join(f"{i}: {p['range_m']:.2f} m {p['az_deg']:+.1f} deg ({p['px']:.0f} px)"
                                       for i, p in sorted(st.items())))
    print(f"moving tag: {res['moving_tag_frames']} frames, ids {res['moving_tag_ids']}, range {res['moving_tag_range_m']}")
    print(f"median |radar - tag|: pack alignment {e['pack_alignment']:.3f} m, beeps only {e['beeps_only']:.3f} m, "
          f"best {e['best']:.3f} m at extra lag {res['extra_lag_s']:+.2f} s (n {res['n_compared']})")
    print(out)
    print(out.with_suffix(".json"))


if __name__ == "__main__":
    main()
