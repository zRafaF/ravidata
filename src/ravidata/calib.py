"""Camera intrinsics from a checkerboard video: ``uv run ravidata calib VIDEO`` (TCC-002 §7, step 1).

The board is the printed ``docs/protocols/targets/checkerboard_A4.pdf`` (``targets.CHECKER``: 10 x 7 squares of
25 mm = 9 x 6 inner corners). The phone films it from 0.5-2 m while it moves through the image and tilts.

What it does:

1. Decodes every ``step``-th frame with PyAV (grey, 8 bit) and stretches the contrast (1st-99th percentile):
   Log / flat colour profiles are very low contrast and the detector likes a full range.
2. Finds the inner corners at half resolution (``findChessboardCornersSB``, robust and fast) and refines them at
   full resolution (``cornerSubPix``).
3. ``calibrateCamera`` on all views, drops views whose reprojection error is > 3 x the median (blur, bad
   detections) and calibrates again (up to 3 rounds).
4. Checks the result: a 2-fold hold-out by time (calibrate on one half of the clip, solvePnP the other half with
   those intrinsics), a bootstrap spread of fx, fy, cx, cy, and the image coverage.

Coordinates: the intrinsics are given for the **stored** frame (what PyAV decodes, ``raw``) and for the **upright**
frame (the stored frame turned by the video's display-matrix rotation, as ``ravidata align`` shows it, ``display``).
Phone videos are usually stored sideways (rotation -90); pick the block that matches the pixels you work on.

Output: ``<video>.calib.json`` and ``<video>.calib.png`` (corners used + an undistorted frame) beside the video.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .targets import CHECKER

MODELS = {   # name -> OpenCV flags. k1k2 is the default: phone main lenses are close to distortion-free
    "k1k2": "CALIB_FIX_K3|CALIB_ZERO_TANGENT_DIST",
    "k1k2p": "CALIB_FIX_K3",
    "k1k2k3p": "",
}


def board() -> tuple[tuple[int, int], float]:
    nx, ny = CHECKER["squares"]
    return (nx - 1, ny - 1), float(CHECKER["square_mm"])


def object_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square_mm
    return obj


def stretch(g: np.ndarray) -> np.ndarray:
    """Grey image -> uint8 with the 1st..99th percentile mapped to 0..255 (Log footage is very flat)."""
    lo, hi = np.percentile(g[::4, ::4], (1, 99))
    return np.clip((g.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255).astype(np.uint8)


def detect(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Inner corners (N, 2) in full-resolution pixels, or None. Search at half size, refine at full size."""
    import cv2
    g = stretch(gray)
    small = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ok, c = cv2.findChessboardCornersSB(small, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    c = cv2.cornerSubPix(g, (c * 2 + 0.5).astype(np.float32), (5, 5), (-1, -1), crit)   # half-res px centre -> full
    return c.reshape(-1, 2)


def to_display(pts: np.ndarray, w: int, h: int, rotation: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Stored-frame pixels -> upright-frame pixels. ``rotation`` is PyAV's ``frame.rotation`` (display matrix, degrees;
    the upright image is ``np.rot90(stored, k=rotation // 90)``, i.e. -90 = turn clockwise). Returns (pts, (W, H))."""
    k = (int(round(rotation / 90.0))) % 4
    x, y = pts[..., 0], pts[..., 1]
    if k == 0:
        return pts.copy(), (w, h)
    if k == 3:     # clockwise
        return np.stack([h - 1 - y, x], -1), (h, w)
    if k == 1:     # counter-clockwise
        return np.stack([y, w - 1 - x], -1), (h, w)
    return np.stack([w - 1 - x, h - 1 - y], -1), (w, h)


def upright(img: np.ndarray, rotation: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(img, k=int(round(rotation / 90.0)) % 4))


@dataclass
class VideoDetections:
    path: str
    width: int
    height: int
    rotation: int
    fps: float
    n_frames: int
    step: int
    frames: list[int] = field(default_factory=list)      # frame index of each detection
    times: list[float] = field(default_factory=list)
    corners: list[np.ndarray] = field(default_factory=list)
    n_tested: int = 0
    sample: np.ndarray | None = None                      # one stored grey frame (for the report image)
    seconds: float = 0.0


def detect_frames(frames: Iterable[tuple[int, float, np.ndarray]], pattern, det: VideoDetections) -> VideoDetections:
    """frames: (index, time s, grey stored frame). Fills ``det``."""
    for i, t, g in frames:
        det.n_tested += 1
        if det.sample is None or det.n_tested == 40:
            det.sample = g
        c = detect(g, pattern)
        if c is not None:
            det.frames.append(int(i))
            det.times.append(float(t))
            det.corners.append(c)
    return det


def detect_video(path: str | Path, step: int | None = None, pattern=None,
                 progress: Callable[[float], None] | None = None) -> VideoDetections:
    import av
    pattern = pattern or board()[0]
    t0 = time.monotonic()
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        fps = float(vs.average_rate or vs.guessed_rate or 30)
        n = int(vs.frames or 0)
        step = step or max(1, int(round(fps / 6)))            # ~6 frames per second is plenty
        det = VideoDetections(str(path), vs.codec_context.width, vs.codec_context.height, 0, fps, n, step)

        def gen():
            for i, fr in enumerate(c.decode(vs)):
                if i == 0:
                    det.rotation = int(round(getattr(fr, "rotation", 0) or 0))
                if progress and n and i % 60 == 0:
                    progress(i / n)
                if i % step:
                    continue
                yield i, float(fr.time or i / fps), fr.to_ndarray(format="gray")
        detect_frames(gen(), pattern, det)
    det.seconds = time.monotonic() - t0
    return det


def _flags(model: str) -> int:
    import cv2
    f = 0
    for name in filter(None, MODELS[model].split("|")):
        f |= getattr(cv2, name)
    return f


def _calibrate(corners: list[np.ndarray], obj: np.ndarray, size: tuple[int, int], flags: int):
    import cv2
    ret, K, d, rv, tv, _, _, pve = cv2.calibrateCameraExtended(
        [obj] * len(corners), [c.reshape(-1, 1, 2).astype(np.float32) for c in corners], size, None, None,
        flags=flags)
    return float(ret), K, d.ravel(), rv, tv, pve.ravel()


def fx_of(K) -> float:
    return float(K[0][0])


def _pnp_rms(corners: list[np.ndarray], obj: np.ndarray, K, d) -> float:
    import cv2
    err = []
    for c in corners:
        ok, rv, tv = cv2.solvePnP(obj, c.astype(np.float64), K, d)
        if ok:
            p, _ = cv2.projectPoints(obj, rv, tv, K, d)
            err.append(np.sum((p.reshape(-1, 2) - c) ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(err)))) if err else float("nan")


def solve(corners: list[np.ndarray], size: tuple[int, int], model: str = "k1k2", pattern=None,
          square_mm: float | None = None, n_boot: int = 10, seed: int = 0,
          checks: bool = True) -> dict:
    """Calibrate with outlier rejection, then the hold-out / bootstrap / coverage checks."""
    import cv2
    pattern = pattern or board()[0]
    square_mm = square_mm or board()[1]
    obj = object_points(pattern, square_mm)
    flags = _flags(model)
    if len(corners) < 8:
        raise ValueError(f"only {len(corners)} views with the board found; need >= 8 (aim for 30+)")
    keep = np.arange(len(corners))
    for _ in range(3):
        rms, K, d, rv, tv, pve = _calibrate([corners[k] for k in keep], obj, size, flags)
        good = pve <= max(3.0 * float(np.median(pve)), 0.5)
        if good.all():
            break
        keep = keep[good]
    used = [corners[k] for k in keep]

    # hold-out by time: 8 consecutive blocks; leave one out, calibrate on the rest, solvePnP the block left out.
    # Neighbouring frames are near copies, so this is the honest spread (a random-half bootstrap underestimates it).
    blk = np.arange(len(used)) * 8 // max(len(used), 1)
    hold = {}
    if checks and len(used) >= 24:
        rows = []
        for b in range(8):
            A = [c for c, g in zip(used, blk) if g != b]
            B = [c for c, g in zip(used, blk) if g == b]
            _, Ka, da, *_ = _calibrate(A, obj, size, flags)
            rows.append([Ka[0, 0], Ka[1, 1], Ka[0, 2], Ka[1, 2], _pnp_rms(B, obj, Ka, da)])
        rows = np.array(rows)
        hold = {"blocks": 8, "fx": rows[:, 0].tolist(), "cx": rows[:, 2].tolist(), "cy": rows[:, 3].tolist(),
                "test_rms_px": rows[:, 4].tolist(),
                "fx_spread_pct": float(100 * (rows[:, 0].max() - rows[:, 0].min()) / fx_of(K)),
                "c_spread_px": float(max(np.ptp(rows[:, 2]), np.ptp(rows[:, 3]))),
                "test_rms_max_px": float(rows[:, 4].max())}
    # bootstrap (half samples)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot if len(used) >= 16 else 0):
        s = rng.choice(len(used), size=len(used) // 2, replace=False)
        _, Kb, db, *_ = _calibrate([used[k] for k in s], obj, size, flags)
        boots.append([Kb[0, 0], Kb[1, 1], Kb[0, 2], Kb[1, 2], db[0], db[1]])
    boots = np.array(boots) if boots else np.zeros((0, 6))
    # geometry of the views
    dist_m = [float(np.linalg.norm(t)) / 1000 for t in tv]
    tilt = [float(np.degrees(np.arccos(min(1.0, abs(cv2.Rodrigues(r)[0][2, 2]))))) for r in rv]
    pts = np.concatenate(used)
    hist, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=6, range=[[0, size[1]], [0, size[0]]])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return {
        "model": model, "image_size": [int(size[0]), int(size[1])],
        "K": K.tolist(), "dist": d.tolist(), "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / (2 * fx)))),
        "vfov_deg": float(np.degrees(2 * np.arctan(size[1] / (2 * fy)))),
        "rms_px": rms, "views_found": len(corners), "views_used": len(used), "kept": keep.tolist(),
        "per_view_rms_px": {"median": float(np.median(pve)), "max": float(np.max(pve))},
        "holdout": hold,
        "bootstrap_std": dict(zip(["fx", "fy", "cx", "cy", "k1", "k2"], boots.std(0).tolist())) if len(boots) else {},
        "board_distance_m": {"min": min(dist_m), "median": float(np.median(dist_m)), "max": max(dist_m)},
        "board_tilt_deg": {"median": float(np.median(tilt)), "p90": float(np.percentile(tilt, 90)),
                           "max": max(tilt)},
        "coverage_6x6": (hist > 0).astype(int).tolist(),
        "coverage_frac": float((hist > 0).mean()),
    }


def verdict(r: dict) -> list[str]:
    """Plain-language checks. Lines starting with 'WARN' need attention."""
    out = []
    w, h = r["image_size"]
    ok = lambda c, s: out.append(("ok   " if c else "WARN ") + s)  # noqa: E731
    ok(r["views_used"] >= 30, f"{r['views_used']} views used (of {r['views_found']} found); want >= 30")
    ok(r["rms_px"] < 1.0, f"reprojection RMS {r['rms_px']:.2f} px; want < 1 px")
    ok(abs(r["fx"] / r["fy"] - 1) < 0.01, f"fx/fy = {r['fx'] / r['fy']:.4f}; square pixels expected (1 +- 0.01)")
    ok(abs(r["cx"] - (w - 1) / 2) < 0.05 * w and abs(r["cy"] - (h - 1) / 2) < 0.05 * h,
       f"centre ({r['cx']:.0f}, {r['cy']:.0f}) vs image middle ({(w - 1) / 2:.0f}, {(h - 1) / 2:.0f}); want within 5 %")
    hv = r["holdout"]
    if hv:
        ok(hv["fx_spread_pct"] < 2.0 and hv["test_rms_max_px"] < 1.5,
           f"leave-one-block-out (8 time blocks): fx {min(hv['fx']):.0f}..{max(hv['fx']):.0f} "
           f"(spread {hv['fx_spread_pct']:.1f} %; want < 2 %), centre spread {hv['c_spread_px']:.0f} px, "
           f"worst held-out RMS {hv['test_rms_max_px']:.2f} px")
    if r["bootstrap_std"]:
        s = r["bootstrap_std"]
        ok(s["fx"] / r["fx"] < 0.01, f"bootstrap std: fx {s['fx']:.1f} px ({100 * s['fx'] / r['fx']:.2f} %), "
           f"cx {s['cx']:.1f} px, cy {s['cy']:.1f} px")
    ok(r["coverage_frac"] >= 0.8, f"corners reach {100 * r['coverage_frac']:.0f} % of a 6 x 6 image grid; want >= 80 %")
    ok(r["board_tilt_deg"]["p90"] >= 25, f"board tilt: median {r['board_tilt_deg']['median']:.0f} deg, "
       f"90th pct {r['board_tilt_deg']['p90']:.0f} deg; want some views >= 25-45 deg")
    return out


def report_image(det: VideoDetections, res_raw: dict, path: Path) -> Path:
    """Upright: left = all corners used (on a sample frame), right = the same frame undistorted, with a grid."""
    import cv2
    rot = det.rotation
    g = stretch(det.sample) if det.sample is not None else np.full((det.height, det.width), 128, np.uint8)
    K, d = np.array(res_raw["K"]), np.array(res_raw["dist"])
    und = cv2.undistort(g, K, d)
    left = cv2.cvtColor(upright(g, rot), cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(upright(und, rot), cv2.COLOR_GRAY2BGR)
    for k in res_raw["kept"]:
        p, _ = to_display(det.corners[k], det.width, det.height, rot)
        for x, y in p:
            cv2.circle(left, (int(x), int(y)), 2, (0, 200, 255), -1)
    H, W = right.shape[:2]
    for x in range(0, W, W // 8):
        cv2.line(right, (x, 0), (x, H), (0, 255, 0), 1)
    for y in range(0, H, H // 8):
        cv2.line(right, (0, y), (W, y), (0, 255, 0), 1)
    img = np.hstack([left, right])
    cv2.putText(img, f"corners used ({res_raw['views_used']} views)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 255), 2)
    cv2.putText(img, f"undistorted  fx={res_raw['fx']:.0f}  rms={res_raw['rms_px']:.2f}px", (W + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    s = 1600 / img.shape[1]
    cv2.imwrite(str(path), cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))
    return path


def run(video: str | Path, out: str | Path | None = None, step: int | None = None, model: str = "k1k2",
        settings: str = "", progress: Callable[[float], None] | None = None) -> dict:
    video = Path(video)
    pattern, square = board()
    det = detect_video(video, step, pattern, progress)
    raw = solve(det.corners, (det.width, det.height), model, pattern, square)
    disp_corners = [to_display(c, det.width, det.height, det.rotation)[0] for c in det.corners]
    dsize = to_display(np.zeros((1, 2)), det.width, det.height, det.rotation)[1]
    disp = solve([disp_corners[k] for k in raw["kept"]], dsize, model, pattern, square, n_boot=0,
                 checks=False)
    disp.pop("kept", None)
    res = {
        "tool": "ravidata calib", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video": video.name, "frames": det.n_frames, "fps": det.fps, "step": det.step,
        "frames_tested": det.n_tested, "detect_seconds": round(det.seconds, 1),
        "stored_size": [det.width, det.height], "rotation_deg": det.rotation,
        "board": {"inner_corners": list(pattern), "square_mm": square, "pdf": "docs/protocols/targets/checkerboard_A4.pdf"},
        "camera_settings": settings,
        "raw": raw, "display": disp,
        "checks": verdict(raw),
        "use": "display = upright frame (np.rot90(stored, rotation_deg // 90), as ravidata align shows it); "
               "raw = frame as decoded. K is [[fx,0,cx],[0,fy,cy],[0,0,1]], dist is OpenCV (k1,k2,p1,p2,k3).",
    }
    out = Path(out) if out else video.with_suffix(".calib.json")
    png = out.with_suffix(".png")
    report_image(det, raw, png)
    raw.pop("kept", None)
    res["report_png"] = png.name
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    res["_json"] = str(out)
    return res


def fmt(res: dict) -> str:
    d = res["display"]
    lines = [
        f"{res['video']}: {res['stored_size'][0]}x{res['stored_size'][1]} stored, rotation {res['rotation_deg']} deg, "
        f"{res['fps']:.2f} fps; every {res['step']}th frame, {res['frames_tested']} tested, "
        f"board in {res['raw']['views_found']} ({res['detect_seconds']} s)",
        f"upright {d['image_size'][0]}x{d['image_size'][1]}: fx {d['fx']:.1f}  fy {d['fy']:.1f}  "
        f"cx {d['cx']:.1f}  cy {d['cy']:.1f}  dist {np.round(d['dist'], 4).tolist()}",
        f"field of view {d['hfov_deg']:.1f} x {d['vfov_deg']:.1f} deg; RMS {d['rms_px']:.2f} px; "
        f"board at {res['raw']['board_distance_m']['min']:.2f}-{res['raw']['board_distance_m']['max']:.2f} m",
        *res["checks"],
    ]
    return "\n".join(lines)
