"""Frames for the map models: decode a take's video, undistort, tone-map Log -> display, resize, sample.

    uv run python pod/frames.py --data data/ravidata --take indoor_course --out work/frames/indoor_course \
        [--select motion|step] [--step 2] [--min-flow 0.04] [--res 518] [--max-frames N]

Writes ``<out>/rgb/000000.jpg …`` and ``<out>/frames.json``:
``{"take", "video", "res", "step", "K" (3x3, pixels of the written frames, undistorted pinhole), "frames":
[{"i": written index, "video_frame": index in the video, "t_video": s, "flow": …}, …], "tone": {…}}``.

- Which frames: keyframes by motion (default). Near-identical frames add nothing to a reconstruction and make the
  windows span less of the walk, so a frame is kept only when the view moved enough since the last kept one
  (median optical flow of tracked corners >= 4 % of the width), at least one per second when standing still.

- Orientation: frames are turned upright with the video's display rotation (the published videos are already
  upright; the phone originals are stored sideways).
- Intrinsics: ``calibration/camera.calib.json`` (``display`` block for upright frames), scaled to the video size;
  frames are undistorted with it, so ``K`` is a plain pinhole.
- Log -> display: one fixed curve per video (per-channel 0.5 / 99.5 percentiles of 24 frames spread over the
  video, a mild contrast curve, saturation x1.25). It is not colorimetric; it only gives the models a normal-looking
  picture, the same for every frame of the video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def calib_for(data: Path, w: int, h: int, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    cal = json.loads((data / "calibration" / "camera.calib.json").read_text(encoding="utf-8"))
    block = cal["raw"] if (rotation and rotation == cal.get("rotation_deg")) else cal["display"]
    cw, ch = block["image_size"]
    if abs(w / h - cw / ch) > 1e-3:
        raise SystemExit(f"video {w}x{h} vs calibration {cw}x{ch}: different aspect ratio")
    K = np.array(block["K"], float)
    K[:2] *= w / cw
    return K, np.array(block["dist"], float)


def upright(img: np.ndarray, rotation: int) -> np.ndarray:
    k = int(round(rotation / 90.0)) % 4
    return np.ascontiguousarray(np.rot90(img, k=k)) if k else img


def tone_params(video: Path, n: int = 24) -> dict:
    """Seek to n times spread over the video (no full decode) and take the colour percentiles."""
    import av
    samples = []
    with av.open(str(video)) as c:
        vs = c.streams.video[0]
        dur = float(vs.duration * vs.time_base) if vs.duration else float(c.duration or 0) / 1e6
        for t in np.linspace(0.05, 0.95, n) * dur:
            c.seek(int(t / vs.time_base), stream=vs, backward=True)
            fr = next(c.decode(vs), None)
            if fr is not None:
                samples.append(fr.to_ndarray(format="rgb24")[::4, ::4].reshape(-1, 3))
    x = np.concatenate(samples).astype(np.float32)
    return {"lo": np.percentile(x, 0.5, axis=0).tolist(), "hi": np.percentile(x, 99.5, axis=0).tolist(),
            "contrast": 0.35, "saturation": 1.25}


def tone(rgb: np.ndarray, p: dict) -> np.ndarray:
    import cv2
    lo, hi = np.array(p["lo"], np.float32), np.array(p["hi"], np.float32)
    x = np.clip((rgb.astype(np.float32) - lo) / np.maximum(hi - lo, 1.0), 0, 1)
    x = x + p["contrast"] * (x - 0.5) * (1 - np.abs(2 * x - 1))           # gentle S-curve around mid-grey
    hsv = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * p["saturation"], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class MotionGate:
    """Keyframe selection: keep a frame when the view has moved enough since the last kept one (median optical
    flow of tracked corners >= ``min_flow`` x image width), or ``max_gap`` s have passed; never closer than
    ``min_gap`` s. Standing still -> 1 frame per ``max_gap``; walking or turning -> more frames."""

    def __init__(self, min_flow: float = 0.04, min_gap: float = 0.1, max_gap: float = 1.0, size: int = 256):
        self.min_flow, self.min_gap, self.max_gap, self.size = min_flow, min_gap, max_gap, size
        self.ref, self.pts, self.t_ref = None, None, None

    def __call__(self, rgb: np.ndarray, t: float) -> tuple[bool, float]:
        import cv2
        g = cv2.cvtColor(cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        if self.ref is None:
            return self._keep(g, t), float("nan")
        dt = t - self.t_ref
        if dt < self.min_gap:
            return False, float("nan")
        flow = 1.0                                            # nothing to track = the view changed completely
        if self.pts is not None and len(self.pts) >= 20:
            nxt, ok, _ = cv2.calcOpticalFlowPyrLK(self.ref, g, self.pts, None, winSize=(21, 21), maxLevel=3)
            ok = ok.ravel() == 1
            if ok.sum() >= 20:
                flow = float(np.median(np.linalg.norm((nxt - self.pts)[ok].reshape(-1, 2), axis=1))) / self.size
        if flow >= self.min_flow or dt >= self.max_gap:
            return self._keep(g, t), flow
        return False, flow

    def _keep(self, g, t) -> bool:
        import cv2
        self.ref, self.t_ref = g, t
        self.pts = cv2.goodFeaturesToTrack(g, maxCorners=300, qualityLevel=0.01, minDistance=7)
        return True


def main(argv=None):
    import av
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="the downloaded dataset (folder with calibration/ and takes/)")
    ap.add_argument("--take", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--select", choices=("motion", "step"), default="motion",
                    help="motion: keyframes by optical flow (default); step: every --step-th frame")
    ap.add_argument("--step", type=int, default=2,
                    help="look at every N-th video frame (motion: candidates, 2 = 15 fps; step: 6 = 5 fps)")
    ap.add_argument("--min-flow", type=float, default=0.04, help="motion: median flow, fraction of the width")
    ap.add_argument("--min-gap", type=float, default=0.1, help="motion: at most one keyframe per this many s")
    ap.add_argument("--max-gap", type=float, default=1.0, help="motion: at least one keyframe per this many s")
    ap.add_argument("--res", type=int, default=518, help="output size (square; 518 = 37 x 14 for ViT patch 14)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N written frames (tests)")
    ap.add_argument("--video", help="override the video path (default <data>/takes/<take>/video.mp4)")
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    video = Path(a.video) if a.video else data / "takes" / a.take / "video.mp4"
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    tp = tone_params(video)
    gate = MotionGate(a.min_flow, a.min_gap, a.max_gap) if a.select == "motion" else None
    frames, K_out, maps, n_video = [], None, None, 0
    with av.open(str(video)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(vs)):
            n_video = i + 1
            if i % a.step:
                continue
            rot = int(round(getattr(fr, "rotation", 0) or 0))
            img = upright(fr.to_ndarray(format="rgb24"), rot)
            t = float(fr.time or 0.0)
            flow = float("nan")
            if gate is not None:
                keep, flow = gate(img, t)
                if not keep:
                    continue
            if maps is None:
                H, W = img.shape[:2]
                K, D = calib_for(data, W, H, 0)            # frames are upright now: the "display" intrinsics
                if W != H:
                    raise SystemExit("expected square frames (1:1 video)")
                maps = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
                K_out = K.copy()
                K_out[:2] *= a.res / W
            img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            img = cv2.resize(tone(img, tp), (a.res, a.res), interpolation=cv2.INTER_AREA)
            k = len(frames)
            cv2.imwrite(str(out / "rgb" / f"{k:06d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            frames.append({"i": k, "video_frame": i, "t_video": t,
                           **({} if np.isnan(flow) else {"flow": round(flow, 4)})})
            if a.max_frames and len(frames) >= a.max_frames:
                break
    sel = ({"select": "motion", "min_flow": a.min_flow, "min_gap_s": a.min_gap, "max_gap_s": a.max_gap}
           if gate is not None else {"select": "step"})
    meta = {"take": a.take, "video": video.name, "res": a.res, "step": a.step, **sel, "n_video_frames": n_video,
            "K": K_out.tolist(), "undistorted": True, "tone": tp, "frames": frames}
    (out / "frames.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    dur = frames[-1]["t_video"] - frames[0]["t_video"] if len(frames) > 1 else 0.0
    print(f"{a.take}: {len(frames)} frames over {dur:.0f} s ({len(frames) / max(dur, 1e-6):.1f} per s) -> {out}")

if __name__ == "__main__":
    main()
