"""Shared by the map-model runners (each runs inside its model's own venv; numpy + Pillow + torch only).

Every runner reads ``<frames>/frames.json`` + ``rgb/*.jpg`` (pod/frames.py) and writes, per window,
``<out>/win_NNN.npz`` with:

- ``idx``    (n,)      frame numbers (``frames.json`` → ``frames[i]``)
- ``c2w``    (n, 4, 4) camera-to-world, OpenCV axes (x right, y down, z forward), in the model's own frame and scale
- ``depth``  (n, H, W) float16, z-depth in the model's units (0 = invalid)
- ``conf``   (n, H, W) float16, the model's confidence (higher = better; scale differs per model)
- ``K``      (n, 3, 3) intrinsics the model used / predicted, at the depth's resolution
plus ``<out>/run.json`` (model, weights, repo commit, settings, seconds, peak VRAM, frames, windows).
Windows overlap, so they can be chained afterwards; ``window 0`` = all frames in one pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def parse(model: str, default_weights: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=f"{model} runner")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=64, help="frames per pass; 0 = all at once")
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--weights", default=default_weights)
    ap.add_argument("--repo", default="", help="the model's git checkout (commit recorded in run.json)")
    return ap.parse_args()


def load(frames_dir: str, max_frames: int = 0) -> tuple[dict, list[Path], np.ndarray]:
    d = Path(frames_dir)
    meta = json.loads((d / "frames.json").read_text(encoding="utf-8"))
    paths = [d / "rgb" / f"{f['i']:06d}.jpg" for f in meta["frames"]]
    if max_frames:
        paths = paths[:max_frames]
    return meta, paths, np.array(meta["K"], np.float32)


def images(paths: list[Path]) -> np.ndarray:
    """(n, H, W, 3) uint8."""
    from PIL import Image
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])


def windows(n: int, size: int, overlap: int) -> list[np.ndarray]:
    if size <= 0 or size >= n:
        return [np.arange(n)]
    step = max(1, size - overlap)
    starts = list(range(0, max(1, n - overlap), step))
    out = [np.arange(s, min(n, s + size)) for s in starts]
    if out[-1][-1] != n - 1:
        out.append(np.arange(max(0, n - size), n))
    return out


def save(out: Path, k: int, idx, c2w, depth, conf, K):
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"win_{k:03d}.npz", idx=np.asarray(idx, np.int32),
                        c2w=np.asarray(c2w, np.float32), depth=np.asarray(depth, np.float16),
                        conf=np.asarray(conf, np.float16), K=np.asarray(K, np.float32))


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """(n, 3, 4) or (n, 4, 4) world-to-camera -> (n, 4, 4) camera-to-world."""
    n = w2c.shape[0]
    M = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    M[:, :3, :4] = w2c[:, :3, :4]
    return np.linalg.inv(M)


class Run:
    """Timing, peak VRAM and run.json."""

    def __init__(self, a: argparse.Namespace, model: str):
        import torch
        self.a, self.model, self.t0 = a, model, time.time()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.windows: list[dict] = []
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)

    def window_done(self, k: int, n: int, seconds: float):
        self.windows.append({"k": k, "frames": n, "seconds": round(seconds, 2)})
        print(f"[{self.model}] window {k}: {n} frames, {seconds:.1f} s", flush=True)

    def finish(self, frames: int, extra: dict | None = None):
        import torch
        commit = ""
        if self.a.repo:
            try:
                commit = subprocess.check_output(["git", "-C", self.a.repo, "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                pass
        info = {"model": self.model, "weights": self.a.weights, "repo_commit": commit, "frames": frames,
                "window": self.a.window, "overlap": self.a.overlap, "windows": self.windows,
                "seconds": round(time.time() - self.t0, 1),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if self.cuda else None,
                "gpu": torch.cuda.get_device_name(0) if self.cuda else "cpu", "torch": torch.__version__,
                **(extra or {})}
        (self.out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(f"[{self.model}] done: {frames} frames, {info['seconds']} s, peak VRAM {info['peak_vram_gb']} GB",
              flush=True)
