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


_T = {"load_s": 0.0, "save_s": 0.0}     # time spent in images() / save(), split out of each window's time


def images(paths: list[Path]) -> np.ndarray:
    """(n, H, W, 3) uint8."""
    from PIL import Image
    t = time.time()
    x = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths])
    _T["load_s"] += time.time() - t
    return x


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
    t = time.time()
    _save(out, k, idx, c2w, depth, conf, K)
    _T["save_s"] += time.time() - t


def _save(out: Path, k: int, idx, c2w, depth, conf, K):
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
    """Every running metric, unfiltered (for the paper): per window the frames, wall time split into image loading
    / model (inference + host copies) / saving, frames per second, peak GPU memory allocated and reserved; per run
    the model loading time, totals, settings, versions, GPU and host. Written to run.json after every window, so
    a crash keeps what was measured."""

    def __init__(self, a: argparse.Namespace, model: str):
        import torch
        self.a, self.model, self.t0 = a, model, time.time()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.windows: list[dict] = []
        self.model_load_s: float | None = None
        self.peak_alloc = self.peak_reserved = 0.0
        self._T0 = dict(_T)
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)

    def window_done(self, k: int, n: int, seconds: float):
        import torch
        if self.model_load_s is None:                       # everything before the first window started
            self.model_load_s = round(time.time() - self.t0 - seconds, 2)
        load, save = _T["load_s"] - self._T0["load_s"], _T["save_s"] - self._T0["save_s"]
        self._T0 = dict(_T)
        w = {"k": k, "frames": n, "seconds": round(seconds, 3), "load_s": round(load, 3),
             "model_s": round(max(seconds - load - save, 0.0), 3), "save_s": round(save, 3),
             "fps": round(n / max(seconds, 1e-9), 3), "s_per_frame": round(seconds / max(n, 1), 4)}
        if self.cuda:
            alloc, res = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
            w.update(peak_alloc_gb=round(alloc, 3), peak_reserved_gb=round(res, 3))
            self.peak_alloc, self.peak_reserved = max(self.peak_alloc, alloc), max(self.peak_reserved, res)
            torch.cuda.reset_peak_memory_stats()
        self.windows.append(w)
        print(f"[{self.model}] window {k}: {n} frames, {seconds:.1f} s ({w['model_s']:.1f} model), "
              f"peak {w.get('peak_alloc_gb', 0):.1f} GB", flush=True)
        self._write(status="running")

    def finish(self, frames: int, extra: dict | None = None):
        self.extra = extra or {}
        info = self._write(status="done", frames=frames)
        print(f"[{self.model}] done: {frames} frames, {info['seconds']} s, peak VRAM {info['peak_vram_gb']} GB",
              flush=True)

    def _write(self, status: str, frames: int | None = None) -> dict:
        import platform
        import sys
        import torch
        commit = ""
        if self.a.repo:
            try:
                commit = subprocess.check_output(["git", "-C", self.a.repo, "rev-parse", "HEAD"], text=True).strip()
            except Exception:
                pass
        W = self.windows
        tot = {k: round(sum(w[k] for w in W), 2) for k in ("seconds", "load_s", "model_s", "save_s")} if W else {}
        nf = sum(w["frames"] for w in W)
        info = {"model": self.model, "status": status, "weights": self.a.weights, "repo_commit": commit,
                "frames": frames if frames is not None else nf, "frames_processed": nf,
                "window": self.a.window, "overlap": self.a.overlap, "max_frames": self.a.max_frames,
                "seconds": round(time.time() - self.t0, 1), "model_load_s": self.model_load_s,
                "windows_total": tot, "fps_model": round(nf / max(tot.get("model_s", 0), 1e-9), 3) if W else None,
                "peak_vram_gb": round(self.peak_alloc, 2) if self.cuda else None,
                "peak_reserved_gb": round(self.peak_reserved, 2) if self.cuda else None,
                "gpu": torch.cuda.get_device_name(0) if self.cuda else "cpu",
                "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if self.cuda else None,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version() if self.cuda else None,
                "python": sys.version.split()[0], "host": platform.platform(), "argv": sys.argv,
                "windows": W, **getattr(self, "extra", {})}
        (self.out / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        return info
