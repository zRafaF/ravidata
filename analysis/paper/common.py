"""Shared helpers of the thesis-figure exporters (no text is ever drawn into an image)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from .. import common as C


def out() -> Path:
    return C.out_dir("paper")


def to_rgb(a01, cmap="magma_r"):
    """(H, W) values in 0..1 -> (H, W, 3) uint8. Default map: white = low, dark = high (reads on a white page)."""
    import matplotlib
    m = matplotlib.colormaps[cmap](np.clip(np.nan_to_num(a01), 0, 1))[..., :3]
    return (m * 255).astype(np.uint8)


def save_png(name: str, rgb) -> Path:
    """rgb: (H, W, 3 or 4) uint8, row 0 = top. No margins, no text."""
    import cv2
    p = out() / f"{name}.png"
    rgb = np.asarray(rgb)
    code = cv2.COLOR_RGBA2BGRA if rgb.shape[2] == 4 else cv2.COLOR_RGB2BGR
    cv2.imwrite(str(p), cv2.cvtColor(rgb, code))
    _publish(p)
    return p


def save_json(name: str, obj) -> Path:
    """Sidecar of a panel (xlim, ylim, units, overlays in data coordinates) or the series of a native chart."""
    p = out() / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=lambda v: float(v) if np.isscalar(v) else np.asarray(v).tolist()),
                 encoding="utf-8")
    _publish(p)
    return p


def _publish(p: Path):
    """Copy to the thesis repo when TCC_FIGURES points at its figures/data folder."""
    d = os.environ.get("TCC_FIGURES")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, Path(d) / p.name)


def db_over_median(P, lo=3.0, hi=30.0, sl=slice(8, 110)):
    """Power (n, B) -> 0..1: dB over the frame's median, clipped to lo..hi."""
    L = 10 * np.log10(np.maximum(P, 1e-15))
    L = L - np.median(L[:, sl], 1, keepdims=True)
    return np.clip((L - lo) / (hi - lo), 0, 1)
