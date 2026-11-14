"""Depth Anything 3 runner (nested giant-large: depth in metres). Runs in pod/third_party/da3/.venv.

API (README): DepthAnything3.from_pretrained(id); pred = model.inference(images) -> pred.depth (N,H,W),
pred.conf (N,H,W), pred.extrinsics (N,3,4) OpenCV world-to-camera, pred.intrinsics (N,3,3).
Our intrinsics are passed if the installed version accepts them (pose / intrinsics conditioning); otherwise
images only (logged in run.json).
"""

from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main():
    a = common.parse("da3", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    import torch
    from depth_anything_3.api import DepthAnything3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, paths, K = common.load(a.frames, a.max_frames)
    run = common.Run(a, "da3")
    model = DepthAnything3.from_pretrained(a.weights).to(dev).eval()
    takes_K = "intrinsics" in inspect.signature(model.inference).parameters
    for k, idx in enumerate(common.windows(len(paths), a.window, a.overlap)):
        t0 = time.time()
        ims = [im for im in common.images([paths[i] for i in idx])]
        kw = {"intrinsics": np.repeat(K[None], len(idx), 0)} if takes_K else {}
        with torch.no_grad():
            pred = model.inference(ims, **kw)
        common.save(run.out, k, idx, common.w2c_to_c2w(np.asarray(pred.extrinsics)), np.asarray(pred.depth),
                    np.asarray(pred.conf), np.asarray(pred.intrinsics))
        run.window_done(k, len(idx), time.time() - t0)
    run.finish(len(paths), {"intrinsics_given": takes_K})


if __name__ == "__main__":
    main()
