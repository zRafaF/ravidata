"""MapAnything runner (metric; given our undistorted intrinsics). Runs in pod/third_party/mapanything/.venv.

API (README): views = [{"img": (H,W,3) uint8 tensor, "intrinsics": (3,3)}]; preprocess_inputs(views);
model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", ...) -> per view
"camera_poses" (B,4,4) OpenCV cam2world, "depth_z" (B,H,W,1), "conf" (B,H,W), "intrinsics" (B,3,3),
"metric_scaling_factor" (B,).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main():
    a = common.parse("mapanything", "facebook/map-anything")
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta, paths, K = common.load(a.frames, a.max_frames)
    run = common.Run(a, "mapanything")
    model = MapAnything.from_pretrained(a.weights).to(dev).eval()
    scales = []
    for k, idx in enumerate(common.windows(len(paths), a.window, a.overlap)):
        t0 = time.time()
        imgs = common.images([paths[i] for i in idx])
        views = [{"img": torch.from_numpy(im), "intrinsics": torch.from_numpy(K)} for im in imgs]
        with torch.no_grad():
            preds = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True,
                                amp_dtype="bf16", apply_mask=True, mask_edges=True, apply_confidence_mask=False,
                                confidence_percentile=10)

        def cat(key):
            return np.concatenate([p[key].float().cpu().numpy() for p in preds])
        depth = cat("depth_z")
        depth = depth[..., 0] if depth.ndim == 4 else depth
        scales.append(float(cat("metric_scaling_factor").mean()))
        common.save(run.out, k, idx, cat("camera_poses"), depth, cat("conf"), cat("intrinsics"))
        run.window_done(k, len(idx), time.time() - t0)
    run.finish(len(paths), {"metric_scaling_factor_mean_per_window": scales})


if __name__ == "__main__":
    main()
