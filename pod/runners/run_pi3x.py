"""Pi3X runner (approximate metric; conditioned on our intrinsics). Runs in pod/third_party/pi3/.venv.

API (Pi3 repo, example_mm.py): model(imgs=(B,N,3,H,W) in [0,1], intrinsics=(B,N,3,3) pixels) ->
"camera_poses" (B,N,4,4) OpenCV cam2world, "local_points" (B,N,H,W,3) camera frame, "conf" (B,N,H,W,1) logits.
Pi3 is permutation-equivariant and meant to see all frames at once (--window 0) when memory allows; windows are
for 32 GB cards and are chained afterwards. If conditioning fails, it falls back to images only (logged).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main():
    a = common.parse("pi3x", "yyfz233/Pi3X")
    import torch
    from pi3.models.pi3x import Pi3X

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    meta, paths, K = common.load(a.frames, a.max_frames)
    run = common.Run(a, "pi3x")
    model = Pi3X.from_pretrained(a.weights).to(dev).eval()
    conditioned = True
    for k, idx in enumerate(common.windows(len(paths), a.window, a.overlap)):
        t0 = time.time()
        imgs = torch.from_numpy(common.images([paths[i] for i in idx])).permute(0, 3, 1, 2).float().div(255)
        imgs = imgs.to(dev)[None]
        Ks = torch.from_numpy(np.repeat(K[None], len(idx), 0)).to(dev)[None]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=(dev == "cuda")):
            try:
                res = model(imgs=imgs, intrinsics=Ks) if conditioned else model(imgs=imgs)
            except Exception as e:                       # conditioning API changed: images only
                print(f"[pi3x] intrinsics conditioning failed ({e!r}); images only", flush=True)
                conditioned = False
                if hasattr(model, "disable_multimodal"):
                    model.disable_multimodal()
                res = model(imgs=imgs)
        c2w = res["camera_poses"][0].float().cpu().numpy()
        local = res["local_points"][0].float().cpu().numpy()
        conf = torch.sigmoid(res["conf"][0].float()).cpu().numpy()
        conf = conf[..., 0] if conf.ndim == 4 else conf
        common.save(run.out, k, idx, c2w, local[..., 2], conf, np.repeat(K[None], len(idx), 0))
        run.window_done(k, len(idx), time.time() - t0)
    run.finish(len(paths), {"intrinsics_conditioning": conditioned})


if __name__ == "__main__":
    main()
