"""VGGT baseline (images only, no metric scale). Runs in pod/third_party/vggt/.venv.

API (VGGT repo): images = load_and_preprocess_images(paths); preds = model(images);
extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:]) (world-to-camera);
preds["depth"] (B,N,H,W,1), preds["depth_conf"] (B,N,H,W).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402


def main():
    a = common.parse("vggt", "facebook/VGGT-1B")
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    meta, paths, K = common.load(a.frames, a.max_frames)
    run = common.Run(a, "vggt")
    model = VGGT.from_pretrained(a.weights).to(dev).eval()
    for k, idx in enumerate(common.windows(len(paths), a.window, a.overlap)):
        t0 = time.time()
        ims = load_and_preprocess_images([str(paths[i]) for i in idx]).to(dev)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=(dev == "cuda")):
            preds = model(ims)
        ext, intr = pose_encoding_to_extri_intri(preds["pose_enc"], ims.shape[-2:])
        depth = preds["depth"][0].float().cpu().numpy()
        depth = depth[..., 0] if depth.ndim == 4 else depth
        common.save(run.out, k, idx, common.w2c_to_c2w(ext[0].float().cpu().numpy()), depth,
                    preds["depth_conf"][0].float().cpu().numpy(), intr[0].float().cpu().numpy())
        run.window_done(k, len(idx), time.time() - t0)
    run.finish(len(paths))


if __name__ == "__main__":
    main()
