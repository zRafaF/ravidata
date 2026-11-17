"""Pieces of the thesis' overview figure (run after fig_quadro and fig_estatico): a deck of keyframes of the map
window around fig_quadro's instant, and a close-up of an AprilTag in one of them with its corners and axes from PnP
(JSON, in the crop's pixel coordinates). Text-free PNGs."""
from __future__ import annotations

import json

import cv2
import numpy as np

from .. import common as C
from ..maps import TakeMaps
from . import common as P

TAKE = "indoor_course"
CROP = 150


def run():
    q = json.loads((P.out() / "fq_dados.json").read_text(encoding="utf-8"))
    i0 = int(q["keyframe"])
    tm = TakeMaps(TAKE, "pi3x")
    rgb_dir = C.DATA / "derived" / "maps-v1" / "frames" / TAKE / "rgb"
    kfs = [int(k) for k in tm.kf if abs(int(k) - i0) <= 12][::3][:6]
    for j, k in enumerate(kfs):
        im = cv2.imread(str(rgb_dir / f"{k:06d}.jpg"))
        b = int(0.03 * im.shape[0])
        P.save_png(f"hero_kf{j}", cv2.cvtColor(cv2.resize(im[b:-b, b:-b], (220, 220)), cv2.COLOR_BGR2RGB))
    # an AprilTag detected directly in the keyframes near the instant (largest one), pose by PnP with the known size
    K = tm.K
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    sizes = C.tag_sizes()
    best = None
    for k in sorted(tm.kf, key=lambda k: abs(int(k) - i0))[:80]:
        im = cv2.cvtColor(cv2.imread(str(rgb_dir / f"{int(k):06d}.jpg")), cv2.COLOR_BGR2RGB)
        corners, ids, _ = det.detectMarkers(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
        for cn, tid in zip(corners or [], ids.ravel() if ids is not None else []):
            cn = cn.reshape(4, 2); side = float(np.linalg.norm(cn[0] - cn[1]))
            cx, cy = cn.mean(0)
            if int(tid) in sizes and CROP * 0.6 < cx < tm.res - CROP * 0.6 and CROP * 0.6 < cy < tm.res - CROP * 0.6 \
                    and side > 22 and (best is None or side > best[0]):
                best = (side, int(k), int(tid), cn, im)
        if best is not None and best[0] > 40:
            break
    side, kt, tid, cn, im = best
    hs = sizes[tid] / 2
    obj = np.array([[-hs, hs, 0], [hs, hs, 0], [hs, -hs, 0], [-hs, -hs, 0]], np.float64)
    ok, rv, tv = cv2.solvePnP(obj, cn.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    ax = np.array([[0, 0, 0], [2 * hs, 0, 0], [0, 2 * hs, 0], [0, 0, 2 * hs]], np.float64)
    axp, _ = cv2.projectPoints(ax, rv, tv, K, None); axp = axp.reshape(-1, 2)
    c = cn.mean(0); x0 = int(np.clip(c[0] - CROP / 2, 0, im.shape[1] - CROP)); y0 = int(np.clip(c[1] - CROP / 2, 0, im.shape[0] - CROP))
    P.save_png("hero_tag", im[y0:y0 + CROP, x0:x0 + CROP])
    r = type("R", (), dict(id=tid, kf=kt))
    tv = tv.ravel()
    P.save_json("hero_dados", dict(keyframes=kfs, tag_id=int(r.id), tag_kf=int(r.kf), tag_dist_m=float(np.linalg.norm(tv)),
                                   corners=(cn - [x0, y0]) / CROP, axes=(axp - [x0, y0]) / CROP, crop=CROP))
    print("keyframes", kfs, "tag", int(r.id), "kf", int(r.kf), "dist", round(float(np.linalg.norm(tv)), 2))


if __name__ == "__main__":
    run()
