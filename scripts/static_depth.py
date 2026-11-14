"""Metric depth for the static takes (radar still, camera still): a monocular depth model on the CPU, frame by
frame, grounded on the AprilTags.

    uv run --group depth python scripts/static_depth.py --take static_pass1
    make static-depth TAKE=static_pass2 PER_PART=8

The camera does not move in the static takes, so a 3D model of the scene needs no multi-view reconstruction: one
depth map per chosen frame is enough, and many frames of the same view can be compared (stability) or averaged.

1. Frames: ``PER_PART`` frames spread over each part (S0 empty scene, S1 objects, S2 things moving) of the take,
   from the video times of the radar recordings (``take.json``: beeps + sync; the sync lead-in is skipped).
2. Each frame: upright, undistorted with the calibration (full 1440 x 1440), the same Log -> display tone curve
   as the pod (``pod/frames.py``).
3. Depth: a monocular depth model through ``transformers`` on the CPU: ``--model depthpro`` (default, Apple Depth
   Pro: metric, sharp object boundaries, about a minute per frame on a CPU), ``da2l`` / ``da2s`` (Depth Anything V2
   Metric Indoor Large / Small; Small is fast but blurs chairs and loses the brooms) or any ``depth-estimation``
   model id.
4. Grounding: every AprilTag (tag36h11, printed sizes in ``docs/protocols/targets/tags.json``) is located with
   solvePnP (IPPE_SQUARE); a 5 x 5 grid of points on each tag gives true depths; scale = median over tags of
   (true / model depth) (``--fit scale``, default) or scale + offset by least squares (``--fit affine``). Broom tags
   (20, 21, 50, 51) are reported but not used for the fit: the handles are too thin for the models. The residual
   per tag is reported (leave-one-tag-out as well, so the check is honest).

Outputs in ``data/ravidata/derived/static-depth-v1/<model>/<take>/``: per frame ``<part>_<t>.npz`` (metric depth 720 x 720
float16, K, model, fit, tags) + ``.jpg`` + ``.ply`` (coloured points, opens in CloudCompare / MeshLab), and
``summary.json`` (fit and per-tag errors of every frame).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pod"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames as F  # noqa: E402  (pod/frames.py: upright, calib_for, tone_params, tone)

DEFAULT_MODEL = "depthpro"
MODELS = {                                          # short name -> transformers model id
    "da2s": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",   # ~25 M params, seconds per frame, blurry
    "da2l": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",   # ~335 M params
    "depthpro": "apple/DepthPro-hf",                                     # sharp boundaries, metric, ~1 min / frame CPU
}
THIN_TAGS = {20, 21, 50, 51}                        # tags on the brooms: the handle is too thin for a depth model


def part_times(take: dict, per_part: int, offset: float) -> list[tuple[str, float]]:
    """(part, video time) for `per_part` frames spread over each part (union of its recordings)."""
    a, b = take["sync"]["video_a"], take["sync"]["video_b"]
    spans: dict[str, list[tuple[float, float]]] = {}
    for r in take["recordings"]:
        t0 = (r["t_beep_start"] - b - offset) / a + float(r.get("lead_in_s") or 0.0)
        t1 = (r["t_beep_end"] - b - offset) / a
        if t1 - t0 > 1.0:
            spans.setdefault(r.get("part") or "free", []).append((t0 + 0.5, t1 - 0.5))
    out = []
    for part, sp in spans.items():
        total = sum(t1 - t0 for t0, t1 in sp)
        for q in (np.arange(per_part) + 0.5) / per_part * total:   # evenly over the part's recordings
            for t0, t1 in sp:
                if q <= t1 - t0:
                    out.append((part, t0 + q))
                    break
                q -= t1 - t0
    return sorted(out, key=lambda x: x[1])


def grab(video: Path, times: list[float]) -> list[tuple[float, np.ndarray]]:
    import av
    out = []
    with av.open(str(video)) as c:
        vs = c.streams.video[0]
        for t in times:
            c.seek(int(max(t - 0.5, 0) / vs.time_base), stream=vs, backward=True)
            fr = None
            for f in c.decode(vs):
                fr = f
                if (f.time or 0) >= t:
                    break
            if fr is not None:
                rot = int(round(getattr(fr, "rotation", 0) or 0))
                out.append((float(fr.time or t), F.upright(fr.to_ndarray(format="rgb24"), rot)))
    return out


def tag_sizes() -> dict[int, float]:
    d = json.loads((ROOT / "docs" / "protocols" / "targets" / "tags.json").read_text(encoding="utf-8"))
    return {int(k): v["size_mm"] / 1000.0 for k, v in d["tags"].items()}


def detect_tags(gray: np.ndarray, K: np.ndarray, sizes: dict[int, float]) -> list[dict]:
    import cv2
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i not in sizes:
            continue
        s = sizes[i]
        c = cv2.cornerSubPix(gray, c.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)).reshape(4, 2)
        obj = np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        g = np.linspace(-0.4, 0.4, 5) * s                             # 5 x 5 grid inside the tag
        G = np.array([[x, y, 0.0] for y in g for x in g])
        Xc = G @ R.T + tvec.ravel()
        uv = Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        out.append({"id": i, "size_m": s, "distance_m": float(np.linalg.norm(tvec)), "uv": uv, "z": Xc[:, 2],
                    "corners": c})
    return out


def sample(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    import cv2
    m = cv2.remap(depth.astype(np.float32), uv[:, 0].astype(np.float32).reshape(-1, 1),
                  uv[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR, borderValue=np.nan)
    return m.ravel()


def fit(tags: list[dict], depth: np.ndarray, mode: str, exclude=THIN_TAGS) -> tuple[tuple[float, float], list[dict]]:
    """depth_metric = a * depth_model + b (b = 0 for "scale"); per tag the relative error with the fit on all tags
    and leave-one-tag-out. Scale = median of the per-tag ratios (one vote per tag, robust to one bad tag); tags in
    `exclude` (thin objects the model cannot resolve) are reported but not used, unless they are all there is."""
    def ratio(t):
        m = np.isfinite(t["d"]) & (t["d"] > 0)
        return float(np.median(t["z"][m] / t["d"][m])) if m.sum() >= 5 else np.nan

    def solve(ts):
        ts = [t for t in ts if np.isfinite(ratio(t))]
        if not ts:
            return None
        if mode == "affine" and len({t["id"] for t in ts}) >= 2:
            z = np.concatenate([t["z"] for t in ts]); d = np.concatenate([t["d"] for t in ts])
            m = np.isfinite(d) & (d > 0)
            A = np.stack([d[m], np.ones(m.sum())], 1)
            return tuple(np.linalg.lstsq(A, z[m], rcond=None)[0])
        return (float(np.median([ratio(t) for t in ts])), 0.0)

    for t in tags:
        t["d"] = sample(depth, t["uv"])
    used = [t for t in tags if t["id"] not in exclude] or tags
    ab = solve(used)
    rows = []
    if ab is None:
        return (np.nan, np.nan), rows
    for t in tags:
        others = [u for u in used if u is not t]
        ab_loo = solve(others) if others else None
        def err(p):
            if p is None:
                return None
            pred = p[0] * t["d"] + p[1]
            m = np.isfinite(pred)
            return float(np.median(np.abs(pred[m] - t["z"][m]) / t["z"][m])) if m.any() else None
        rows.append({"id": t["id"], "distance_m": round(t["distance_m"], 3), "ratio": ratio(t),
                     "used_in_fit": t in used, "rel_err": err(ab), "rel_err_leave_one_out": err(ab_loo)})
    return ab, rows


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\n"
            f"property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T
    rec["r"], rec["g"], rec["b"] = rgb.T
    with open(path, "wb") as fh:
        fh.write(head.encode("ascii")); fh.write(rec.tobytes())


def main(argv=None):
    import cv2
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", default=str(ROOT / "data" / "ravidata"))
    ap.add_argument("--take", default="static_pass1", help="a static take, or 'all' (both passes + the pilot)")
    ap.add_argument("--per-part", type=int, default=5, help="frames per part (S0, S1, S2)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"short name ({', '.join(MODELS)}) or any transformers depth-estimation model id")
    ap.add_argument("--fit", choices=("scale", "affine", "auto"), default="auto",
                    help="auto: per frame, the fit with the lower median error over every tag (the broom tags, not "
                         "used for fitting, are a held-out check)")
    ap.add_argument("--refit", action="store_true",
                    help="no model: re-ground the depth maps already saved for --model (new fit, same depth)")
    ap.add_argument("--out-res", type=int, default=720)
    a = ap.parse_args(argv)
    if a.take == "all":                         # every static take (radar and camera still)
        for t in ("static_pass1", "static_pass2", "pilot_chairs"):
            main([*(argv if argv is not None else sys.argv[1:]), "--take", t])
        return
    data = Path(a.data)
    tdir = data / "takes" / a.take
    take = json.loads((tdir / "take.json").read_text(encoding="utf-8"))
    video = tdir / "video.mp4"
    if not video.exists():
        raise SystemExit(f"{video} missing: uv run hf download DoninhaD/ravidata --repo-type dataset "
                         f"--local-dir data/ravidata --include \"takes/{a.take}/*\" --include \"calibration/*\"")
    short = a.model if a.model in MODELS else a.model.split("/")[-1]
    a.model = MODELS.get(a.model, a.model)
    out = data / "derived" / "static-depth-v1" / short / a.take
    out.mkdir(parents=True, exist_ok=True)
    offset = float(take["sync"].get("recommended_video_offset_s", 0.0))
    picks = part_times(take, a.per_part, offset)
    print(f"{a.take}: {len(picks)} frames ({', '.join(sorted({p for p, _ in picks}))}); model {a.model}")
    frames = grab(video, [t for _, t in picks])
    H, W = frames[0][1].shape[:2]
    K, D = F.calib_for(data, W, H, 0)
    maps = cv2.initUndistortRectifyMap(K, D, None, K, (W, H), cv2.CV_32FC1)
    tp = F.tone_params(video)
    sizes = tag_sizes()

    pipe = None
    if not a.refit:
        import torch
        from PIL import Image
        from transformers import pipeline
        torch.set_num_threads(max(1, torch.get_num_threads()))
        pipe = pipeline("depth-estimation", model=a.model, device="cpu")

    part_of = {}
    for (pp, _), (tt_, _) in zip(picks, frames):
        part_of[tt_] = pp

    def picks_name(t):
        return f"{part_of[t]}_{t:07.2f}s"

    def saved_model_depth(t):
        z = np.load(out / f"{picks_name(t)}.npz", allow_pickle=True)
        if "depth_model" in z.files:
            d = z["depth_model"].astype(np.float32)
        else:                                            # older files: undo the saved fit
            d = (z["depth"].astype(np.float32) - float(z["b"])) / float(z["a"])
        return cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)

    def best_fit(tags, dm):
        if a.fit != "auto":
            return fit(tags, dm, a.fit)
        cands = []
        for mode in ("scale", "affine"):
            ab, rows_ = fit([dict(t) for t in tags], dm, mode)
            errs = [r["rel_err"] for r in rows_ if r["rel_err"] is not None]
            cands.append((np.median(errs) if errs else np.inf, mode, ab, rows_))
        e, mode, ab, rows_ = min(cands, key=lambda c: c[0])
        for r in rows_:
            r["fit"] = mode
        return ab, rows_

    rows, res = [], []
    s_out = a.out_res / W
    Ko = K.copy(); Ko[:2] *= s_out
    for (part, t_want), (t, rgb) in zip(picks, frames):                 # pass 1: depth + per-frame fit
        t0 = time.time()
        und = cv2.remap(rgb, maps[0], maps[1], cv2.INTER_LINEAR)
        img = F.tone(und, tp)
        tags = detect_tags(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K, sizes)
        if pipe is None:
            dm = saved_model_depth(t)
        else:
            pred = pipe(Image.fromarray(img))["predicted_depth"]
            dm = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            dm = cv2.resize(dm, (W, H), interpolation=cv2.INTER_LINEAR)
        (sa, sb), tag_rows = best_fit(tags, dm)
        res.append((part, t, img, dm, sa, sb, tag_rows, time.time() - t0))
        print(f"  {part} {t:7.2f} s: {len(tags)} tags {sorted(r['id'] for r in tag_rows)}, "
              f"scale {sa:.3f} ({time.time() - t0:.1f} s)", flush=True)
    fitted = [(r[4], r[5]) for r in res if np.isfinite(r[4])]
    take_ab = (float(np.median([f[0] for f in fitted])), float(np.median([f[1] for f in fitted]))) if fitted else None
    if take_ab is None:
        print("  no AprilTag seen in any frame: depths left in the model's own units")
    for part, t, img, dm, sa, sb, tag_rows, secs in res:                  # pass 2: ground + save
        how = "own tags"
        if not np.isfinite(sa):
            sa, sb, how = (*take_ab, "take median (no tag in this frame; the camera is still)") if take_ab else \
                (np.nan, np.nan, "none")
        metric = sa * dm + sb if np.isfinite(sa) else dm
        name = f"{part}_{t:07.2f}s"
        small = cv2.resize(metric, (a.out_res, a.out_res), interpolation=cv2.INTER_AREA)
        im_s = cv2.resize(img, (a.out_res, a.out_res), interpolation=cv2.INTER_AREA)
        dm_s = cv2.resize(dm, (a.out_res, a.out_res), interpolation=cv2.INTER_AREA)
        fit_used = tag_rows[0].get("fit", a.fit) if tag_rows else a.fit
        np.savez_compressed(out / f"{name}.npz", depth=small.astype(np.float16), depth_model=dm_s.astype(np.float16),
                            K=Ko, t_video=t, part=part, model=a.model, fit=fit_used, a=sa, b=sb, grounded=how)
        cv2.imwrite(str(out / f"{name}.jpg"), cv2.cvtColor(im_s, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        vv, uu = np.mgrid[0:a.out_res:2, 0:a.out_res:2]
        z = small[vv, uu]
        ok = np.isfinite(z) & (z > 0)
        xyz = np.stack([(uu[ok] - Ko[0, 2]) / Ko[0, 0] * z[ok], (vv[ok] - Ko[1, 2]) / Ko[1, 1] * z[ok], z[ok]], 1)
        write_ply(out / f"{name}.ply", xyz.astype(np.float32), im_s[vv, uu][ok])
        errs = [r["rel_err_leave_one_out"] for r in tag_rows if r["rel_err_leave_one_out"] is not None]
        rows.append({"frame": name, "part": part, "t_video": round(t, 3), "tags": tag_rows, "a": sa, "b": sb,
                     "grounded": how, "seconds": round(secs, 1)})
        print(f"  {name}: scale {sa:.3f}" + (f" offset {sb:+.3f}" if sb else "") + f" ({fit_used}) [{how}]"
              + (f", leave-one-tag-out error median {np.median(errs) * 100:.1f} %" if errs else "")
              + ("; true/model per tag " + " ".join(f"{r['id']}{'' if r['used_in_fit'] else '*'}:{r['ratio']:.2f}"
                                                     for r in tag_rows) if tag_rows else ""), flush=True)
    summary = {"take": a.take, "model": a.model, "fit": a.fit, "per_part": a.per_part, "frames": rows,
               "how": "depth_metric = a * depth_model + b, fitted on 5 x 5 points per AprilTag (solvePnP, IPPE_SQUARE)"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
