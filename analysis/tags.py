"""GT2: AprilTag (tag36h11) poses in every N-th video frame (plus every map keyframe) -> tags table.

Detection runs on the distorted upright frame; the corners are undistorted with the display calibration and the
pose comes from solvePnP (IPPE_SQUARE, no distortion on undistorted points). Columns: take, frame, t_video,
t_host, id, kind, size_m, tx, ty, tz (camera frame, m), rvec, reproj_px, view_deg (angle between the tag normal
and the line of sight; poses beyond ~70 deg are ambiguous), u, v (centre pixel)."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import common as C


def keyframes(take: str) -> set[int]:
    p = C.DATA / "derived" / "maps-v1" / f"frames_{take}.json"
    if not p.exists():
        return set()
    return {int(f["video_frame"]) for f in json.loads(p.read_text())["frames"]}


def _detect_chunk(args):
    import av
    import cv2
    take, t0, t1, step, forced, clahe = args
    K, D, W = C.camera_K()
    sizes = C.tag_sizes()
    s = C.sync_of(take)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
                                  cv2.aruco.DetectorParameters())
    cl = cv2.createCLAHE(2.0, (8, 8)) if clahe else None
    rows = []
    with av.open(str(C.DATA / "takes" / take / "video.mp4")) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        fps = float(vs.average_rate)
        if t0 > 0:
            c.seek(int(max(t0 - 1.0, 0) / vs.time_base), stream=vs, backward=True)
        for fr in c.decode(vs):
            tv = float(fr.time or 0.0)
            if tv < t0 - 1e-6:
                continue
            if tv >= t1:
                break
            fi = int(round(tv * fps))
            if fi % step and fi not in forced:
                continue
            g = fr.to_ndarray(format="gray")
            rot = int(round(getattr(fr, "rotation", 0) or 0))
            if rot:
                g = np.ascontiguousarray(np.rot90(g, rot // 90))
            if cl is not None:
                g = cl.apply(g)
            corners, ids, _ = det.detectMarkers(g)
            if ids is None:
                continue
            for cn, i in zip(corners, ids.ravel()):
                i = int(i)
                if i not in sizes:
                    continue
                sz = sizes[i]
                cn = cv2.cornerSubPix(g, cn.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
                und = cv2.undistortPoints(cn, K, D, P=K).reshape(4, 2)
                obj = np.array([[-sz / 2, sz / 2, 0], [sz / 2, sz / 2, 0], [sz / 2, -sz / 2, 0],
                                [-sz / 2, -sz / 2, 0]], np.float64)
                ok, rv, tvec = cv2.solvePnP(obj, und.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    continue
                proj, _ = cv2.projectPoints(obj, rv, tvec, K, None)
                err = float(np.sqrt(((proj.reshape(4, 2) - und) ** 2).sum(1).mean()))
                R, _ = cv2.Rodrigues(rv)
                tv3 = tvec.ravel()
                cosang = abs(float(R[:, 2] @ (-tv3 / np.linalg.norm(tv3))))
                ctr = cn.reshape(4, 2).mean(0)
                rows.append(dict(take=take, frame=fi, t_video=tv, t_host=float(s.video_to_host(tv)), id=i,
                                 kind=C.TAG_KIND.get(i, "other"), size_m=sz, tx=tv3[0], ty=tv3[1], tz=tv3[2],
                                 rx=float(rv.ravel()[0]), ry=float(rv.ravel()[1]), rz=float(rv.ravel()[2]), reproj_px=err,
                                 view_deg=float(np.degrees(np.arccos(min(1.0, cosang)))),
                                 side_px=float(np.linalg.norm(cn.reshape(4, 2)[0] - cn.reshape(4, 2)[1])),
                                 u=float(ctr[0]), v=float(ctr[1])))
    return rows


def video_span(take: str) -> float:
    import av
    with av.open(str(C.DATA / "takes" / take / "video.mp4")) as c:
        vs = c.streams.video[0]
        return float(vs.duration * vs.time_base) if vs.duration else float(c.duration / 1e6)


def run(takes=None, step: int = 2, workers: int = 4, clahe: bool = True, seconds: float | None = None):
    import pandas as pd
    takes = C.available_takes(takes)
    out = []
    for take in takes:
        T = video_span(take)
        t_start = 0.0
        if C.QUICK:                                # quick: a short span in the middle of the first scene part
            recs = C.recordings(take)
            r = recs[min(3, len(recs) - 1)]
            t_start = float(C.sync_of(take).host_to_video(r.t0 + r.lead_in + 1.0))
            T = t_start + (seconds or 6.0)
        forced = keyframes(take)
        n = max(1, workers)
        n_chunks = n if C.QUICK else max(n, int((T - t_start) // 10))        # ~10 s of video per chunk
        edges = np.linspace(t_start, T + 0.1, n_chunks + 1)
        jobs = [(take, edges[i], edges[i + 1], step, forced, clahe) for i in range(n_chunks)]
        C.log(f"tags {take}: {T - t_start:.0f} s of video, step {step}, {len(forced)} keyframes forced, {n} workers")
        rows = []
        pb = C.bar(total=n_chunks, desc=f"tags {take}", unit="chunk")
        if n == 1:
            for j in jobs:
                rows += _detect_chunk(j)
                pb.update()
        else:
            from concurrent.futures import as_completed
            with ProcessPoolExecutor(n) as ex:
                for fu in as_completed([ex.submit(_detect_chunk, j) for j in jobs]):
                    rows += fu.result()
                    pb.update()
        pb.close()
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["frame", "id"]).drop_duplicates(["frame", "id"])
            C.log(f"  {df.frame.nunique()} frames with tags; per id: "
                  + ", ".join(f"{i}:{n}" for i, n in df.id.value_counts().sort_index().items()))
        out.append(df)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    C.save_table(df, "tags", takes=takes)
    return df
