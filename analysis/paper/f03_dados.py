"""F03: the dataset at a glance. One video frame per take (text-free, 360 px) and a JSON with what each take holds
(still / carried, recordings, radar frames, configurations, truth source) for the native info blocks."""
from __future__ import annotations

import re

import cv2
import numpy as np

from .. import common as C
from . import common as PC

FRAME_AT_S = dict(pilot_chairs=51.0, static_pass1=243.0, static_pass2=116.0, indoor_course=95.0,
                  outdoor_course=203.0)          # video seconds of the frame shown
SIDE = 360


def _config(name: str) -> dict:
    dim = "2D" if "-2d-" in name else "1D"
    m = re.search(r"-(\d+)fps", name)
    return dict(name=name, dim=dim, fps=int(m.group(1)) if m else None)


def run():
    T = C.load_table("truth_frames")
    takes = []
    for take in C.available_takes():
        j = C.take_json(take)
        recs = j["recordings"]
        cap = cv2.VideoCapture(str(C.DATA / "takes" / take / "video.mp4"))
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-9)
        cap.set(cv2.CAP_PROP_POS_MSEC, FRAME_AT_S[take] * 1000)
        ok, fr = cap.read()
        cap.release()
        assert ok, take
        h, w = fr.shape[:2]
        s = SIDE / max(h, w)
        fr = cv2.resize(fr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        PC.save_png(f"f03_{take}", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cfgs = []
        for name in dict.fromkeys(r["config"] for r in recs):
            sel = [r for r in recs if r["config"] == name]
            cfgs.append(dict(_config(name), recordings=len(sel), frames=int(sum(r["frames"] for r in sel))))
        src = sorted(set(T[T["take"] == take].source.astype(str)))
        takes.append(dict(
            take=take, image=f"f03_{take}", image_px=[fr.shape[1], fr.shape[0]], frame_at_video_s=FRAME_AT_S[take],
            kind=j["kind"], place=j.get("place", "closed"), recordings=len(recs),
            radar_frames=int(sum(r["frames"] for r in recs)), video_s=float(dur), configs=cfgs,
            truth_source=src, truth="map" if any(x.startswith("map") for x in src) else "depth"))
        print(take, takes[-1]["kind"], takes[-1]["recordings"], takes[-1]["radar_frames"], src)
    PC.save_json("f03_dados", dict(
        takes=takes, total_recordings=int(sum(t["recordings"] for t in takes)),
        total_radar_frames=int(sum(t["radar_frames"] for t in takes)),
        total_video_s=float(np.sum([t["video_s"] for t in takes]))))


if __name__ == "__main__":
    run()
