"""F05: phase of the horizontal pair vs sin(true azimuth): still tags (old method) and walking person (new). JSON only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .. import common as C, radar as R
from . import common as P

K = 2 * np.pi * R.D_OVER_LAMBDA
wrap = lambda a: np.angle(np.exp(1j * np.asarray(a, float)))


def _calib(version):
    return json.loads((C.RESULTS / f"analysis-{version}" / "tables" / "angle_calib.json").read_text())["az"]


def _panel(d, cal, extra=None):
    s = np.sin(np.radians(d.az_true.to_numpy(float)))
    line = lambda x: cal["sign"] * (K * x + cal["offset_rad"])
    ph = d.ph_az.to_numpy(float)
    y = line(s) + wrap(ph - line(s))                 # unwrapped around the fitted line (display only)
    res = wrap(ph - line(s))
    pts = [{"x": float(a), "y": float(b), "raw": float(c)} for a, b, c in zip(s, y, ph)]
    xs = np.array([-0.55, 0.55])
    return {"points": pts, "line": {"x": xs, "y": line(xs)}, "sign": cal["sign"], "offset_rad": cal["offset_rad"],
            "R": cal["R"] if "R" in cal else None, "n": len(d),
            "resid_mean_abs_rad": float(np.abs(res).mean()), "resid_rms_rad": float(np.sqrt((res ** 2).mean()))}


def run():
    old, new = _calib("v1"), _calib("v2")
    t = pd.read_parquet(C.RESULTS / "analysis-v1" / "tables" / "calib_targets.parquet").reset_index(drop=True)
    m = pd.read_parquet(C.RESULTS / "analysis-v2" / "tables" / "calib_moving.parquet")
    n_all = len(m)
    m = m.sample(n=min(400, len(m)), random_state=1).reset_index(drop=True)
    left = _panel(t, old)
    links = []
    for _, g in t.groupby(["rec", "bin"]):
        if len(g) > 1:
            links.append([int(i) for i in g.index])
    left["links"] = links
    left["angle_deg_range"] = [float(t.az_true.min()), float(t.az_true.max())]
    right = _panel(m, new)
    right["n_total"] = n_all
    right["angle_deg_range"] = [float(m.az_true.min()), float(m.az_true.max())]
    P.save_json("f05_calibracao", {"d_over_lambda": R.D_OVER_LAMBDA, "tags": left, "moving": right})


if __name__ == "__main__":
    run()
