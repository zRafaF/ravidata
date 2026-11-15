"""F10: truth distance histogram (indoor_course) and plain vs balanced found rate per tuned pipeline. JSON only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import common as C, score as S
from . import common as P

TAKE = "indoor_course"
FAMILIES = ("thr", "ca_cfar", "rd_cfar", "grid")


def run():
    T = C.RESULTS / "analysis-v2" / "tables"
    tf = pd.read_parquet(T / "truth_frames.parquet")
    tf = tf[(tf["take"] == TAKE) & tf.arc_complete.astype(bool)]
    a = tf.arc_r.to_numpy(float)
    a = a[np.isfinite(a)]
    edges = np.arange(0.3, 3.3001, 0.1)
    cnt, _ = np.histogram(a, bins=edges)
    cs = np.arange(0.3, 4.1, 0.02)
    hit = np.array([(np.abs(a - c) <= 0.25).mean() for c in cs])
    c = float(cs[int(hit.argmax())])
    F = pd.read_parquet(T / "eval_frames_tuned_pooled_arc.parquet")
    F = F[F["take"] == TAKE]
    pipes = []
    for fam in FAMILIES:
        s = S.summary(F[F.family == fam])
        pipes.append({"family": fam, **{k: float(s[k]) for k in ("arc_pd", "arc_pd_bal", "arc_pd_const",
                      "arc_pd_bal_const", "arc_const_m", "arc_frames")}})
    P.save_json("f10_metrica", {
        "take": TAKE, "n_frames": len(a), "edges": edges, "counts": cnt, "frac": cnt / len(a),
        "const": {"m": c, "lo": c - 0.25, "hi": c + 0.25, "frac": float(hit.max())},
        "median_m": float(np.median(a)), "pipelines": pipes})


if __name__ == "__main__":
    run()
