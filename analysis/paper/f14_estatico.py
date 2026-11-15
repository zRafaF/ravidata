"""F14: objects found and precision at 0.5 m per pipeline (default settings) on the three static takes. JSON only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import common as C, score as S
from . import common as P

TAKES = ("pilot_chairs", "static_pass1", "static_pass2")
ORDER = ("thr_nearest", "thr", "ca_cfar", "os_cfar", "cfar_dbscan", "rd_cfar", "grid", "grid_legacy")


def run():
    F = pd.read_parquet(C.RESULTS / "analysis-v2" / "tables" / "eval_frames.parquet")
    F = F[F["take"].isin(TAKES)]
    out = []
    for fam in ORDER:
        if fam not in set(F.family):
            continue
        row = {"family": fam, "takes": {}}
        for tk in TAKES:
            f = F[(F.family == fam) & (F["take"] == tk)]
            s = S.summary(f)
            row["takes"][tk] = {"frames": s.get("frames", 0), "obj_pd": s.get("obj_pd", np.nan),
                                "precision_50": s.get("precision_50", np.nan)}
        out.append(row)
    P.save_json("f14_estatico", {"takes": TAKES, "pipelines": out})


if __name__ == "__main__":
    run()
