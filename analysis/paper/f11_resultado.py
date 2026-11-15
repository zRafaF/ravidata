"""F11: balanced found rate vs early-alarm rate, indoor_course, three states per pipeline family. JSON only."""
from __future__ import annotations

import pandas as pd

from .. import common as C, score as S
from . import common as P

TAKE = "indoor_course"
FAMILIES = ("thr", "ca_cfar", "rd_cfar", "grid")
STATES = (("v1", "eval_frames", "antes"), ("v2", "eval_frames", "corrigido"),
          ("v2", "eval_frames_tuned_pooled_arc", "ajustado"))
KEYS = ("arc_early_rate", "arc_pd_bal", "arc_pd_bal_const", "arc_pd", "arc_frames")


def run():
    pts = []
    for run_, table, state in STATES:
        F = pd.read_parquet(C.RESULTS / f"analysis-{run_}" / "tables" / f"{table}.parquet")
        F = F[F["take"] == TAKE]
        for fam in FAMILIES:
            f = F[F.family == fam]
            if "variant" in f and state != "ajustado":
                f = f[f.variant == "default"]
            s = S.summary(f)
            pts.append({"family": fam, "state": state, **{k: float(s[k]) for k in KEYS}})
    const = {p["family"]: p["arc_pd_bal_const"] for p in pts if p["state"] == "corrigido"}
    P.save_json("f11_resultado", {"take": TAKE, "points": pts, "const": const,
                                  "const_main": const["thr"]})


if __name__ == "__main__":
    run()
