"""Ground truth of the carried takes: is the camera map metric? (a) Leave-one-out at the AprilTags: a window's scale
from its other tag sightings, error of the map depth at the left-out tag. (b) Metric scale per map window: from the
window's own tags (>= 3 sightings), from the fused map's pose graph x one factor fitted on the tagged windows (used
since analysis-v3), and the median of all tags (used until analysis-v2 for windows without tags). JSON only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import common as C
from ..maps import TakeMaps
from . import common as P


def run():
    tags = C.load_tags()
    out = {}
    for take in C.CARRIED:
        tm = TakeMaps(take, "pi3x")
        rows = pd.DataFrame(tm.fit_scale(tags))
        err = []
        for k, g in rows.groupby("window"):
            if len(g) < 4:
                continue
            r = g.ratio.to_numpy(); mz = g.map_z.to_numpy(); tz = g.tag_z.to_numpy()
            for i in range(len(g)):
                s = np.median(np.delete(r, i))
                err.append((float(tz[i]), float(mz[i] * s - tz[i])))
        err = np.array(err)
        ws = np.asarray(tm.fused["window_s"], float)
        n = rows.groupby("window").size()
        win = []
        for k in range(len(tm.wfiles)):
            q = rows[rows.window == k].ratio
            win.append(dict(k=k, n_tags=int(n.get(k, 0)), tags=float(q.median()) if len(q) >= 3 else None,
                            used=float(tm.scale_w[k]), src=tm.scale_src[k]))
        g_ = np.median([w["tags"] / ws[w["k"]] for w in win if w["tags"] is not None])
        for w in win:
            w["pose_graph"] = float(ws[w["k"]] * g_)
        old = float(rows.ratio.median())
        # "calibrate once": the scale of a tagged window predicted from the pose graph and a factor fitted on the
        # OTHER tagged windows only (leave one window out), checked at that window's tags
        lowo = []
        tagged_k = [w["k"] for w in win if w["tags"] is not None]
        for k in tagged_k:
            g_k = np.median([w["tags"] / ws[w["k"]] for w in win if w["tags"] is not None and w["k"] != k])
            q = rows[rows.window == k]
            lowo += [(float(tz), float(mz * ws[k] * g_k - tz)) for tz, mz in zip(q.tag_z, q.map_z)]
        lowo = np.array(lowo)
        tagged = [w for w in win if w["tags"] is not None]
        out[take] = dict(lowo=lowo, lowo_med_abs_cm=float(np.median(np.abs(lowo[:, 1])) * 100),
                         lowo_med_abs_pct=float(np.median(np.abs(lowo[:, 1]) / lowo[:, 0]) * 100), n_lowo=int(len(lowo)),
                         loo=err, loo_med_abs_cm=float(np.median(np.abs(err[:, 1])) * 100),
                         loo_med_abs_pct=float(np.median(np.abs(err[:, 1]) / err[:, 0]) * 100), n_loo=int(len(err)),
                         windows=win, old_fallback=old, factor=float(g_),
                         agree_med_pct=float(np.median([abs(w["tags"] / w["pose_graph"] - 1) for w in tagged]) * 100),
                         agree_max_pct=float(np.max([abs(w["tags"] / w["pose_graph"] - 1) for w in tagged]) * 100),
                         untagged_old_over_new=[old / w["pose_graph"] for w in win if w["tags"] is None])
        print(take, "LOWO (pose graph, factor from other windows) |err| median %.1f cm (%.1f %%) n %d" % (
            out[take]["lowo_med_abs_cm"], out[take]["lowo_med_abs_pct"], len(lowo)))
        print(take, "LOO |err| median %.1f cm (%.1f %%) n %d; pose graph vs tags: median %.1f %% max %.1f %%; untagged old/new %s" % (
            out[take]["loo_med_abs_cm"], out[take]["loo_med_abs_pct"], len(err), out[take]["agree_med_pct"],
            out[take]["agree_max_pct"], np.round(out[take]["untagged_old_over_new"], 2)))
    P.save_json("fv_dados", out)


if __name__ == "__main__":
    run()
