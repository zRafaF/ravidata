"""B / C / D: every pipeline family (default settings, or settings from the optimiser) over every recording,
scored against the truth. Tables: eval_frames (per frame), eval_dets (per scored detection), eval_summary
(per family x variant x take x config, with block-bootstrap 95 % intervals)."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from . import common as C
from . import geometry as G
from . import pipeline as PL
from . import score as SC


@lru_cache(maxsize=1)
def _truth_tables():
    S = C.load_table("truth_sectors")
    F = C.load_table("truth_frames")
    return {k: g for k, g in S.groupby("rec")}, {k: g for k, g in F.groupby("rec")}


def truth_arrays(key: str, n: int):
    Sg, Fg = _truth_tables()
    if key not in Fg:
        return None
    s = Sg[key].sort_values(["f", "sector"])
    fr = Fg[key].sort_values("f")
    m = min(n, fr.f.max() + 1)
    def piv(col):
        a = np.full((n, G.N_SECT), np.nan)
        v = s[col].to_numpy().reshape(-1, G.N_SECT)[:m]
        a[:m] = v
        return a
    cov = np.zeros(n, bool); cov[:m] = fr.covered.to_numpy()[:m]
    comp = np.zeros(n, bool); comp[:m] = fr.complete.to_numpy()[:m]
    cor = np.full(n, np.nan); cor[:m] = fr.corridor_r.to_numpy()[:m]
    return piv("obs_r"), piv("obs_az"), piv("floor_r"), comp, cor, cov


def eval_rec(rec, family, params, variant="default", F=None, keep_dets=True, trig=None):
    n_max = C.default_max_frames()
    out = PL.detect(rec, family, params, n_max, F=F, trig=trig)
    d, ms, Fe = out
    if d is None:
        return None
    n = Fe.n if Fe is not None else len(C.radar_times(rec, n_max))
    ta = truth_arrays(rec.key, n)
    if ta is None:
        return None
    fr, dt = SC.score_rec(d, *ta)
    import pandas as pd
    meta = dict(rec=rec.key, take=rec.take, config=rec.config, part=rec.part, env=C.ENV[rec.take],
                family=family, variant=variant)
    Fr = pd.DataFrame(fr).assign(f=np.arange(n), **meta)
    Dt = pd.DataFrame(dt).assign(**meta) if keep_dets else None
    return Fr, Dt, ms


def block_bootstrap(Fr, Dt, fps, metric_keys, n_boot=200, block_s=5.0, seed=0):
    """95 % intervals of summary metrics, resampling blocks of block_s seconds (frames are correlated)."""
    rng = np.random.default_rng(seed)
    blk = (Fr.f.to_numpy() / max(fps, 1) // block_s).astype(int) + Fr.rec.factorize()[0] * 100000
    ub = np.unique(blk)
    if len(ub) < 4:
        return {}
    vals = {k: [] for k in metric_keys}
    gF = {b: g for b, g in Fr.groupby(blk)}
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub))
        Fb = __import__("pandas").concat([gF[b] for b in pick], ignore_index=True)
        s = SC.summary(Fb, None)
        for k in metric_keys:
            vals[k].append(s.get(k, np.nan))
    return {f"{k}_lo": float(np.nanpercentile(v, 2.5)) for k, v in vals.items()} | \
           {f"{k}_hi": float(np.nanpercentile(v, 97.5)) for k, v in vals.items()}


def summarise(Fr_all, Dt_all, by=("family", "variant", "take", "config"), boot=True):
    import pandas as pd
    rows = []
    keys = ["f1_50", "recall_50", "precision_50", "gospa", "cor_pd"]
    groups = list(Fr_all.groupby(list(by)))
    for k, g in C.bar(groups, desc=f"summary by {'/'.join(by)}", unit="group", leave=False):
        dsel = Dt_all
        for col, v in zip(by, k):
            dsel = dsel[dsel[col] == v]
        fps = 33.3 if "1d" in str(dict(zip(by, k)).get("config", "")) else 20.0
        s = SC.summary(g, dsel, fps)
        if boot:
            s.update(block_bootstrap(g, dsel, fps, keys, n_boot=50 if C.QUICK else 200))
        rows.append(dict(zip(by, k)) | s)
    return pd.DataFrame(rows)


def run(takes=None, families=None, variant="default", params_by_family=None, configs=None, keep_dets=True):
    import pandas as pd
    families = families or PL.FAMILIES
    frames, dets, costs = [], [], []
    jobs = [(take, rec, fam) for take in C.available_takes(takes) for rec in C.recordings(take, configs)
            for fam in families if not (fam in PL.TWO_D_ONLY and not rec.is2d)
            and not (fam == "grid_oracle" and take not in C.CARRIED)]
    pb = C.bar(total=len(jobs), desc=f"eval {variant}", unit="run")
    for take, rec, fam in jobs:
        pb.update()
        pb.set_postfix_str(f"{rec.key.split('/')[-1]} {fam}")
        p = (params_by_family or {}).get(fam, {})
        t0 = time.time()
        try:
            r = eval_rec(rec, fam, p, variant, keep_dets=keep_dets)
        except Exception as e:                     # a failing family is reported, not hidden
            C.log(f"  {rec.key} {fam}: FAILED {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        Fr, Dt, ms = r
        frames.append(Fr)
        if Dt is not None:
            dets.append(Dt)
        costs.append(dict(rec=rec.key, take=take, config=rec.config, family=fam, variant=variant,
                          ms_per_frame=ms, frames=len(Fr), wall_s=time.time() - t0))
        s = SC.summary(Fr)
        C.log(f"  {rec.key:44s} {fam:13s} f1@0.5 {s.get('f1_50', np.nan):.2f} "
              f"gospa {s.get('gospa', np.nan):.2f} cor_pd {s.get('cor_pd', np.nan):.2f} ({ms:.2f} ms/frame)")
    pb.close()
    if not frames:
        C.log("nothing evaluated (truth missing? run the truth step first)")
        return None
    Fa = pd.concat(frames, ignore_index=True)
    Da = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame()
    suffix = "" if variant == "default" else f"_{variant}"
    C.save_table(Fa, f"eval_frames{suffix}", takes=C.available_takes(takes))
    if keep_dets:
        C.save_table(Da, f"eval_dets{suffix}", takes=C.available_takes(takes))
    C.save_table(pd.DataFrame(costs), f"eval_cost{suffix}", takes=C.available_takes(takes))
    Fa = C.load_table(f"eval_frames{suffix}")
    Da = C.load_table(f"eval_dets{suffix}", required=False)
    S1 = summarise(Fa, Da)
    S2 = summarise(Fa, Da, by=("family", "variant", "env"), boot=False)
    S3 = summarise(Fa, Da, by=("family", "variant"), boot=True)
    S = pd.concat([S1, S2, S3], ignore_index=True)
    C.save_table(S, f"eval_summary{suffix}")
    show = S3[["family", "variant", "f1_50", "recall_50", "precision_50", "ap_50", "gospa", "cor_pd", "cor_err_m"]]
    C.log("\n" + show.round(3).to_string(index=False))
    return S
