"""F3 / F4: multi-objective tuning of every pipeline family on the ground truth, and how well it generalises.

Objectives (both minimised, Optuna NSGA-II): miss rate and false positives per frame at the 0.5 m ring
(a ROC-like Pareto front). Every trial also records GOSPA, the range-only and corridor metrics and the cost.
Protocols:
  strict  one study per fold; the sampler only ever sees the training takes (no leakage):
            loto  train on all takes but one, test on the held-out take       (5 folds)
            loeo  train on one environment, test on the other                 (2 folds)
            oracle  train = test = one take: the per-scene best case (upper bound, not a usable setting)
  pooled  one study per family over all takes; folds are formed afterwards from per-take statistics (cheap;
          the sampler saw every take, so held-out numbers are slightly optimistic; reported as such).
The operating point of a fold = the training-Pareto trial with the lowest GOSPA. 'default' = DEFAULTS untouched.
Tables: opt_trials, opt_results; opt_best_params.json (all-takes selection, used by `evaluate --variant tuned`)."""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import common as C
from . import pipeline as PL
from . import score as SC

SUFF = ["frames", "n_truth", "tp10", "tp25", "tp50", "tp100", "frames_c", "n_truth_c", "n_det_c", "tpc10", "tpc25",
        "tpc50", "tpc100", "gospa_sum", "n_truth_r", "n_det_r",
        "rtp50", "near_frames", "near_hit", "near_early", "cor_frames", "cor_hit", "cor_fa", "cor_free",
        "arc_frames", "arc_cov", "arc_hit", "arc_early", "obj_frames", "obj_n", "obj_hit", "obj_other"]


def suff(Fr) -> dict:
    F = Fr[Fr.covered.astype(bool)]
    Fc = F[F.complete.astype(bool)]
    nt, nd = F.near_truth.to_numpy(), F.near_det.to_numpy()
    ct, cd = F.cor_truth.to_numpy(), F.cor_det.to_numpy()
    b, c = np.isfinite(nt) & np.isfinite(nd), np.isfinite(ct) & np.isfinite(cd)
    cov = F.arc_covered.to_numpy(bool) if "arc_covered" in F else np.zeros(len(F), bool)
    at = np.where(cov, F.arc_truth.to_numpy(), np.nan) if "arc_truth" in F else np.full(len(F), np.nan)
    ad = F.arc_det.to_numpy() if "arc_det" in F else nd
    a_ok = np.isfinite(at)
    a_b = a_ok & np.isfinite(ad)
    arc = dict(arc_frames=int(a_ok.sum()), arc_cov=int(cov.sum()),
               arc_hit=int((a_b & (np.abs(ad - at) <= 0.25)).sum()),
               arc_early=int((cov & np.isfinite(ad) & (~a_ok | (ad < at - 0.25))).sum()))
    Fo = F[F.obj_n > 0] if "obj_n" in F else F.iloc[:0]
    arc.update(obj_frames=len(Fo), obj_n=float(Fo.obj_n.sum()) if len(Fo) else 0.0,
               obj_hit=float(Fo.obj_hit.sum()) if len(Fo) else 0.0, obj_other=float(Fo.obj_other.sum()) if len(Fo) else 0.0)
    return arc | dict(frames=len(F), n_truth=F.n_truth.sum(), tp10=F.tp10.sum(), tp25=F.tp25.sum(),
                tp50=F.tp50.sum(), tp100=F.tp100.sum(), frames_c=len(Fc), n_truth_c=Fc.n_truth.sum(),
                n_det_c=Fc.n_det.sum(), tpc10=Fc.tp10.sum(), tpc25=Fc.tp25.sum(), tpc50=Fc.tp50.sum(),
                tpc100=Fc.tp100.sum(), gospa_sum=Fc.gospa.sum(), n_truth_r=F.n_truth_r.sum(),
                n_det_r=F.n_det_r.sum(), rtp50=F.rtp50.sum(), near_frames=int(np.isfinite(nt).sum()),
                near_hit=int((b & (np.abs(nd - nt) <= 0.25)).sum()), near_early=int((b & (nd < nt - 0.25)).sum()),
                cor_frames=int(np.isfinite(ct).sum()), cor_hit=int((c & (np.abs(cd - ct) <= 0.25)).sum()),
                cor_fa=int((np.isfinite(cd) & ~np.isfinite(ct)).sum()), cor_free=int((~np.isfinite(ct)).sum()))


def metrics(s: dict) -> dict:
    s = {k: float(s.get(k, 0)) for k in SUFF}
    fr = max(s["frames"], 1)
    fc = max(s["frames_c"], 1)
    out = dict(frames=s["frames"], frames_complete=s["frames_c"])
    for c in ("10", "25", "50", "100"):
        out[f"recall_{c}"] = s[f"tp{c}"] / s["n_truth"] if s["n_truth"] else np.nan
        out[f"fp_per_frame_{c}"] = (s["n_det_c"] - s[f"tpc{c}"]) / fc
        den = s["n_truth_c"] + s["n_det_c"]
        out[f"f1_{c}"] = 2 * s[f"tpc{c}"] / den if den else np.nan
    out["miss_50"] = 1 - out["recall_50"] if np.isfinite(out["recall_50"]) else 1.0
    out["gospa"] = s["gospa_sum"] / fc
    out["r_f1_50"] = 2 * s["rtp50"] / (s["n_truth_r"] + s["n_det_r"]) if s["n_truth_r"] + s["n_det_r"] else np.nan
    out["near_pd"] = s["near_hit"] / max(s["near_frames"], 1)
    out["near_early_rate"] = s["near_early"] / fr
    out["cor_pd"] = s["cor_hit"] / max(s["cor_frames"], 1)
    out["cor_false_alarm_rate"] = s["cor_fa"] / max(s["cor_free"], 1)
    out["arc_pd"] = s["arc_hit"] / s["arc_frames"] if s["arc_frames"] else np.nan
    out["arc_miss"] = 1 - out["arc_pd"]
    out["arc_early_rate"] = s["arc_early"] / s["arc_cov"] if s["arc_cov"] else np.nan
    # found AND not early (a pd - early difference would prefer 'report nothing' over any noisy detector)
    out["arc_score"] = out["arc_pd"] * (1 - out["arc_early_rate"])
    out["obj_pd"] = s["obj_hit"] / s["obj_n"] if s["obj_n"] else np.nan
    out["obj_other_per_frame"] = s["obj_other"] / s["obj_frames"] if s["obj_frames"] else np.nan
    return out


def add(a: dict, b: dict) -> dict:
    return {k: a.get(k, 0) + b.get(k, 0) for k in SUFF}


def eval_params(family, params, takes):
    """Per-take sufficient statistics and the mean cost (ms/frame) of one setting."""
    from .evaluate import eval_rec
    per, ms = {}, []
    for take in takes:
        acc = {}
        for rec in C.recordings(take):
            if family in PL.TWO_D_ONLY and not rec.is2d:
                continue
            r = eval_rec(rec, family, params, keep_dets=False)
            if r is None:
                continue
            acc = add(acc, suff(r[0]))
            ms.append(r[2])
        if acc:
            per[take] = acc
    return per, float(np.nanmean(ms)) if ms else np.nan


def _pool(per, takes):
    s = {}
    for t in takes:
        if t in per:
            s = add(s, per[t])
    return s


def storage_url(name: str) -> str:
    """SQLite file per study under derived/.../optuna (RAVIDATA_OPTUNA_DIR overrides, e.g. for network drives)."""
    import os
    from pathlib import Path
    d = Path(os.environ["RAVIDATA_OPTUNA_DIR"]) if os.environ.get("RAVIDATA_OPTUNA_DIR") else C.out_dir("optuna")
    d.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(d / f'{name}.db').as_posix()}"


OBJECTIVES = {
    # name: (the two minimised objectives, {selection name: (metric, sign: -1 = higher is better)})
    "plane": (("miss_50", "fp_per_frame_50"), {"tuned_f1": ("f1_50", -1), "tuned_gospa": ("gospa", 1)}),
    "arc": (("arc_miss", "arc_early_rate"), {"tuned_arc": ("arc_score", -1)}),
}


def study_job(args):
    import optuna
    family, name, train, n_trials, seed, objective = args[:6]
    q = args[6] if len(args) > 6 else None              # progress queue (one item per finished trial)
    obj_keys = OBJECTIVES[objective][0]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name=name, storage=storage_url(name), load_if_exists=True,
                             directions=["minimize", "minimize"],
                             sampler=optuna.samplers.NSGAIISampler(seed=seed, population_size=24))
    two_d = any(r.is2d for t in train for r in C.recordings(t))
    carried = all(t in C.CARRIED for t in train)
    done = len([t for t in st.trials if t.state.name == "COMPLETE"])
    if q is not None and done:
        q.put(min(done, n_trials))                     # resumed study: these trials count as done
    if done == 0:                                              # the default setting is always trial 0
        st.enqueue_trial({k: v for k, v in PL.DEFAULTS[family].items()
                          if not isinstance(v, bool) or True}, skip_if_exists=True)

    def objective(trial):
        p = PL.space(trial, family, two_d, carried)
        per, ms = eval_params(family, p, train)
        m = metrics(_pool(per, train))
        trial.set_user_attr("params", json.dumps(p, default=str))
        trial.set_user_attr("per_take", json.dumps(per, default=float))
        trial.set_user_attr("ms_per_frame", ms)
        trial.set_user_attr("gospa", m["gospa"])
        return tuple(float(np.nan_to_num(m[k], nan=1.0)) for k in obj_keys)

    t0 = time.time()
    cbs = [lambda study, trial: q.put(1)] if q is not None else []
    st.optimize(objective, n_trials=max(0, n_trials - done), catch=(Exception,), callbacks=cbs)
    return name, time.time() - t0


def folds(takes, protocol):
    envs = {}
    for t in takes:
        envs.setdefault(C.ENV[t], []).append(t)
    out = []
    if len(takes) > 1:
        out += [("loto", h, [t for t in takes if t != h], [h]) for h in takes]
    if len(envs) > 1:
        for e, ts in envs.items():
            out.append(("loeo", e, ts, [t for t in takes if t not in ts]))
    out += [("oracle", h, [h], [h]) for h in takes]
    return out


def pareto(vals):
    v = np.asarray(vals)
    keep = np.ones(len(v), bool)
    for i in range(len(v)):
        if keep[i]:
            dom = np.all(v <= v[i], 1) & np.any(v < v[i], 1)
            if dom.any():
                keep[i] = False
    return keep


def _trials(name):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    try:
        st = optuna.load_study(study_name=name, storage=storage_url(name))
    except KeyError:
        return []
    return [t for t in st.trials if t.state.name == "COMPLETE"]


def select(trials, train, how="tuned_f1", objective="plane"):
    """Index of the training-Pareto trial with the best selection metric (F1 at 0.5 m, GOSPA, arc score) on
    `train`."""
    obj_keys, sels = OBJECTIVES[objective]
    key, sgn = sels[how]
    ms = [metrics(_pool(json.loads(t.user_attrs["per_take"]), train)) for t in trials]
    v = np.array([[m[k] for k in obj_keys] for m in ms])
    v = np.nan_to_num(v, nan=1e9)
    pf = np.flatnonzero(pareto(v))
    g = np.array([sgn * ms[i][key] for i in pf], float)
    if not len(pf) or not np.isfinite(g).any():
        return 0
    return int(pf[np.nanargmin(g)])


def run(takes=None, families=None, n_trials=150, protocol="strict", workers=4, seed=0, objective="plane"):
    obj_keys, SELECT = OBJECTIVES[objective]
    tag = f"{protocol}_{objective}"
    import pandas as pd
    truth = C.load_table("truth_frames")
    takes = [t for t in C.available_takes(takes) if t in set(truth["take"])]
    if objective == "arc":                       # the arc truth (whole beam) exists for the carried takes only
        takes = [t for t in takes if t in C.CARRIED]
    families = families or [f for f in PL.FAMILIES if not f.startswith("app_")]
    jobs = []
    plan = folds(takes, protocol)
    for fam in families:
        if fam.endswith("_oracle"):
            fam_takes = [t for t in takes if t in C.CARRIED]
        elif fam in PL.TWO_D_ONLY:
            fam_takes = takes
        else:
            fam_takes = takes
        if protocol == "pooled":
            jobs.append((fam, f"{fam}__{objective}__pooled", fam_takes, n_trials, seed, objective))
        else:
            for kind, fold, train, test in folds(fam_takes, protocol):
                jobs.append((fam, f"{fam}__{objective}__{kind}__{fold}", train, n_trials, seed, objective))
    C.log(f"optimise: {len(jobs)} studies x {n_trials} trials ({protocol}, objective {objective}: "
          f"{' & '.join(obj_keys)}), {workers} workers")
    import multiprocessing as mp
    import queue as _queue
    import threading
    mgr = mp.Manager()
    q = mgr.Queue()
    jobs = [j + (q,) for j in jobs]
    pb = C.bar(total=len(jobs) * n_trials, desc=f"opt {tag}", unit="trial")
    stop = threading.Event()

    def pump():
        while not stop.is_set() or not q.empty():
            try:
                pb.update(q.get(timeout=0.5))
            except _queue.Empty:
                pass
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        if workers > 1:
            from concurrent.futures import as_completed
            with ProcessPoolExecutor(workers) as ex:
                for fu in as_completed([ex.submit(study_job, j) for j in jobs]):
                    name, s = fu.result()
                    C.log(f"  {name} done ({s / 60:.1f} min)")
        else:
            for j in jobs:
                name, s = study_job(j)
                C.log(f"  {name} done ({s / 60:.1f} min)")
    finally:
        stop.set()
        th.join()
        pb.close()
        mgr.shutdown()
    # ---- tables
    trows, rrows, best = [], [], {}
    for fam in families:
        fam_takes = [t for t in takes if t in C.CARRIED] if fam.endswith("_oracle") else takes
        for kind, fold, train, test in folds(fam_takes, protocol):
            name = f"{fam}__{objective}__pooled" if protocol == "pooled" else f"{fam}__{objective}__{kind}__{fold}"
            trs = _trials(name)
            if not trs:
                continue
            if protocol == "strict":
                for t in trs:
                    trows.append(dict(study=name, family=fam, protocol=protocol, objective=objective, split=kind,
                                      fold=fold, trial=t.number, **{obj_keys[0]: t.values[0], obj_keys[1]: t.values[1]},
                                      gospa=t.user_attrs.get("gospa"), ms_per_frame=t.user_attrs.get("ms_per_frame"),
                                      params=t.user_attrs.get("params")))
            per_def = json.loads(trs[0].user_attrs["per_take"]) if trs[0].number == 0 else None
            if per_def is None or not all(t in per_def for t in test):
                per_def, _ = eval_params(fam, {}, test)
            sels = [("default", 0, {}, per_def)]
            for how in SELECT:
                i = select(trs, train, how, objective)
                p = json.loads(trs[i].user_attrs["params"])
                per_tr = json.loads(trs[i].user_attrs["per_take"])
                if all(t in per_tr for t in test):
                    per_te = per_tr                               # pooled / oracle: test stats already there
                else:
                    per_te, _ = eval_params(fam, p, test)
                sels.append((how, trs[i].number, p, per_te))
            for sel, num, p, per in sels:
                m = metrics(_pool(per, test))
                rrows.append(dict(family=fam, protocol=protocol, objective=objective, split=kind, fold=fold,
                                  selection=sel,
                                  train=",".join(train), test=",".join(test), trial=num,
                                  params=json.dumps(p, default=str), **m))
        if protocol == "pooled":
            trs = _trials(f"{fam}__{objective}__pooled")
            for t in trs:
                per = json.loads(t.user_attrs["per_take"])
                m = metrics(_pool(per, fam_takes))
                trows.append(dict(study=f"{fam}__{objective}__pooled", family=fam, protocol=protocol,
                                  objective=objective, split="all", fold="all", trial=t.number,
                                  **{k: m[k] for k in ("miss_50", "fp_per_frame_50", "arc_miss", "arc_early_rate",
                                                       "gospa", "f1_50", "arc_pd")},
                                  ms_per_frame=t.user_attrs.get("ms_per_frame"), params=t.user_attrs.get("params")))
            if trs:
                best[fam] = json.loads(trs[select(trs, fam_takes, next(iter(SELECT)), objective)]
                                       .user_attrs["params"])
    if protocol == "strict":                 # all-takes setting = the one a fresh all-takes study would pick:
        for fam in families:                 # approximated by the loto fold settings' median-performing one
            first = next(iter(SELECT))
            key = SELECT[first][0]
            sub = [r for r in rrows if r["family"] == fam and r["split"] == "loto" and r["selection"] == first]
            if sub:
                best[fam] = json.loads(sorted(sub, key=lambda r: np.nan_to_num(r[key]))[len(sub) // 2]["params"])
    T = pd.DataFrame(trows)
    Rz = pd.DataFrame(rrows)
    C.save_table(T, f"opt_trials_{tag}")
    C.save_table(Rz, f"opt_results_{tag}")
    C.save_json(best, f"opt_best_params_{tag}")
    if len(Rz):
        show_key = "arc_pd" if objective == "arc" else "f1_50"
        show = Rz.pivot_table(index=["family", "split"], columns="selection", values=show_key, aggfunc="mean")
        C.log(f"{show_key} on the test folds (mean over folds):\n" + show.round(3).to_string())
    return T, Rz
