"""Ablation: what each pre- / post-processing stage contributes, without a black-box search.

A base pipeline (CA-CFAR, defaults) is changed one factor at a time (background removal, RX combining, Doppler
handling, near-range cut, non-coherent integration over frames, detector, M-of-N, nearest-obstacle tracker,
temporal DBSCAN, occupancy grids, IMU height gate). Every variant is run over a sweep of its detection threshold,
so variants are compared by their whole operating curve (found vs early alarms), not by one arbitrary threshold.

Tables: ablation_points (variant x threshold: all metrics), ablation_summary (per variant: best 'found' at an
early-alarm budget of 5 / 10 / 20 %, area under the found-vs-early curve up to 30 %).
Per-recording results are cached (derived/.../cache/ablation.jsonl) under a key of the settings, the truth table
and the code, so a rerun only computes what changed."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import common as C
from . import pipeline as PL

K_SWEEP = [3, 4, 5, 6, 8, 10, 12, 15, 18]
L_SWEEP = [5, 8, 11, 14, 17, 20, 25]
CODE = ["pipeline.py", "detectors.py", "radar.py", "grid.py", "grid_legacy.py", "cluster.py", "score.py",
        "evaluate.py", "poses.py"]


def variants():
    """(axis, label, family, params) - the base is ('base', 'CA-CFAR defaults', 'ca_cfar', {})."""
    v = [("base", "CA-CFAR (defaults)", "ca_cfar", {})]
    v += [("background", f"bg {lab}", "ca_cfar", p) for lab, p in
          (("EMA a=0.9", {"bg": "ema", "bg_alpha": 0.9}), ("EMA a=0.98", {"bg": "ema", "bg_alpha": 0.98}))]
    v += [("combine RX", f"combine {c}", "ca_cfar", {"combine": c}) for c in ("coh", "max", "rx1")]
    v += [("Doppler (2D)", f"doppler {d}", "ca_cfar", {"doppler": d}) for d in ("zero", "nonzero", "max")]
    v += [("near cut", f"r_min {r:.1f} m", "ca_cfar", {"r_min": r}) for r in (0.5, 0.7, 0.9)]
    v += [("integration", f"integrate a={a}", "ca_cfar", {"integ": a}) for a in (0.5, 0.8, 0.9)]
    v += [("detector", lab, fam, p) for lab, fam, p in
          (("threshold", "thr", {}), ("GO-CFAR", "ca_cfar", {"kind": "go"}), ("SO-CFAR", "ca_cfar", {"kind": "so"}),
           ("OS-CFAR", "os_cfar", {}), ("RD-CFAR (2D)", "rd_cfar", {}))]
    v += [("M-of-N", f"{m} of {n}", "ca_cfar", {"m": m, "n": n}) for m, n in ((2, 3), (3, 5))]
    v += [("tracker", lab, "ca_cfar", p) for lab, p in
          (("track (confirm 3, hold 5)", {"track": True}),
           ("track (confirm 5, hold 10)", {"track": True, "track_confirm": 5, "track_hold": 10}))]
    v += [("temporal clustering", "DBSCAN, 5 frames", "cfar_dbscan", {})]
    v += [("grid", lab, fam, p) for lab, fam, p in
          (("log-odds grid", "grid", {}), ("grid, decay 0.7", "grid", {"g_decay": 0.7}),
           ("grid, decay 0.97", "grid", {"g_decay": 0.97}), ("grid + IMU", "grid_imu", {}),
           ("grid + map poses", "grid_oracle", {}), ("legacy energy grid", "grid_legacy", {}),
           ("legacy grid + IMU", "grid_legacy_imu", {}))]
    v += [("IMU height gate", lab, "ca_cfar", dict({"hgate": True}, **p)) for lab, p in
          (("gate", {}), ("gate, el -5°", {"el_off": -5.0}), ("gate, el -10°", {"el_off": -10.0}),
           ("gate, radar 1.4 m", {"radar_h": 1.4}), ("gate, el -10°, radar 1.4 m", {"el_off": -10.0, "radar_h": 1.4}))]
    return v


def sweep(family):
    if family.startswith("grid_legacy"):
        return [("L_thr_db", k) for k in L_SWEEP]
    return [("k_db", k) for k in K_SWEEP]


def _code_hash():
    h = hashlib.sha1()
    for f in CODE:
        h.update((Path(__file__).parent / f).read_bytes())
    tf = C.out_dir("tables") / "truth_frames.parquet"
    h.update(str(tf.stat().st_mtime_ns if tf.exists() else 0).encode())
    return h.hexdigest()[:12]


def _key(rec_key, family, params, code):
    return hashlib.sha1(json.dumps([rec_key, family, params, code], sort_keys=True, default=str).encode()).hexdigest()


def _job(args):
    """Evaluate one (family, params) on a list of recordings -> per-take sufficient statistics."""
    family, params, recs, code, cached = args
    from .evaluate import eval_rec
    from .optimise import suff, add
    per, new = {}, []
    for take, key in recs:
        k = _key(key, family, params, code)
        if k in cached:
            s = cached[k]
        else:
            rec = next(r for r in C.recordings(take) if r.key == key)
            r = eval_rec(rec, family, params, keep_dets=False)
            if r is None:
                continue
            s = {kk: float(vv) for kk, vv in suff(r[0]).items()}
            new.append((k, s))
        per[take] = add(per.get(take, {}), s)
    return family, params, per, new


def run(takes=None, workers=4):
    import pandas as pd
    from .optimise import metrics, _pool
    takes = C.available_takes(takes)
    truth = C.load_table("truth_frames")
    takes = [t for t in takes if t in set(truth["take"])]
    code = _code_hash()
    cache_f = C.out_dir("cache") / "ablation.jsonl"
    cached = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            k, s = json.loads(line)
            cached[k] = s
    jobs, meta = [], []
    for axis, label, fam, p in variants():
        recs = [(t, r.key) for t in takes for r in C.recordings(t)
                if not (fam in PL.TWO_D_ONLY and not r.is2d) and not (fam.endswith("_oracle") and t not in C.CARRIED)
                and not (axis == "Doppler (2D)" and not r.is2d)]
        for name, val in sweep(fam):
            params = dict(p, **{name: val})
            jobs.append((fam, params, recs, code, {k: v for k, v in cached.items()}))
            meta.append((axis, label, fam, name, val))
    C.log(f"ablation: {len(jobs)} runs ({len(variants())} variants x threshold sweep), {len(cached)} cached results")
    results = [None] * len(jobs)
    pb = C.bar(total=len(jobs), desc="ablation", unit="run")
    with open(cache_f, "a") as fh:
        if workers > 1:
            with ProcessPoolExecutor(workers) as ex:
                futs = {ex.submit(_job, j[:4] + (_subset(cached, j),)): n for n, j in enumerate(jobs)}
                for fu in as_completed(futs):
                    n = futs[fu]
                    results[n] = fu.result()
                    for k, s in results[n][3]:
                        fh.write(json.dumps([k, s]) + "\n")
                    pb.update()
        else:
            for n, j in enumerate(jobs):
                results[n] = _job(j[:4] + (_subset(cached, j),))
                for k, s in results[n][3]:
                    fh.write(json.dumps([k, s]) + "\n")
                pb.update()
    pb.close()
    rows = []
    for (axis, label, fam, name, val), (_, params, per, _) in zip(meta, results):
        carried = [t for t in per if t in C.CARRIED]
        m_all = metrics(_pool(per, list(per)))
        m_car = metrics(_pool(per, carried)) if carried else {}
        rows.append(dict(axis=axis, variant=label, family=fam, knob=name, value=val,
                         params=json.dumps(params, default=str),
                         arc_pd=m_car.get("arc_pd"), arc_early_rate=m_car.get("arc_early_rate"),
                         near_pd=m_all["near_pd"], near_early_rate=m_all["near_early_rate"], f1_50=m_all["f1_50"],
                         recall_50=m_all["recall_50"], fp_per_frame_50=m_all["fp_per_frame_50"],
                         r_f1_50=m_all["r_f1_50"], gospa=m_all["gospa"]))
    Pt = pd.DataFrame(rows)
    C.save_table(Pt, "ablation_points")
    S = summarise(Pt)
    C.save_table(S, "ablation_summary")
    C.log("\n" + S[["axis", "variant", "pd_at_5", "pd_at_10", "pd_at_20", "auc_30"]].round(3).to_string(index=False))
    return S


def _subset(cached, job):
    fam, params, recs, code = job[:4]
    out = {}
    for _, key in recs:
        k = _key(key, fam, params, code)
        if k in cached:
            out[k] = cached[k]
    return out


def operating_curve(x, y):
    """Upper envelope of (early, found) points, made monotone: the best 'found' at each early-alarm budget."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.maximum.accumulate(np.asarray(y)[o])
    return x, y


def summarise(Pt):
    import pandas as pd
    rows = []
    for (axis, label), g in Pt.groupby(["axis", "variant"], sort=False):
        g = g.dropna(subset=["arc_pd", "arc_early_rate"])
        if not len(g):
            continue
        x, y = operating_curve(g.arc_early_rate.to_numpy(), g.arc_pd.to_numpy())
        best = lambda b: float(y[x <= b].max()) if (x <= b).any() else 0.0
        grid = np.linspace(0, 0.3, 61)
        yy = np.array([best(b) for b in grid])
        rows.append(dict(axis=axis, variant=label, family=g.family.iloc[0], pd_at_5=best(0.05), pd_at_10=best(0.10),
                         pd_at_20=best(0.20), auc_30=float(np.trapezoid(yy, grid) / 0.3),
                         near_pd_max=float(g.near_pd.max()), f1_50_max=float(g.f1_50.max())))
    return pd.DataFrame(rows)
