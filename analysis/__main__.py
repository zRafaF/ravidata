"""python -m analysis <step> [options]   (make an-<step> / make analysis / make analysis-quick)

Steps, in the order `all` runs them (each writes tables under data/ravidata/derived/analysis-v1/tables;
--quick writes to derived/analysis-quick and uses a few seconds of every recording):

  tags        GT2  AprilTag poses in the videos                          -> tags
  extrinsic   GT3  radar-camera offset, range bias, angle calibration     -> calib_targets, extrinsic.json, angle_calib.json
  truth       GT4-6 obstacle truth per radar frame (maps / tags)        -> truth_sectors, truth_frames, map_scale
  sensor      A1-A6 noise, targets, timing, sync check                  -> noise, targets, timing, sync_check
  eval        B-D  every pipeline family, default settings              -> eval_frames, eval_dets, eval_cost, eval_summary
  opt         F3-F4 multi-objective tuning + generalisation folds       -> opt_trials_*, opt_results_*, opt_best_params_*.json
  eval-tuned       every family with the tuned settings                 -> eval_*_tuned
  cordic      E1   fixed-point CORDIC vs float64 (not part of `all`)    -> cordic_error, cordic_phase, cordic_grid
  plots            figures from the tables only                         -> figures/*.png
"""

from __future__ import annotations

import argparse
import os
import time

from . import common as C

STEPS = ["tags", "extrinsic", "truth", "sensor", "eval", "opt", "eval-tuned", "cordic", "plots"]
ALL = [s for s in STEPS if s != "cordic"]            # CORDIC is done later (on the C code); `cordic` still runs alone


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--quick", action="store_true", help="a few seconds per recording, output to analysis-quick")
    ap.add_argument("--takes", nargs="*", help="default: all takes")
    ap.add_argument("--families", nargs="*", help="pipeline families (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag-step", type=int, default=2, help="tags: every N-th video frame (+ every map keyframe)")
    ap.add_argument("--model", default="pi3x", help="truth: map model for the carried takes")
    ap.add_argument("--trials", type=int, default=150, help="opt: trials per study")
    ap.add_argument("--protocol", default="pooled", choices=["strict", "pooled", "both"])
    ap.add_argument("--pdf", action="store_true", help="plots: also write PDF")
    ap.add_argument("--only", nargs="*", help="plots: only figures whose function name contains one of these")
    a = ap.parse_args(argv)
    C.set_quick(a.quick)
    if a.quick and not os.environ.get("RAVIDATA_OPTUNA_DIR"):     # quick studies always start fresh
        os.environ["RAVIDATA_OPTUNA_DIR"] = str(C.out_dir("optuna") / time.strftime("%Y%m%d-%H%M%S"))
    steps = ALL if a.step == "all" else [a.step]
    t_all = time.time()
    prots = ["strict", "pooled"] if a.protocol == "both" else [a.protocol]
    for i, st in enumerate(steps, 1):
        t0 = time.time()
        C.log(f"=== [{i}/{len(steps)}] {st}{' (quick)' if a.quick else ''}"
              + (f"  (total so far {(t0 - t_all) / 60:.1f} min)" if i > 1 else ""))
        if st == "tags":
            from . import tags
            tags.run(a.takes, step=a.tag_step if not a.quick else 3, workers=a.workers)
        elif st == "extrinsic":
            from . import extrinsic
            extrinsic.run(a.takes)
        elif st == "truth":
            from . import truth
            truth.run(a.takes, model=a.model)
        elif st == "sensor":
            from . import characterise
            characterise.run(a.takes)
        elif st == "eval":
            from . import evaluate
            evaluate.run(a.takes, a.families)
        elif st == "opt":
            from . import optimise
            for pr in prots:
                n = a.trials if not a.quick else 2
                fam = a.families
                if a.quick and pr == "strict" and not fam:
                    fam = ["thr", "grid"]                  # quick: strict folds for two families only
                optimise.run(a.takes, fam, n_trials=n, protocol=pr, workers=a.workers)
        elif st == "eval-tuned":
            from . import evaluate
            for pr in prots:
                best = C.load_json(f"opt_best_params_{pr}", None)
                if not best:
                    C.log(f"  no opt_best_params_{pr}.json: run the opt step first")
                    continue
                evaluate.run(a.takes, [f for f in (a.families or list(best)) if f in best], variant=f"tuned_{pr}",
                             params_by_family=best)
        elif st == "cordic":
            from . import cordic
            cordic.run(a.takes, max_recs=4 if a.quick else None)
        elif st == "plots":
            from . import plots
            plots.run(a.only, a.pdf)
        dt = time.time() - t0
        C.manifest(st, vars(a), dt)
        C.log(f"=== {st} done in {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
