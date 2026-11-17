"""python -m analysis <step> [options]   (make an-<step> / make analysis / make analysis-quick)

Steps, in the order `all` runs them (each writes tables under results/analysis-v4/tables;
--quick writes to results/analysis-v4-quick and uses a few seconds of every recording):

  tags        GT2  AprilTag poses in the videos                          -> tags
  extrinsic   GT3  radar-camera offset, range bias, angle calibration     -> calib_targets, extrinsic.json, angle_calib.json
  truth       GT4-6 obstacle truth per radar frame (maps / tags)        -> truth_sectors, truth_frames, map_scale
  sensor      A1-A6 noise, targets, timing, sync check                  -> noise, targets, timing, sync_check
  eval        B-D  every pipeline family, default settings              -> eval_frames, eval_dets, eval_cost, eval_summary
  ablation         one stage at a time, each over a threshold sweep     -> ablation_points, ablation_summary
  opt         F3-F4 multi-objective tuning + generalisation folds       -> opt_trials_*, opt_results_*, opt_best_params_*.json
  eval-tuned       every family with the tuned settings                 -> eval_*_tuned
  floor            where the early alarms come from (floor in beam, phantom) -> floor_margin, floor_pitch, phantom
  plots            figures from the tables only                         -> figures/*.png
  story            explanatory figures (truth, range-time maps)         -> figures/story/*.png
"""

from __future__ import annotations

import argparse
import os
import time

from . import common as C

STEPS = ["tags", "extrinsic", "truth", "sensor", "eval", "ablation", "opt", "eval-tuned", "floor", "plots",
         "story", "report"]
ALL = list(STEPS)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m analysis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=STEPS + ["all"])
    ap.add_argument("--from", dest="from_step", choices=STEPS, help="all: start at this step (earlier ones done)")
    ap.add_argument("--quick", action="store_true", help="a few seconds per recording, output to analysis-v4-quick")
    ap.add_argument("--takes", nargs="*", help="default: all takes")
    ap.add_argument("--families", nargs="*", help="pipeline families (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag-step", type=int, default=2, help="tags: every N-th video frame (+ every map keyframe)")
    ap.add_argument("--model", default="pi3x", help="truth: map model for the carried takes")
    ap.add_argument("--trials", type=int, default=150, help="opt: trials per study")
    ap.add_argument("--protocol", default="pooled", choices=["strict", "pooled", "both"])
    ap.add_argument("--objective", default="both", choices=["plane", "arc", "both"],
                    help="opt: plane = miss rate & false positives per frame at 0.5 m (per direction); arc = the "
                         "nearest obstacle's distance only (found within 0.25 m & early alarms); both")
    ap.add_argument("--pdf", action="store_true", help="plots: also write PDF")
    ap.add_argument("--only", nargs="*", help="plots: only figures whose function name contains one of these")
    ap.add_argument("--all-figs", action="store_true", help="plots: also the diagnostic figures")
    a = ap.parse_args(argv)
    C.set_quick(a.quick)
    if a.quick and not os.environ.get("RAVIDATA_OPTUNA_DIR"):     # quick studies always start fresh
        os.environ["RAVIDATA_OPTUNA_DIR"] = str(C.out_dir("optuna") / time.strftime("%Y%m%d-%H%M%S"))
    steps = ALL if a.step == "all" else [a.step]
    if a.step == "all" and a.from_step:
        steps = steps[steps.index(a.from_step):] if a.from_step in steps else steps
    t_all = time.time()
    prots = ["strict", "pooled"] if a.protocol == "both" else [a.protocol]
    objs = ["plane", "arc"] if a.objective == "both" else [a.objective]
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
            from . import pipeline as PL
            evaluate.run(a.takes, a.families)
            fams = [f for f in (a.families or PL.FAMILIES) if not f.startswith("app_")]
            C.log("  the same with the IMU height gate (floor returns dropped):")
            evaluate.run(a.takes, fams, variant="hgate", params_by_family={f: {"hgate": True} for f in fams})
        elif st == "ablation":
            from . import ablation
            ablation.run(a.takes, workers=a.workers)
        elif st == "report":
            from . import report
            report.run()
        elif st == "story":
            from . import story
            story.run(pdf=a.pdf)
        elif st == "opt":
            from . import optimise
            for pr in prots:
                for ob in objs:
                    n = a.trials if not a.quick else 2
                    fam = a.families
                    if a.quick and pr == "strict" and not fam:
                        fam = ["thr", "grid"]              # quick: strict folds for two families only
                    optimise.run(a.takes, fam, n_trials=n, protocol=pr, workers=a.workers, objective=ob)
        elif st == "eval-tuned":
            from . import evaluate
            for pr in prots:
                for ob in objs:
                    best = C.load_json(f"opt_best_params_{pr}_{ob}", None)
                    if not best:
                        C.log(f"  no opt_best_params_{pr}_{ob}.json: run the opt step first")
                        continue
                    evaluate.run(a.takes, [f for f in (a.families or list(best)) if f in best],
                                 variant=f"tuned_{pr}_{ob}", params_by_family=best)
        elif st == "floor":
            from . import floor
            floor.run(n_max=40 if a.quick else 200)
        elif st == "plots":
            from . import plots
            plots.run(a.only, a.pdf, all_figs=a.all_figs)
        dt = time.time() - t0
        C.manifest(st, vars(a), dt)
        C.log(f"=== {st} done in {dt / 60:.1f} min")


if __name__ == "__main__":
    main()
