"""Start a new results folder (RAVIDATA_RUN) from the steps of an earlier one that a change does not affect.

    python -m analysis.seed <earlier run> [table ...]      (make analysis-from-truth does this first)

Copies, without overwriting, the tables of `tags` and `extrinsic` (AprilTag poses in the videos, radar-camera
calibration: neither depends on the maps) and records the copy in runs.jsonl, so the new folder says where they
came from."""
from __future__ import annotations

import shutil
import sys
import time

from . import common as C

DEFAULT = ["tags.parquet", "extrinsic.json", "angle_calib.json", "calib_targets.parquet", "calib_moving.parquet"]


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if not a:
        raise SystemExit(__doc__)
    src = C.RESULTS / a[0] / "tables"
    dst = C.out_dir("tables")
    if src.resolve() == dst.resolve():
        raise SystemExit(f"RAVIDATA_RUN is {C.RUN}: the same folder as {a[0]}")
    names = a[1:] or DEFAULT
    t0 = time.time()
    copied, kept = [], []
    for n in names:
        if not (src / n).exists():
            raise SystemExit(f"{src / n} does not exist")
        if (dst / n).exists():
            kept.append(n)
            continue
        shutil.copy2(src / n, dst / n)
        copied.append(n)
    C.log(f"seed {C.RUN} from {a[0]}: copied {copied or 'nothing'}" + (f"; already there: {kept}" if kept else ""))
    C.manifest("seed", {"from_run": a[0], "copied": copied, "already_there": kept}, time.time() - t0)


if __name__ == "__main__":
    main()
