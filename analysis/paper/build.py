"""Every thesis figure's data, in one go: runs each exporter of analysis/paper (text-free rasters + JSON) into
results/<RAVIDATA_RUN>/paper and, with TCC_FIGURES=<thesis>/figures/data, copies them to the thesis repository.
The Typst side (labels, axes, layout) is compiled there by figures/build.py.   make paper-figures"""
from __future__ import annotations

import importlib
import sys
import time

from .. import common as C

# (module, positional arguments): the figures of the thesis, in reading order
FIGS = [
    ("f02_arranjo", []),
    ("f03_dados", []),
    ("fig_estatico", []),
    ("fig_validacao", []),
    ("fig_quadro", []),
    ("fig_cascata", []),
    ("fig_cascata", ["outdoor_course", 2, 20.0, 20.0, "fco"]),
    ("fig_cascata", ["indoor_course", 0, 20.0, 20.0, "fh"]),               # the overview figure: around fig_quadro's 30 s
    ("fig_resultados", []),
    ("fig_hero", []),                         # after fig_quadro (reads fq_dados)
]


def main(only=None):
    C.log(f"thesis figures from results/{C.RUN}")
    failed = []
    for mod, args in FIGS:
        if only and mod not in only:
            continue
        t0 = time.time()
        try:
            importlib.import_module(f"analysis.paper.{mod}").run(*args)
            C.log(f"  {mod} {args}: {time.time() - t0:.0f} s")
        except Exception as e:                      # noqa: BLE001  (report every figure, then fail)
            failed.append(mod)
            C.log(f"  {mod} {args}: FAILED {type(e).__name__}: {e}")
    if failed:
        raise SystemExit(f"failed: {failed}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
