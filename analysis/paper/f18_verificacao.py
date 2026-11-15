"""F18: checks of the camera-built map (held-out tags, radar as altimeter, radar returns on map surfaces).
Only collects the eight work/audit/gtcheck/<take>_<model>.json files into one JSON; Typst draws everything."""
from __future__ import annotations

import json
from pathlib import Path

from . import common as P

SRC = Path(__file__).resolve().parents[2] / "work" / "audit" / "gtcheck"
TAKES = ("indoor_course", "outdoor_course")
MODELS = ("pi3x", "da3", "mapanything", "vggt")


def run():
    out = {"takes": list(TAKES), "models": list(MODELS), "data": {}}
    for t in TAKES:
        out["data"][t] = {}
        for m in MODELS:
            d = json.loads((SRC / f"{t}_{m}.json").read_text(encoding="utf-8"))
            out["data"][t][m] = {k: d[k] for k in ("A", "B", "C")}
    P.save_json("f18_verificacao", out)


if __name__ == "__main__":
    run()
