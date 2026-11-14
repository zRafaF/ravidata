import json
import time

import numpy as np

from ravidata.groundtruth import (ScoreCfg, Take, leaderboard, parse_takes, report_md, score_take,
                                 take_start_text, truth_range)


def _mk(frame, text, kind="marker"):
    from ravidata.evaluate import _DIST
    m = _DIST.search(text)
    return {"frame": frame, "text": text, "kind": kind,
            "dist": float(m.group(1)) if (m and not text.startswith("take")) else None}


def test_parse_and_interpolate_walk_take():
    t = np.arange(200) * 0.04
    mk = [_mk(10, take_start_text("S3", 2, obj="board")), _mk(20, "d=3.0 m"), _mk(70, "d=2.0 m"),
          _mk(120, "d=1.0 m"), _mk(130, "F4 MISSED obstacle"), _mk(150, "take end"), _mk(170, "d=0.5 m")]
    takes = parse_takes("r", mk, 200)
    assert len(takes) == 1 and takes[0].scene == "S3" and takes[0].meta["rep"] == 2
    tk = takes[0]
    assert tk.expect == "alarm" and tk.mode == "walk" and tk.owner_missed == 1
    assert [d for _, d in tk.dist_markers] == [3.0, 2.0, 1.0]       # marker after take end ignored
    tr = truth_range(tk, t)
    assert np.isnan(tr[19]) and tr[20] == 3.0 and abs(tr[45] - 2.5) < 1e-9 and tr[120] == 1.0 and np.isnan(tr[121])


def test_score_alarm_and_no_alarm_takes():
    t = np.arange(200) * 0.04
    tk = Take("r", "S3", {"mode": "walk", "expect": "alarm"}, 0, 200, [(0, 3.0), (199, 0.5)])
    truth = truth_range(tk, t)
    y = truth.copy()
    y[:80] = np.nan                         # detected late: from truth ~2.0 m on
    r = score_take(y, tk, t, 0.04, ScoreCfg())
    assert r["detected"] and 1.9 < r["warn_m"] < 2.1 and 55 < r["hit %"] < 65 and r["clutter %"] == 0
    y2 = truth - 1.0                        # always reports something nearer (clutter)
    r2 = score_take(y2, tk, t, 0.04, ScoreCfg())
    assert not r2["detected"] and r2["warn_m"] == 0 and r2["clutter %"] > 90
    na = Take("r", "S0", {"mode": "walk", "expect": "no_alarm"}, 0, 200)
    y3 = np.full(200, np.nan)
    y3[50:60] = 1.2
    y3[100:105] = 2.9                       # beyond alarm_max: not a false alarm
    r3 = score_take(y3, na, t, 0.04, ScoreCfg())
    assert abs(r3["fa %"] - 5.0) < 1e-9 and r3["fa/min"] > 0
    rows = [dict(r, detector="a", preset="p"), dict(r2, detector="b", preset="p"), dict(r3, detector="a", preset="p"),
            dict(r3, detector="b", preset="p")]
    lb = leaderboard(rows)
    assert lb[0]["detector"] == "a" and lb[0]["detected"] == 1
    md = report_md(rows, [tk, na], ScoreCfg())
    assert "Leaderboard" in md and "| p | a |" in md


def test_score_end_to_end_on_simulated_recording(tmp_path):
    """A still take at the simulated 1 m target: the default detectors must hit it; `ravidata score` path."""
    from ravidata.config import presets
    from ravidata.engine import Engine
    from ravidata.groundtruth import score_runs
    eng = Engine(runs_dir=tmp_path)
    eng.start_sim(presets()["legal-1d-40mm"].copy(frame_period_ms=20), "static-target-1m", record=True)
    t0 = time.time()
    while eng.frame_counter < 20 and time.time() - t0 < 20:
        time.sleep(0.02)
    eng.note(take_start_text("S8", 1), kind="marker")
    eng.note("d=1.0 m", kind="marker")
    while eng.frame_counter < 120 and time.time() - t0 < 40:
        time.sleep(0.02)
    eng.note("take end", kind="marker")
    d = eng.session.dir
    eng.stop()
    from ravidata.processing.pipeline import DETECTORS
    rows, takes = score_runs([d], {k: {"kind": k} for k in ("track", "cfar") if k in DETECTORS})
    assert len(takes) == 1 and takes[0].mode == "still"
    best = max(r.get("hit %", 0) for r in rows)
    assert best > 80, rows
