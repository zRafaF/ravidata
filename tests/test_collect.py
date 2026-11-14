"""TCC-001 scripted recorder on the simulator (no hardware)."""

import csv

from ravidata.collect import CONFIGS, CSV_COLS, Collector, ask_session_meta, tcc_plan


def test_plan_shapes():
    closed, open_ = tcc_plan("closed"), tcc_plan("open", bonus=True)
    ids = [s.sid for s in open_]
    assert len(ids) == len(set(ids))
    assert not any(s.test == "T3" for s in closed) and sum(s.test == "T3" for s in open_) == 5
    assert any(s.test == "T4" for s in open_)
    walk = tcc_plan("open", walk=True, bonus=True)
    assert sum(s.test == "T5" for s in walk) == 5 and not any(s.test == "T5" for s in open_)
    can = next(s for s in closed if s.sid == "T1-can-r2.0-a+30")
    assert can.xy() == (1.0, 1.73)


def test_collect_sim(tmp_path):
    steps = [s for s in tcc_plan("closed") if s.sid in ("T0-empty", "T1-can-r1.0-a+0")]
    for s in steps:
        s.seconds = 0.6
    answers = iter(["", ""])
    col = Collector(lambda eng, cfg, label: eng.start_sim(cfg, record=True), "closed", runs_dir=tmp_path,
                    ask=lambda prompt: next(answers, ""), beep=lambda n=1: None, out=lambda *a: None)
    rows = col.run(steps)
    assert len(rows) == 2 * len(CONFIGS)
    assert all(r["status"] == "ok" and r["frames"] > 0 for r in rows), rows
    with (tmp_path / "TCC_TAKES.csv").open(encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert [g["config"] for g in got] == list(CONFIGS) * 2
    assert list(got[0].keys()) == CSV_COLS and all(float(g["t_beep_end"]) > float(g["t_beep_start"]) for g in got)
    assert all((tmp_path.joinpath(g["run_dir"]).exists() or __import__("pathlib").Path(g["run_dir"]).exists()) for g in got)


def test_session_meta(tmp_path):
    answers = iter(["stand", "0.92"])
    sid, meta = ask_session_meta(tmp_path, "closed", ask=lambda p: next(answers, ""), out=lambda *a: None)
    assert meta["mount"] == "stand" and meta["radar_height_m"] == "0.92"
    assert (tmp_path / f"TCC_SESSION_{sid}.json").exists()
    sid2, meta2 = ask_session_meta(tmp_path, "open", ask=lambda p: "", out=lambda *a: None)   # Enter keeps the last values
    assert meta2["radar_height_m"] == "0.92"
