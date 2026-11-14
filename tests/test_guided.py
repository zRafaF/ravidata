"""Simple tab of the recorder: scenes, the sync-swipe check, and a guided take on the simulator."""

import csv
import os
import time

import numpy as np
import pytest

from ravidata.scenes import SCENES, SwipeDetector, config_order, scene, take_queue, verdict


def test_scenes_and_order():
    from ravidata.config import presets
    assert config_order(["a", "b"], 2) == ["a", "b", "b", "a"]
    assert config_order(["a", "b", "c"], 1) == ["a", "b", "c"]
    for s in SCENES:
        for c in s.configs or []:
            assert c in presets(), c
    assert scene("D2-carried-outdoor").env == "open"
    st = scene("S-static")
    assert [p.pid for p in st.phases] == ["S0", "S1", "S2"]
    q = take_queue(st, ["1d", "2d"], 1, [10, 30, 45])
    assert q == [(0, "1d", 10), (0, "2d", 10), (1, "1d", 30), (1, "2d", 30), (2, "1d", 45), (2, "2d", 45)]
    assert [x[2] for x in take_queue(st, ["1d"], 1)] == [10, 20, 0]                 # the parts' defaults


def test_swipe_detector():
    r = np.arange(128) * 0.04
    rng = np.random.default_rng(0)
    det = SwipeDetector()
    base = 1.0 + 0.05 * rng.random(128)
    base[60] = 50.0                                   # a strong static target at 2.4 m: not near, ignored
    for k in range(60):                               # 6 s at 10 updates/s; hand at 0.5 m during 2.0-2.5 s
        p = base * (1 + 0.05 * rng.standard_normal(128))
        t = k / 10
        if 2.0 <= t < 2.5:
            p[12:14] += 20.0
        det.update(t, r, p)
    assert det.events == [2.0]
    assert det.seen_start()
    quiet = SwipeDetector()
    for k in range(60):
        quiet.update(k / 10, r, base * (1 + 0.05 * rng.standard_normal(128)))
    assert quiet.events == []


def test_verdict():
    ok = {"status": "ok", "frames": 100, "loss_pct": 2.0}
    assert verdict(ok, [1.0])[0] == "ok"
    assert verdict(ok, [])[0] == "warn"
    assert verdict(ok | {"loss_pct": 12.0}, [1.0])[0] == "ok"                        # 2D at 20 fps loses ~10 %
    assert verdict(ok | {"loss_pct": 13.0}, [1.0])[0] == "warn"
    assert verdict({"status": "high_loss", "frames": 50, "loss_pct": 20.0}, [1.0])[0] == "bad"
    assert verdict({"status": "no_frames", "frames": 0}, None)[0] == "bad"


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _spin(app, w, seconds, until=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        app.processEvents()
        w.tick()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return False


def test_rig_file(tmp_path):
    from ravidata.collect import load_rig, save_rig
    assert load_rig(tmp_path) == {}
    save_rig(tmp_path, {"radar_rotation_deg": "270", "phone_orientation": "landscape", "place": "lab"})
    rig = load_rig(tmp_path)
    assert rig["radar_rotation_deg"] == "270" and "place" not in rig


def test_guided_take_sim(tmp_path, monkeypatch):
    try:
        from PySide6 import QtWidgets
        from PySide6.QtCore import Qt
    except ImportError as e:   # no libEGL in a headless Linux VM
        pytest.skip(f"Qt not available: {e}")
    from ravidata.engine import Engine
    from ravidata.gui import guided as G
    from ravidata.gui import recorder as R

    R.beep = lambda n=1: None
    monkeypatch.setattr(G.GuidedPanel, "countdown_s", 0.0)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    eng = Engine(runs_dir=tmp_path, process=False)
    w = R.RecorderWindow(eng, tmp_path, sim_default=True)
    w.timer.stop()
    g = w.guided
    names = iter(["Rafael static, white wall", None])
    g.ask_name = lambda default: next(names)                             # the naming box at the end of a take

    def tick_configs(names):
        for i in range(g.cfg_list.count()):
            it = g.cfg_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in names else Qt.Unchecked)

    def rows():
        with (tmp_path / "TCC_TAKES.csv").open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    try:
        assert w.tabs.currentIndex() == 0 and not g.btn.isEnabled()      # checklist not ticked yet
        g.scene.setCurrentIndex([s.sid for s in SCENES].index("S-static"))
        tick_configs(("tcc-1d-40mm", "tcc-2d-40mm"))
        assert [sp.value() for sp in g.part_secs] == [10, 20, 0]
        for sp in g.part_secs:                                           # 1 s per config in every part
            sp.setValue(1)
        g.lead.setValue(1)                                               # sync lead-in of the first recording
        for c in g.checks:
            c.setChecked(True)
        assert g.btn.isEnabled()
        w.toggle_record()                                                # F9 on the Simple tab starts the take
        assert g.busy
        for part in (1, 2):                                              # F9 between the parts
            assert _spin(app, w, 60, until=lambda: g.run is not None and g.run["phase"] == "ready")
            assert g.run["part"] == part and "F9" in g.now.text()
            w.toggle_record()
        assert _spin(app, w, 60, until=lambda: not g.busy)
        rr = rows()
        assert [r["config"] for r in rr] == ["tcc-1d-40mm", "tcc-2d-40mm"] * 3
        assert [r["test"] for r in rr] == ["S0", "S0", "S1", "S1", "S2", "S2"]
        assert all(r["status"] == "ok" for r in rr) and len({r["step"] for r in rr}) == 1
        assert rr[0]["step"].startswith("S-static@") and "Empty scene" in rr[0]["text"]
        assert "sync lead-in: first 1.0 s" in rr[0]["text"] and "lead-in" not in rr[1]["text"]
        assert float(rr[0]["seconds"]) >= 1.8 and float(rr[1]["seconds"]) < 1.8
        assert "TAKE" in g.now.text() and len(g.results) == 6
        assert not g.checks[0].isChecked()                               # phone boxes reset for the next take
        import json
        f1 = g.last_folder
        assert f1.parent == tmp_path / "recs" and f1.name.startswith("Rafael-static-white-wall_")
        assert len(f1.name.split("_")[-1]) == len("20261113-171846") and (f1 / "README.md").exists()
        info = json.loads((f1 / "take.json").read_text(encoding="utf-8"))
        assert info["scene"] == "S-static" and [x["part"] for x in info["recordings"]] == ["S0", "S0", "S1", "S1",
                                                                                           "S2", "S2"]
        assert info["recordings"][0]["run"] and info["take_id"] == rr[0]["step"]
        assert info["seconds_per_part"] == {"S0": 1, "S1": 1, "S2": 1} and info["recordings"][0]["lead_in_s"] == 1

        # length 0: F9 moves on to the next config, then ends the take
        g.scene.setCurrentIndex([s.sid for s in SCENES].index("S2-moving-only"))
        assert [sp.value() for sp in g.part_secs] == [0]
        real_start, calls = w.start_stream, []

        def flaky(cfg=None, label=""):                                  # the radar fails once: retried
            calls.append(cfg.name)
            return False if len(calls) == 1 else real_start(cfg, label=label)
        w.start_stream = flaky
        tick_configs(("tcc-1d-40mm",))
        g.notes.setText("broom tags 20/21")
        for c in g.checks:
            c.setChecked(True)
        g.f9()
        assert _spin(app, w, 30, until=lambda: g.run is not None and g.run["phase"] == "recording")
        assert calls == ["tcc-1d-40mm", "tcc-1d-40mm"]
        _spin(app, w, 1.5)                                               # past the 1 s lead-in
        w.toggle_record()
        assert _spin(app, w, 10, until=lambda: not g.busy)
        rr = rows()
        assert len(rr) == 7 and rr[-1]["step"].startswith("S2-moving-only@") and rr[-1]["test"] == "S2"
        assert "broom tags 20/21" in rr[-1]["text"] and float(rr[-1]["seconds"]) >= 0.8
        assert g.last_folder.name.startswith("s2-moving-only-pass1_")                    # cancelled: default name
        # the rig: lock -> runs/RIG.json -> loaded (and locked) by the next window
        w.meta_edits["radar_height_m"].setText("1.10")
        w.rot_combo.setCurrentIndex(3)
        w.rig_lock.setChecked(True)
        assert (tmp_path / "RIG.json").exists() and not w.rot_combo.isEnabled()
        assert "locked" in g.rig_lbl.text() and "1.10 m" in g.rig_lbl.text()
    finally:
        w.close()
    w2 = R.RecorderWindow(Engine(runs_dir=tmp_path, process=False), tmp_path, sim_default=True)
    w2.timer.stop()
    try:
        assert w2.rig_lock.isChecked() and w2.meta_edits["radar_height_m"].text() == "1.10"
        assert w2._rotation() == 270 and not w2.rot_combo.isEnabled()
    finally:
        w2.close()
