"""Recorder GUI on the simulator (skipped where Qt cannot load, e.g. without libEGL)."""

import csv
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6 import QtWidgets
except ImportError as e:   # e.g. no libEGL in a headless Linux VM
    pytest.skip(f"Qt not available: {e}", allow_module_level=True)


def _spin(app, w, seconds, until=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        app.processEvents()
        w.tick()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return False


def test_recorder_sim(tmp_path):
    from ravidata.engine import Engine
    from ravidata.gui import recorder as R

    R.beep = lambda n=1: None
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    eng = Engine(runs_dir=tmp_path, process=False)
    w = R.RecorderWindow(eng, tmp_path, sim_default=True)
    w.timer.stop()
    try:
        # free recording with the 2D config
        w.preset.setCurrentText("tcc-2d-40mm")
        assert w.start_stream()
        assert _spin(app, w, 10, until=lambda: eng.latest_frame is not None and getattr(w, "_peak", None) is not None)
        assert w._peak["rd"] is not None and w._wf is not None
        w.secs.setValue(0)
        assert w.start_record()
        _spin(app, w, 1.0)
        row = w.stop_record()
        assert row["status"] == "ok" and row["frames"] > 0 and row["config"] == "tcc-2d-40mm"
        w.stop_stream()
        # one protocol step = both TCC configs back to back
        idx = [s.sid for s in w.steps].index("T1-can-r2.0-a+30") + 1
        w.step.setCurrentIndex(idx)
        w.secs.setValue(1)
        w.record_step()
        assert _spin(app, w, 40, until=lambda: w.seq is None)
        with (tmp_path / "TCC_TAKES.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["config"] for r in rows] == ["tcc-2d-40mm", "tcc-1d-40mm", "tcc-2d-40mm"]
        assert all(r["status"] == "ok" for r in rows) and rows[-1]["step"] == "T1-can-r2.0-a+30"
        assert rows[-1]["x_m"] == "1.0" and list(tmp_path.glob("TCC_SESSION_*.json"))
        assert w.step.currentIndex() == idx + 1          # moved on to the next step
    finally:
        w.close()


def test_check_link_sim(tmp_path, monkeypatch):
    import ravidata.session
    from ravidata.engine import Engine
    from ravidata.gui import recorder as R

    monkeypatch.setattr(ravidata.session, "default_runs_dir", lambda: tmp_path)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = R.RecorderWindow(Engine(runs_dir=tmp_path, process=False), tmp_path, sim_default=True)
    w.timer.stop()
    try:
        w.check_link()
        assert _spin(app, w, 30, until=lambda: getattr(w, "_probe", None) is None)
        assert "SPI link OK" in w.state.text(), w.state.text()
        assert list(tmp_path.glob("SPIPROBE_*.md"))
    finally:
        w.close()
