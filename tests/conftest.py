"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_recorder_settings(tmp_path, monkeypatch):
    """The recorder keeps its settings in QSettings (the Windows registry); tests must not touch the owner's."""
    try:
        from PySide6.QtCore import QSettings

        import ravidata.gui.recorder as R
    except Exception:          # no Qt (e.g. no libEGL in a headless Linux VM): the GUI tests skip anyway
        yield
        return
    ini = str(tmp_path / "qt-settings.ini")
    monkeypatch.setattr(R, "QSettings", lambda *a, **k: QSettings(ini, QSettings.IniFormat))
    yield
