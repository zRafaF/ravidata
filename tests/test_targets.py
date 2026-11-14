"""Printable targets: every AprilTag in the PDF is found again by the detector at its nominal size, and the
checkerboard's inner corners are found."""

import os

import numpy as np
import pytest

pytest.importorskip("cv2")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtGui import QGuiApplication
except ImportError as e:   # no libEGL in a headless Linux VM
    pytest.skip(f"Qt not available: {e}", allow_module_level=True)


def test_targets(tmp_path):
    import cv2

    from ravidata.targets import CHECKER, detect_tags, render_pdf_pages, tag_table, write_all
    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    write_all(tmp_path)
    tt = tag_table()["tags"]
    dpi = 150
    found = {}
    for page in render_pdf_pages(tmp_path / "apriltags_A4.pdf", dpi):
        for tid, c in detect_tags(page):
            assert tid not in found, f"tag {tid} printed twice"
            found[tid] = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)]) / (dpi / 25.4)
    assert sorted(found) == sorted(int(k) for k in tt)
    for i, edge in found.items():
        assert edge == pytest.approx(tt[str(i)]["size_mm"], abs=0.6)
    board = render_pdf_pages(tmp_path / "checkerboard_A4.pdf", dpi)[0]
    nx, ny = CHECKER["squares"]
    ok, _ = cv2.findChessboardCorners(board, (nx - 1, ny - 1))
    assert ok


def test_new_2d_presets_valid():
    from ravidata.config import has_errors, presets
    P = presets()
    for n in ("tcc-2d-40mm", "tcc-2d-40mm-12fps", "tcc-2d-40mm-k8"):
        assert not has_errors(P[n].validate()), n
