"""Printable calibration targets: AprilTags (tag36h11) in several sizes, and a camera-calibration checkerboard.

``uv run ravidata print-targets`` writes, to ``docs/protocols/targets/``:

* ``apriltags_A4.pdf``: every tag of ``TAG_SET`` at its exact size (vector PDF, one ID per tag, IDs unique across sizes),
* ``checkerboard_A4.pdf``: 10 x 7 squares of 25 mm (9 x 6 inner corners) on landscape A4,
* ``tags.json``: tag ID -> black-square size in mm, family and suggested use (the detector reads it).

Print at **100 % / "actual size"** (not "fit to page"), then measure the 100 mm scale bar and the black square: if
they differ, write the measured size into ``tags.json`` (``size_mm``) - the pose accuracy depends on it.

Tag size convention (AprilTag library): the size is the edge of the **black square** (8 x 8 cells for tag36h11:
the black border + 6 x 6 data cells). A white quiet zone of 1 cell is printed around it; keep it when cutting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FAMILY = "tag36h11"
# (ids, black-square size in mm, tags per row, rows per page, suggested use)
TAG_SET = [
    (list(range(0, 3)), 150.0, 1, 1, "person's chest (front/back), large board"),
    (list(range(10, 18)), 90.0, 1, 2, "panel, chair, head-height board, wall, floor marks"),
    (list(range(20, 32)), 55.0, 2, 3, "broom (on a flat card like a flag), can stand, small objects"),
    (list(range(40, 52)), 40.0, 3, 4, "can, very small objects, close range only"),
]
QUIET_CELLS = 1                    # white border of one cell (= the 10 x 10 layout of tag36h11); keep it when cutting
A4 = (210.0, 297.0)
CHECKER = {"squares": (10, 7), "square_mm": 25.0}


def tag_bits(tag_id: int) -> np.ndarray:
    """8 x 8 cells (True = black) of a tag36h11 marker, black border included (OpenCV's AprilTag dictionary)."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    img = cv2.aruco.generateImageMarker(d, int(tag_id), 8, borderBits=1)
    return img == 0


def tag_table() -> dict:
    tags = {}
    for ids, size, *_rest, use in TAG_SET:
        for i in ids:
            tags[str(i)] = {"family": FAMILY, "size_mm": size, "use": use}
    return {"family": FAMILY, "size_convention": "edge of the black square (8 of the 10 cells incl. quiet zone)",
            "printed": "docs/protocols/targets/apriltags_A4.pdf", "tags": tags,
            "checkerboard": {"squares": list(CHECKER["squares"]), "inner_corners": [CHECKER["squares"][0] - 1,
                             CHECKER["squares"][1] - 1], "square_mm": CHECKER["square_mm"],
                             "printed": "docs/protocols/targets/checkerboard_A4.pdf"}}


class _Pdf:
    """A4 PDF with a millimetre coordinate system (origin top left), via Qt's QPdfWriter."""

    def __init__(self, path: Path, landscape: bool = False, dpi: int = 1200):
        from PySide6.QtCore import QMarginsF
        from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
        self.w = QPdfWriter(str(path))
        self.w.setResolution(dpi)
        self.w.setPageSize(QPageSize(QPageSize.A4))
        self.w.setPageOrientation(QPageLayout.Landscape if landscape else QPageLayout.Portrait)
        self.w.setPageMargins(QMarginsF(0, 0, 0, 0), QPageLayout.Millimeter)
        self.w.setTitle(path.stem)
        self.w.setCreator("ravidata print-targets")
        self.k = dpi / 25.4
        self.p = QPainter(self.w)
        self.page_mm = (A4[1], A4[0]) if landscape else A4

    def rect(self, x, y, w, h, black=True):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor
        self.p.fillRect(QRectF(x * self.k, y * self.k, w * self.k, h * self.k), QColor("black" if black else "white"))

    def line(self, x0, y0, x1, y1, width_mm=0.2, dash=False, grey=False):
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QColor, QPen
        pen = QPen(QColor("#888" if grey else "black"), width_mm * self.k)
        if dash:
            pen.setStyle(Qt.DashLine)
        self.p.setPen(pen)
        self.p.drawLine(QPointF(x0 * self.k, y0 * self.k), QPointF(x1 * self.k, y1 * self.k))

    def text(self, x, y, s, pt=9, w=180.0, h=8.0, bold=False, center=False):
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont
        f = QFont("Arial")
        f.setPointSizeF(pt)
        f.setBold(bold)
        self.p.setFont(f)
        self.p.setPen(QColor("black"))
        self.p.drawText(QRectF(x * self.k, y * self.k, w * self.k, h * self.k),
                        (Qt.AlignHCenter if center else Qt.AlignLeft) | Qt.AlignVCenter, s)

    def scale_bar(self, x, y, length=100.0):
        self.line(x, y, x + length, y, 0.3)
        for i in range(0, int(length) + 1, 10):
            self.line(x + i, y - (3 if i % 50 == 0 else 1.5), x + i, y, 0.2)
        self.text(x + length + 3, y - 4, f"{length:.0f} mm: check with a ruler (print at 100 %)", pt=7, w=80, h=6)

    def new_page(self):
        self.w.newPage()

    def close(self):
        self.p.end()


def _draw_tag(pdf: _Pdf, x: float, y: float, tag_id: int, size: float, use: str):
    """Tag with its quiet zone at (x, y) = top-left of the quiet zone; cut lines and label below."""
    cell = size / 8.0
    q = QUIET_CELLS * cell
    full = size + 2 * q
    bits = tag_bits(tag_id)
    for r in range(8):
        for c in range(8):
            if bits[r, c]:
                pdf.rect(x + q + c * cell, y + q + r * cell, cell + 0.01, cell + 0.01)
    # cut marks just outside the quiet zone (dashed, light)
    for (x0, y0, x1, y1) in ((x, y, x + full, y), (x, y + full, x + full, y + full),
                             (x, y, x, y + full), (x + full, y, x + full, y + full)):
        pdf.line(x0, y0, x1, y1, 0.15, dash=True, grey=True)
    lab_pt = 9 if size >= 100 else 7 if size >= 60 else 6
    pdf.text(x, y + full + 0.5, f"{FAMILY}  ID {tag_id}   black square {size:.0f} mm", pt=lab_pt, w=full + 30, h=5,
             bold=True)
    if size >= 60:
        pdf.text(x, y + full + 5, use, pt=6, w=full + 30, h=4)


def write_tags_pdf(path: Path) -> Path:
    pdf = _Pdf(path)
    first = True
    for ids, size, per_row, rows, use in TAG_SET:
        cell = size / 8.0
        full = size + 2 * QUIET_CELLS * cell
        label_h = 11 if size >= 60 else 7
        per_page = per_row * rows
        for start in range(0, len(ids), per_page):
            if not first:
                pdf.new_page()
            first = False
            chunk = ids[start:start + per_page]
            gx = (A4[0] - per_row * full) / (per_row + 1)
            top, bottom = 22.0, A4[1] - 16.0                       # header above, scale bar below
            gy = (bottom - top - rows * (full + label_h)) / (rows + 1)
            assert gx >= 5 and gy >= 0, f"{size} mm tags do not fit {per_row} x {rows} on A4"
            pdf.text(15, 8, f"AprilTags {FAMILY}: black square {size:.0f} mm  (ravidata, docs/protocols/targets)",
                     pt=10, w=180, h=6, bold=True)
            pdf.text(15, 14, "Print at 100 % (actual size). Cut on the dashed line: keep the white border. "
                     "Glue flat on cardboard, matte paper, no foil.", pt=7, w=185, h=5)
            pdf.scale_bar(15, A4[1] - 8)
            for k, tid in enumerate(chunk):
                r, c = divmod(k, per_row)
                x = gx + c * (full + gx)
                y = top + gy + r * (full + label_h + gy)
                _draw_tag(pdf, x, y, tid, size, use)
    pdf.close()
    return path


def write_checkerboard_pdf(path: Path) -> Path:
    nx, ny = CHECKER["squares"]
    s = CHECKER["square_mm"]
    pdf = _Pdf(path, landscape=True)
    W, H = pdf.page_mm
    x0, y0 = (W - nx * s) / 2, (H - ny * s) / 2 + 3
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                pdf.rect(x0 + i * s, y0 + j * s, s + 0.01, s + 0.01)
    pdf.text(x0, 6, f"Camera calibration checkerboard: {nx} x {ny} squares of {s:.0f} mm = {nx - 1} x {ny - 1} inner "
             "corners. Print at 100 %, glue FLAT on a rigid board.", pt=8, w=nx * s, h=6)
    pdf.scale_bar(x0, H - 6)
    pdf.close()
    return path


def write_all(out_dir: Path) -> list[Path]:
    import os

    from PySide6.QtGui import QGuiApplication
    if QGuiApplication.instance() is None and not os.environ.get("DISPLAY") and os.name != "nt":
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    _app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841  (QPdfWriter needs an application)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [write_tags_pdf(out_dir / "apriltags_A4.pdf"), write_checkerboard_pdf(out_dir / "checkerboard_A4.pdf")]
    p = out_dir / "tags.json"
    p.write_text(json.dumps(tag_table(), indent=1), encoding="utf-8")
    return paths + [p]


def render_pdf_pages(path: Path, dpi: int = 150) -> list[np.ndarray]:
    """Rasterise a PDF (Qt PDF module) into grey images, for checking the tags with the detector."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtPdf import QPdfDocument
    doc = QPdfDocument()
    doc.load(str(path))
    out = []
    for i in range(doc.pageCount()):
        sz = doc.pagePointSize(i)
        img = doc.render(i, QSize(int(sz.width() / 72 * dpi), int(sz.height() / 72 * dpi)))
        bg = QImage(img.size(), QImage.Format_RGB32)          # the render is transparent: put it on white paper
        bg.fill(QColor("white"))
        qp = QPainter(bg)
        qp.drawImage(0, 0, img)
        qp.end()
        img = bg.convertToFormat(QImage.Format_Grayscale8)
        a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width()].copy()
        out.append(a)
    doc.close()
    return out


def detect_tags(gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """[(id, 4x2 corners px)] of tag36h11 markers in a grey image (OpenCV)."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    return [] if ids is None else [(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]
