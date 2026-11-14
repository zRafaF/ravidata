"""Lean radar recorder: choose/edit a radar config, stream, record, sanity plots and link health.

Start with ``uv run ravidata rec`` (or ``make rec``); ``--sim`` for a dry run without hardware.

Made for a slow laptop during data collection (docs/protocols/TCC-001-collection.md):

* no detectors run (``Engine(process=False)``); the plots redraw ~8 times a second from the newest frame only,
  so drawing never holds up the radar link;
* every recording is a normal run directory (``cubes.h5`` + logs) and one row in ``runs/TCC_TAKES.csv``;
* a beep when a recording starts and two when it stops: the phone's microphone hears them, which syncs the
  phone (video + IMU) with the radar;
* "Record step" runs one TCC-001 protocol step: every TCC config back to back, each for N seconds.

Tabs on the left: **Simple** (default: a guided take = one scene recorded with several configs back to back, a
checklist, what to do now, the sync-swipe check and a verdict per recording; ``gui/guided.py``), **Advanced** (config
editor, free recording, TCC-001 steps) and **Setup** (rig measurements, once per rig).
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont, QKeySequence, QPainter, QPen, QPolygonF, QShortcut
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget)

from ..collect import (CONFIGS, RIG_KEYS, SESSION_FIELDS, Step, append_take_row, last_session_meta, load_rig, save_rig,
                       save_session_meta, tcc_plan)
from .. import orientation
from ..config import RadarConfig, has_errors, presets
from ..engine import Engine
from ..logsetup import get_logger, setup_base
from ..session import default_runs_dir
from .forms import ConfigForm
from .guided import GuidedPanel

log = get_logger("rec")

LINKS = ["SPI (CH347, MODE SEL ON)", "UART (2 ports)", "Simulator"]
RX_COLORS = ["#4fc3f7", "#ffb74d", "#aed581"]
WATERFALL_ROWS = 160
MIN_RANGE_M = 0.25          # ignore TX leakage bins for the peak readout
BAD_LOSS_PCT = 10.0


def beep(n: int = 1) -> None:
    """Non-blocking beep(s): winsound on Windows, the terminal bell elsewhere."""
    def run():
        try:
            import winsound
            for _ in range(n):
                winsound.Beep(1200, 180)
                time.sleep(0.08)
        except Exception:
            sys.stdout.write("\a" * n)
            sys.stdout.flush()
    threading.Thread(target=run, daemon=True).start()


def _ports() -> list[str]:
    try:
        from serial.tools import list_ports
        return [f"{p.device} - {p.description}" for p in sorted(list_ports.comports(), key=lambda p: p.device)]
    except Exception:
        return []


def _port_name(text: str) -> str:
    return text.split(" - ")[0].strip()


def _likely_radar_port(ports: list[str]) -> str:
    """The dev board's USB bridge is a WCH CH34x; prefer it when nothing was chosen before."""
    for key in ("CH34", "WCH", "USB-SERIAL", "USB Serial", "USB"):
        for p in ports:
            if key.lower() in p.lower():
                return p
    return ports[0] if ports else ""


class PortCombo(QComboBox):
    """Editable port list that re-scans the serial ports every time it is opened."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)

    def refresh(self):
        cur = self.currentText()
        ports = _ports()
        self.blockSignals(True)
        self.clear()
        self.addItems(ports)
        if cur:
            names = [_port_name(x) for x in ports]
            self.setCurrentIndex(names.index(_port_name(cur))) if _port_name(cur) in names else self.setEditText(cur)
        elif ports:
            self.setCurrentText(_likely_radar_port(ports))
        self.blockSignals(False)

    def showPopup(self):
        self.refresh()
        super().showPopup()


def frame_views(cube: np.ndarray, cfg: RadarConfig, pairs: dict | None = None) -> dict:
    """Everything the plots show, from one frame. ``cube``: complex (n_dop, n_rx, n_range), +-1 full scale.
    ``pairs``: ``orientation.pairs(rotation)`` = which RX pair is horizontal / vertical in this mounting."""
    nd, nrx, nr = cube.shape
    d = cfg.derive()
    bin_m = d.bin_mm / 1000.0
    r = np.arange(nr) * bin_m
    pw = np.abs(cube) ** 2                                   # (nd, nrx, nr)
    prof_rx = 10 * np.log10(pw.mean(axis=0) + 1e-12)          # dBFS per RX, Doppler/intervals averaged
    tot = pw.mean(axis=(0, 1))
    tot_db = 10 * np.log10(tot + 1e-12)
    valid = (r >= MIN_RANGE_M) & (np.arange(nr) < nr - 2)
    if not valid.any():
        valid[:] = True
    floor = float(np.median(tot_db[valid]))
    k = int(np.flatnonzero(valid)[np.argmax(tot_db[valid])])
    # interferometric angle per range bin for the horizontal and the vertical RX pair of this mounting:
    # phase of b relative to a (b = right of / above a, seen from the front). Uncalibrated, sign not verified.
    pairs = pairs or orientation.pairs(0)
    out = {"r": r, "prof_rx": prof_rx, "tot_db": tot_db, "floor": floor, "peak_i": k, "peak_r": float(r[k]),
           "peak_snr": float(tot_db[k] - floor), "rd": None, "v_axis": None}
    for key in ("horizontal", "vertical"):
        ang = None
        a, b = (orientation.RX_INDEX[n] for n in pairs[key])
        if nrx > max(a, b):
            x = (cube[:, b, :] * np.conj(cube[:, a, :])).sum(axis=0)       # (nr,)
            s = np.angle(x) / (2 * np.pi * d.d_over_lambda)
            ang = np.degrees(np.arcsin(np.clip(s, -1, 1)))
        out["angle_" + key[0]] = ang
        out["peak_ang_" + key[0]] = None if ang is None else float(ang[k])
    if nd > 1:
        if cfg.dop_fft:
            S = cube                                         # chip Doppler FFT, zero velocity in the middle
        else:                                                # slow-time samples: Doppler on the host for display
            w = np.hanning(nd).astype(np.float32)
            S = np.fft.fftshift(np.fft.fft(cube * w[:, None, None], axis=0), axes=0)
        out["rd"] = 10 * np.log10((np.abs(S) ** 2).mean(axis=1) + 1e-12)          # (nd, nr)
        vres = (cfg.veloc_res_mm_s or 0) / 1000.0
        if not vres and cfg.veloc_max_mm_s:
            vres = 2 * cfg.veloc_max_mm_s / 1000.0 / nd
        out["v_axis"] = (np.arange(nd) - nd // 2) * (vres or 1.0)
    return out


FIELD_LABELS = {"radar_height_m": "h: radar height (m)", "cam_dx_cm": "dx: camera to the right (cm)",
                "cam_dy_cm": "dy: camera above (cm)", "cam_dz_cm": "dz: camera behind (cm)"}


MODULE_MM = (18.0, 12.0)          # MRS6130-P1812 outline (module guide Fig. 7-1), rotation 0 = landscape
CHIP_MM = (5.6, 8.8)              # RS6130 package on the module, portrait at rotation 0 (approx., Fig. 7-1)
H_COL, V_COL = QColor("#1e88e5"), QColor("#fb8c00")


def _arrow(p: QPainter, a: QPointF, b: QPointF, color: QColor, both: bool = False):
    p.setPen(QPen(color, 2))
    p.drawLine(a, b)
    d = b - a
    n = max((d.x() ** 2 + d.y() ** 2) ** 0.5, 1e-6)
    ux, uy = d.x() / n, d.y() / n
    for tip, s in ((b, 1),) + (((a, -1),) if both else ()):
        base = QPointF(tip.x() - s * 9 * ux, tip.y() - s * 9 * uy)
        left = QPointF(base.x() - 4.5 * uy, base.y() + 4.5 * ux)
        right = QPointF(base.x() + 4.5 * uy, base.y() - 4.5 * ux)
        p.setBrush(QBrush(color))
        p.drawPolygon(QPolygonF([tip, left, right]))


def _pin1_xy(rotation: int, mirror: bool) -> tuple[int, int]:
    """Corner of the module's pin-1 dot as (sx, sy) in {-1, +1} (x right, y down on screen)."""
    sx, sy = {0: (-1, -1), 90: (1, -1), 180: (1, 1), 270: (-1, 1)}[rotation]
    return (-sx if mirror else sx), sy


class OffsetDiagram(QWidget):
    """Sketch of the session measurements: dx/dy seen from behind the rig, dz/dy/h seen from the side.
    All distances are measured from the centre of the radar chip to the centre of the phone lens.
    Follows the chosen radar rotation (module outline) and phone orientation (portrait / landscape)."""

    RED, GREEN, BLUE, ORANGE = QColor("#e53935"), QColor("#43a047"), QColor("#1e88e5"), QColor("#fb8c00")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation, self.phone = 0, "landscape"
        self.setMinimumSize(400, 215)
        self.setToolTip("Measure from the centre of the radar chip (antenna package) to the centre of the phone lens. "
                        "The small circle on the radar is the module's pin-1 dot (on the front side, seen through).")

    def set_orientation(self, rotation: int, phone: str):
        self.rotation, self.phone = int(rotation) % 360, phone
        self.update()

    _arrow = staticmethod(_arrow)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#fafafa"))
        w, hgt = self.width(), self.height()
        half = w / 2
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        dark, grey = QColor("#333"), QColor("#9e9e9e")
        top = 22                                           # below the titles
        portrait = self.phone == "portrait"

        # ---- left: seen from behind the rig (you look the same way as the radar)
        p.setPen(dark)
        p.drawText(QRectF(2, 2, half - 4, 16), Qt.AlignHCenter, "from BEHIND the rig")
        chip = QPointF(half * 0.30, hgt - 55)
        k = min(4.2, half * 0.40 / MODULE_MM[0])
        mw, mh = (MODULE_MM if self.rotation in (0, 180) else MODULE_MM[::-1])
        mw, mh = mw * k, mh * k
        p.setPen(QPen(dark, 1))
        p.setBrush(QBrush(QColor("#cfd8dc")))
        p.drawRect(QRectF(chip.x() - mw / 2, chip.y() - mh / 2, mw, mh))          # radar module (back side)
        sx, sy = _pin1_xy(self.rotation, mirror=True)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(chip.x() + sx * (mw / 2 - 5), chip.y() + sy * (mh / 2 - 5)), 2.5, 2.5)
        p.setBrush(QBrush(self.RED))
        p.drawEllipse(chip, 3, 3)
        p.setPen(dark)
        p.drawText(QRectF(chip.x() - 50, chip.y() + mh / 2 + 2, 100, 14), Qt.AlignHCenter, "radar chip")
        pw_, ph_ = (34, 66) if portrait else (66, 34)
        lens = QPointF(half * 0.74, top + 30)
        # rear camera near the top of a portrait phone, near one end of a landscape one
        body = (QRectF(lens.x() - pw_ / 2, lens.y() - 12, pw_, ph_) if portrait
                else QRectF(lens.x() - 12, lens.y() - ph_ / 2, pw_, ph_))
        p.setBrush(QBrush(QColor("#eceff1")))
        p.drawRoundedRect(body, 5, 5)                                              # phone
        p.setBrush(QBrush(QColor("#212121")))
        p.drawEllipse(lens, 5, 5)
        p.setPen(dark)
        p.drawText(QRectF(lens.x() - 60, top - 4, 120, 14), Qt.AlignHCenter, f"phone lens ({self.phone})")
        corner = QPointF(lens.x(), chip.y())
        self._arrow(p, chip, corner, self.RED)
        p.setPen(self.RED)
        p.drawText(QRectF(chip.x() + 8, chip.y() + 3, corner.x() - chip.x(), 14), Qt.AlignHCenter, "dx (+ right)")
        self._arrow(p, corner, QPointF(lens.x(), lens.y() + 7), self.GREEN)
        p.setPen(self.GREEN)
        p.drawText(QRectF(corner.x() + 4, (corner.y() + lens.y()) / 2 + 14, half - corner.x() - 4, 14),
                   Qt.AlignLeft, "dy (+ up)")

        # ---- right: seen from the side, radar looking to the right
        x0 = half
        p.setPen(QPen(grey, 1, Qt.DashLine))
        p.drawLine(QPointF(half, 4), QPointF(half, hgt - 4))
        p.setPen(dark)
        p.drawText(QRectF(x0 + 2, 2, half - 4, 16), Qt.AlignHCenter, "from the SIDE (radar looks →)")
        floor_y = hgt - 10
        p.setPen(QPen(dark, 2))
        p.drawLine(QPointF(x0 + 6, floor_y), QPointF(w - 6, floor_y))
        p.drawText(QRectF(w - 48, floor_y - 14, 44, 12), Qt.AlignRight, "floor")
        face_x = x0 + half * 0.50
        chip2 = QPointF(face_x, hgt - 75)
        p.setPen(QPen(grey, 1, Qt.DotLine))
        p.drawLine(QPointF(face_x, top), QPointF(face_x, floor_y))                # plane of the radar face
        p.setPen(QPen(dark, 1))
        p.setBrush(QBrush(QColor("#cfd8dc")))
        p.drawRect(QRectF(face_x - 6, chip2.y() - mh / 2, 6, mh))                 # radar board, face on the right
        p.setBrush(QBrush(self.RED))
        p.drawEllipse(chip2, 3, 3)
        self._arrow(p, QPointF(face_x + 4, chip2.y()), QPointF(min(w - 8, face_x + 60), chip2.y()), grey)
        p.setPen(grey)
        p.drawText(QRectF(face_x + 4, chip2.y() - 16, 70, 14), Qt.AlignLeft, "boresight")
        lens2 = QPointF(face_x - 30, top + 40)
        p.setPen(QPen(dark, 1))
        p.setBrush(QBrush(QColor("#eceff1")))
        p.drawRect(QRectF(lens2.x() - 8, lens2.y() - 12, 8, ph_))                 # phone, camera facing right
        p.setBrush(QBrush(QColor("#212121")))
        p.drawEllipse(lens2, 3.5, 3.5)
        self._arrow(p, QPointF(face_x, top + 8), QPointF(lens2.x(), top + 8), self.BLUE)
        p.setPen(self.BLUE)
        p.drawText(QRectF(x0 + 2, top + 10, lens2.x() - x0 - 12, 28), Qt.AlignRight, "dz\n(+ behind)")
        self._arrow(p, QPointF(face_x + 20, chip2.y() - mh / 2 - 2), QPointF(face_x + 20, lens2.y()), self.GREEN)
        p.setPen(self.GREEN)
        p.drawText(QRectF(face_x + 24, lens2.y() + 4, 40, 14), Qt.AlignLeft, "dy")
        hx = face_x - 22
        self._arrow(p, QPointF(hx, floor_y), QPointF(hx, chip2.y()), self.ORANGE, both=True)
        p.setPen(self.ORANGE)
        p.drawText(QRectF(x0 + 2, (floor_y + chip2.y()) / 2 - 14, hx - x0 - 6, 28), Qt.AlignRight,
                   "h" if hx - x0 < 110 else "h (floor to chip)")
        p.end()


class AntennaDiagram(QWidget):
    """The radar face seen from the FRONT (as in the datasheets): the module with its pin-1 dot, an enlarged view
    of the chip with its antennas, and which RX pair is horizontal (azimuth) / vertical (elevation)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rotation = 0
        self.setMinimumSize(400, 200)
        self.setToolTip("Antenna positions from the RS6130 datasheet (Fig. 5-1) and the MRS6130-P1812 module guide "
                        "(Fig. 3-1). Check them with your hand: moving up/down changes the orange (vertical-pair) "
                        "angle, left/right the blue (horizontal-pair) angle.")

    def set_rotation(self, rotation: int):
        self.rotation = int(rotation) % 360
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#fafafa"))
        w, hgt = self.width(), self.height()
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        dark, grey = QColor("#333"), QColor("#9e9e9e")
        p.setPen(dark)
        p.drawText(QRectF(2, 2, w - 4, 16), Qt.AlignHCenter, "radar face seen from the FRONT (as in the datasheets)")
        land = self.rotation in (0, 180)
        mw, mh = MODULE_MM if land else MODULE_MM[::-1]
        cw, ch = CHIP_MM if land else CHIP_MM[::-1]

        # small module: where the pin-1 dot is
        km = 4.0
        mc = QPointF(8 + mw * km / 2, hgt / 2 + 6)
        mrect = QRectF(mc.x() - mw * km / 2, mc.y() - mh * km / 2, mw * km, mh * km)
        p.setPen(QPen(dark, 1))
        p.setBrush(QBrush(QColor("#ef6c4a")))
        p.drawRect(mrect)
        sx, sy = _pin1_xy(self.rotation, mirror=False)
        p.setBrush(QBrush(QColor("white")))
        p.drawEllipse(QPointF(mc.x() + sx * (mw * km / 2 - 5), mc.y() + sy * (mh * km / 2 - 5)), 3, 3)
        small_chip = QRectF(mc.x() - cw * km / 2, mc.y() - ch * km / 2, cw * km, ch * km)
        p.setBrush(QBrush(QColor("#212121")))
        p.drawRect(small_chip)
        p.setPen(dark)
        lw = max(mrect.width() + 12, 84.0)
        p.drawText(QRectF(max(2.0, mc.x() - lw / 2), mrect.bottom() + 2, lw, 28), Qt.AlignHCenter,
                   "module\n○ = pin-1 dot")

        # enlarged chip with the antennas
        kc = 11.0
        cc = QPointF(mrect.right() + 22 + cw * kc / 2, hgt / 2 + 6)
        crect = QRectF(cc.x() - cw * kc / 2, cc.y() - ch * kc / 2, cw * kc, ch * kc)
        p.setPen(QPen(grey, 1, Qt.DashLine))
        p.drawLine(small_chip.topRight(), crect.topLeft())
        p.drawLine(small_chip.bottomRight(), crect.bottomLeft())
        p.setPen(QPen(dark, 1))
        p.setBrush(QBrush(QColor("#263238")))
        p.drawRect(crect)
        pos = orientation.antenna_positions(self.rotation)
        xs = [v[0] for v in pos.values()]
        ys = [v[1] for v in pos.values()]
        ox, oy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

        def scr(name):
            x, y = pos[name]
            return QPointF(cc.x() + (x - ox) * kc, cc.y() - (y - oy) * kc)

        pr = orientation.pairs(self.rotation)
        for key, col in (("horizontal", H_COL), ("vertical", V_COL)):
            p.setPen(QPen(col, 4))
            p.drawLine(scr(pr[key][0]), scr(pr[key][1]))
        a = 18.0
        fs = p.font()
        fs.setPointSize(7)
        p.setFont(fs)
        for name in ("TX", "RX0", "RX1", "RX2"):
            c = scr(name)
            p.setPen(QPen(QColor("#90a4ae"), 1))
            p.setBrush(QBrush(QColor("#78909c") if name == "TX" else QColor("#eceff1")))
            p.drawRect(QRectF(c.x() - a / 2, c.y() - a / 2, a, a))
            p.setPen(QColor("white") if name == "TX" else dark)
            p.drawText(QRectF(c.x() - a / 2 - 4, c.y() - a / 2, a + 8, a), Qt.AlignCenter, name)
        p.setFont(f)
        p.setPen(dark)
        p.drawText(QRectF(crect.left() - 10, crect.bottom() + 2, crect.width() + 20, 14), Qt.AlignHCenter,
                   "chip (enlarged)")

        # summary on the right
        x0 = crect.right() + 14
        h, v = pr["horizontal"], pr["vertical"]
        lines = [(f"pin-1 dot: {orientation.pin1_corner(self.rotation)}", dark, True),
                 (f"{h[0]}–{h[1]}: horizontal", H_COL, True), ("   → azimuth (left/right)", H_COL, False),
                 (f"{v[0]}–{v[1]}: vertical", V_COL, True), ("   → elevation (height)", V_COL, False),
                 ("", dark, False),
                 ("Check with a hand at ~0.5 m:", dark, False),
                 ("up/down moves the orange angle,", dark, False),
                 ("left/right moves the blue one.", dark, False)]
        y = 24
        for txt, col, bold in lines:
            fb = p.font()
            fb.setBold(bold)
            p.setFont(fb)
            p.setPen(col)
            p.drawText(QRectF(x0, y, w - x0 - 2, 16), Qt.AlignLeft, txt)
            y += 16
        p.setFont(f)
        _arrow(p, QPointF(x0 + 6, hgt - 6), QPointF(x0 + 6, hgt - 30), dark)
        p.setPen(dark)
        p.drawText(QRectF(x0 + 14, hgt - 24, 40, 14), Qt.AlignLeft, "up")
        p.end()


class NoWheel(QObject):
    """Scrolling the panel with the mouse wheel must not change a dropdown or number box it passes over
    (2026-11-13: the rig rotation and phone orientation changed that way). They take the wheel only when focused."""

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and not obj.hasFocus():
            ev.ignore()
            return True
        return False


class RecorderWindow(QMainWindow):
    def __init__(self, eng: Engine, runs_dir: Path, sim_default: bool = False):
        super().__init__()
        self.eng, self.runs_dir = eng, Path(runs_dir)
        self.settings = QSettings("RAVIDATA", "ravidata-rec")
        self.setWindowTitle("ravidata recorder")
        self.resize(1400, 860)
        self.cfg: RadarConfig = presets()["tcc-1d-40mm"]
        self.session_id = ""
        self.rec: dict | None = None             # the recording in progress
        self.seq: dict | None = None             # the "record step" sequence in progress
        self._last_seq_seen = -1
        self._wf: np.ndarray | None = None
        self._bytes_prev: tuple[float, int] | None = None
        self._spi_cache: dict = {}
        self._spi_t = 0.0

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(self._link_box(sim_default))
        adv = QWidget()
        av = QVBoxLayout(adv)
        av.addWidget(self._config_box())
        av.addWidget(self._record_box())
        av.addStretch(1)
        setup = QWidget()
        sv = QVBoxLayout(setup)
        sv.addWidget(self._session_box())
        sv.addStretch(1)
        self.guided = GuidedPanel(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.guided, "Simple")
        self.tabs.addTab(adv, "Advanced")
        self.tabs.addTab(setup, "Setup (once per rig)")
        lv.addWidget(self.tabs, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        scroll.setMinimumWidth(430)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.stats = QLabel("idle")
        self.stats.setTextFormat(Qt.RichText)
        f = QFont("Consolas")
        f.setStyleHint(QFont.Monospace)
        f.setPointSize(10)
        self.stats.setFont(f)
        self.stats.setStyleSheet("background:#111; color:#ddd; padding:6px;")
        self.stats.setMinimumHeight(118)
        rv.addWidget(self.stats)
        rv.addWidget(self.guided.now)
        rv.addWidget(self._plots(), 1)

        split = QSplitter()
        split.addWidget(scroll)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        self.setCentralWidget(split)

        QShortcut(QKeySequence("F5"), self, activated=self.toggle_stream)
        QShortcut(QKeySequence("F9"), self, activated=self.toggle_record)
        QShortcut(QKeySequence("Ctrl+M"), self, activated=self.add_marker)

        self._nowheel = NoWheel(self)
        for cls in (QComboBox, QAbstractSpinBox):
            for wdg in left.findChildren(cls):
                wdg.setFocusPolicy(Qt.StrongFocus)
                wdg.installEventFilter(self._nowheel)
        self._load_settings()
        self.guided._load()
        self.on_preset(self.preset.currentText())
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(125)

    # ------------------------------------------------------------------ panels
    def _link_box(self, sim_default: bool) -> QGroupBox:
        b = QGroupBox("1. Radar link")
        g = QGridLayout(b)
        self.link = QComboBox()
        self.link.addItems(LINKS)
        if sim_default:
            self.link.setCurrentIndex(2)
        self.link.currentIndexChanged.connect(self._link_changed)
        self.ctrl = PortCombo()
        self.ctrl.setToolTip("SPI mode: the board's only COM port. UART mode: the higher COM number. "
                             "The list is re-scanned every time you open it.")
        self.data = PortCombo()
        self.data.setToolTip("UART mode only: the lower COM number (radar data).")
        refresh = QPushButton("Check link")
        refresh.setToolTip("SPI: a short step-by-step SPI check on the chosen port (bridge, radar config, data-ready "
                           "line, frames). Writes runs/SPIPROBE_*.md. Stop the stream first.")
        refresh.clicked.connect(self.check_link)
        self.btn_check = refresh
        self.btn_stream = QPushButton("Start  (F5)")
        self.btn_stream.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_stream.clicked.connect(self.toggle_stream)
        self.state = QLabel("idle")
        self.state.setWordWrap(True)
        self.state.setTextFormat(Qt.RichText)
        g.addWidget(QLabel("link"), 0, 0)
        g.addWidget(self.link, 0, 1, 1, 2)
        g.addWidget(QLabel("control port"), 1, 0)
        g.addWidget(self.ctrl, 1, 1)
        g.addWidget(refresh, 1, 2)
        self.data_lbl = QLabel("data port")
        g.addWidget(self.data_lbl, 2, 0)
        g.addWidget(self.data, 2, 1, 1, 2)
        g.addWidget(self.btn_stream, 3, 0, 1, 3)
        g.addWidget(self.state, 4, 0, 1, 3)
        self._fill_ports()
        self._link_changed()
        return b

    def _config_box(self) -> QGroupBox:
        b = QGroupBox("2. Radar config")
        v = QVBoxLayout(b)
        h = QHBoxLayout()
        self.preset = QComboBox()
        names = list(presets())
        self.preset.addItems(list(CONFIGS) + [n for n in names if n not in CONFIGS])
        self.preset.currentTextChanged.connect(self.on_preset)
        h.addWidget(QLabel("preset"))
        h.addWidget(self.preset, 1)
        v.addLayout(h)
        self.show_all = QCheckBox("show / edit all parameters")
        self.form = ConfigForm(self._form_changed)
        self.form_scroll = QScrollArea()
        self.form_scroll.setWidgetResizable(True)
        self.form_scroll.setWidget(self.form)
        self.form_scroll.setFixedHeight(260)
        self.form_scroll.setVisible(False)
        self.show_all.toggled.connect(self.form_scroll.setVisible)
        v.addWidget(self.show_all)
        v.addWidget(self.form_scroll)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFixedHeight(150)
        self.summary.setStyleSheet("font-family:Consolas,monospace; font-size:9pt;")
        v.addWidget(self.summary)
        self.btn_apply = QPushButton("Apply config (restarts the stream)")
        self.btn_apply.clicked.connect(self.apply_config)
        v.addWidget(self.btn_apply)
        return b

    def _record_box(self) -> QGroupBox:
        b = QGroupBox("3. Record")
        g = QGridLayout(b)
        self.env = QComboBox()
        self.env.addItems(["closed", "open"])
        self.env.currentTextChanged.connect(self._fill_steps)
        self.step = QComboBox()
        self.step.currentIndexChanged.connect(self._step_changed)
        prev, nxt = QPushButton("<"), QPushButton(">")
        prev.setFixedWidth(28)
        nxt.setFixedWidth(28)
        prev.clicked.connect(lambda: self.step.setCurrentIndex(max(0, self.step.currentIndex() - 1)))
        nxt.clicked.connect(lambda: self.step.setCurrentIndex(min(self.step.count() - 1, self.step.currentIndex() + 1)))
        self.step_text = QLabel("")
        self.step_text.setWordWrap(True)
        self.step_text.setStyleSheet("color:#ffd54f;")
        self.label = QLineEdit()
        self.label.setPlaceholderText("take label (free text; the step id is used if empty)")
        self.secs = QDoubleSpinBox()
        self.secs.setRange(0, 600)
        self.secs.setDecimals(0)
        self.secs.setValue(5)
        self.secs.setSuffix(" s")
        self.secs.setToolTip("Length of each recording. 0 = manual stop (F9).")
        self.btn_rec = QPushButton("● Record  (F9)")
        self.btn_rec.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_rec.clicked.connect(self.toggle_record)
        self.btn_step = QPushButton("▶ Record step (all TCC configs)")
        self.btn_step.setStyleSheet("font-weight:bold; padding:6px;")
        self.btn_step.clicked.connect(self.record_step)
        self.btn_step.setToolTip("Records the selected protocol step with tcc-1d-40mm and the '2D for steps' config "
                                 "back to back.")
        self.step2d = QComboBox()
        self.step2d.addItems([n for n in presets() if n.startswith("tcc-2d")])
        self.step2d.setToolTip("2D config used by 'Record step': pick the one with the fewest dropped frames.")
        self.marker = QLineEdit()
        self.marker.setPlaceholderText("marker text (Ctrl+M)")
        bm = QPushButton("marker")
        bm.clicked.connect(self.add_marker)
        self.rec_status = QLabel("")
        self.rec_status.setWordWrap(True)
        g.addWidget(QLabel("place"), 0, 0)
        g.addWidget(self.env, 0, 1, 1, 3)
        g.addWidget(QLabel("step"), 1, 0)
        g.addWidget(prev, 1, 1)
        g.addWidget(self.step, 1, 2)
        g.addWidget(nxt, 1, 3)
        g.addWidget(self.step_text, 2, 0, 1, 4)
        g.addWidget(QLabel("label"), 3, 0)
        g.addWidget(self.label, 3, 1, 1, 3)
        g.addWidget(QLabel("length"), 4, 0)
        g.addWidget(self.secs, 4, 1, 1, 3)
        g.addWidget(QLabel("2D for steps"), 5, 0)
        g.addWidget(self.step2d, 5, 1, 1, 3)
        g.addWidget(self.btn_rec, 6, 0, 1, 2)
        g.addWidget(self.btn_step, 6, 2, 1, 2)
        g.addWidget(self.marker, 7, 0, 1, 3)
        g.addWidget(bm, 7, 3)
        g.addWidget(self.rec_status, 8, 0, 1, 4)
        g.setColumnStretch(2, 1)
        self._fill_steps()
        return b

    def _session_box(self) -> QGroupBox:
        b = QGroupBox("Rig (once per rig) + session measurements (saved with every recording)")
        fl = QFormLayout(b)
        rig = load_rig(self.runs_dir)
        prev = {**last_session_meta(self.runs_dir), **rig}           # the locked rig wins over the last session
        self.rig_lock = QCheckBox("Rig fixed: lock the rig values (unlock only if you rebuild the rig)")
        self.rig_lock.setToolTip("Locked values are kept in runs/RIG.json and loaded every time the recorder starts.")
        fl.addRow(self.rig_lock)
        # --- mounting: radar rotation (where the module's pin-1 dot is, seen from the front) and phone orientation
        self.rot_combo = QComboBox()
        for deg, corner in orientation.ROTATIONS.items():
            self.rot_combo.addItem(f"{corner} ({deg}°)", deg)
        rot_btn = QPushButton("↻ 90°")
        rot_btn.setFixedWidth(52)
        rot_btn.setToolTip("Rotate the radar by 90° clockwise (seen from the front).")
        rot_btn.clicked.connect(lambda: self.rot_combo.setCurrentIndex((self.rot_combo.currentIndex() + 1) % 4))
        self._rot_btn = rot_btn
        rh = QHBoxLayout()
        rh.addWidget(self.rot_combo, 1)
        rh.addWidget(rot_btn)
        fl.addRow("radar: pin-1 dot (front view)", rh)
        self.phone_combo = QComboBox()
        self.phone_combo.addItems(["landscape", "portrait"])
        ph_btn = QPushButton("↻ 90°")
        ph_btn.setFixedWidth(52)
        ph_btn.setToolTip("Turn the phone by 90° (landscape <-> portrait).")
        ph_btn.clicked.connect(lambda: self.phone_combo.setCurrentIndex(1 - self.phone_combo.currentIndex()))
        self._ph_btn = ph_btn
        phh = QHBoxLayout()
        phh.addWidget(self.phone_combo, 1)
        phh.addWidget(ph_btn)
        fl.addRow("phone", phh)
        self.ant_diag = AntennaDiagram()
        fl.addRow(self.ant_diag)
        self.off_diag = OffsetDiagram()
        fl.addRow(self.off_diag)
        note = QLabel("Offsets: from the centre of the radar chip to the centre of the phone lens. "
                      "Negative values are fine (e.g. camera to the left: dx < 0).")
        note.setWordWrap(True)
        fl.addRow(note)
        try:
            rot0 = int(prev.get("radar_rotation_deg", self.settings.value("radar_rotation") or 180)) % 360
        except (TypeError, ValueError):
            rot0 = 0
        self.rot_combo.setCurrentIndex(list(orientation.ROTATIONS).index(rot0 if rot0 in orientation.ROTATIONS else 180))
        ph0 = str(prev.get("phone_orientation", self.settings.value("phone") or "portrait"))
        self.phone_combo.setCurrentText(ph0 if ph0 in ("landscape", "portrait") else "portrait")
        self.rot_combo.currentIndexChanged.connect(self._mount_changed)
        self.phone_combo.currentIndexChanged.connect(self._mount_changed)
        self.meta_edits: dict[str, QLineEdit] = {}
        for key, label, default in SESSION_FIELDS:
            if key in ("radar_rotation_deg", "phone_orientation"):
                continue
            e = QLineEdit(str(prev.get(key, default)))
            if key == "radar_orientation":
                e.setReadOnly(True)
                e.setStyleSheet("color:#555;")
            e.setToolTip(label)
            e.editingFinished.connect(self._meta_edited)
            fl.addRow(FIELD_LABELS.get(key, key), e)
            self.meta_edits[key] = e
        self.session_lbl = QLabel("not saved yet (saved at the first recording)")
        fl.addRow(self.session_lbl)
        self._apply_mount()
        self.rig_lock.setChecked(bool(rig))
        self._lock_rig(bool(rig))
        self.rig_lock.toggled.connect(self._rig_lock_toggled)
        return b

    def _lock_rig(self, on: bool):
        for wdg in (self.rot_combo, self.phone_combo, self._rot_btn, self._ph_btn):
            wdg.setEnabled(not on)
        for k in RIG_KEYS:
            if k in self.meta_edits:
                self.meta_edits[k].setReadOnly(on)
                self.meta_edits[k].setStyleSheet("color:#666;" if on else "")

    def _rig_lock_toggled(self, on: bool):
        self._lock_rig(on)
        if on:
            p = save_rig(self.runs_dir, self._meta())
            self.session_lbl.setText(f"rig locked: {p.name}")
        if hasattr(self, "guided"):
            self.guided._refresh()

    def _rotation(self) -> int:
        return int(self.rot_combo.currentData())

    def _apply_mount(self):
        rot, phone = self._rotation(), self.phone_combo.currentText()
        self.pairs = orientation.pairs(rot)
        self.ant_diag.set_rotation(rot)
        self.off_diag.set_orientation(rot, phone)
        self.meta_edits["radar_orientation"].setText(orientation.describe(rot))
        self.settings.setValue("radar_rotation", rot)
        self.settings.setValue("phone", phone)

    def _mount_changed(self, *_):
        self._apply_mount()
        self._meta_edited()
        if hasattr(self, "guided"):
            self.guided._refresh()

    def _plots(self) -> QWidget:
        pg.setConfigOptions(antialias=False, background="#151515")
        w = pg.GraphicsLayoutWidget()
        self.p_prof = w.addPlot(row=0, col=0, colspan=2, title="Range profile (dBFS, Doppler/intervals averaged)")
        self.p_prof.showGrid(x=True, y=True, alpha=0.25)
        self.p_prof.setLabel("bottom", "range", "m")
        self.p_prof.addLegend(offset=(-10, 10))
        self.curves = [self.p_prof.plot(pen=pg.mkPen(c, width=1.5), name=f"RX{i}") for i, c in enumerate(RX_COLORS)]
        self.floor_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#888", style=Qt.DashLine))
        self.peak_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff5252", width=1))
        self.p_prof.addItem(self.floor_line)
        self.p_prof.addItem(self.peak_line)

        self.p_wf = w.addPlot(row=1, col=0, title="Range over time (newest at the top)")
        self.p_wf.setLabel("bottom", "range", "m")
        self.p_wf.setLabel("left", "updates ago")
        self.wf_img = pg.ImageItem()
        self.wf_img.setColorMap(pg.colormap.get("viridis"))
        self.p_wf.addItem(self.wf_img)
        self.p_wf.invertY(True)

        self.p_rd = w.addPlot(row=1, col=1, title="Range-Doppler (2D configs)")
        self.p_rd.setLabel("bottom", "range", "m")
        self.p_rd.setLabel("left", "velocity", "m/s")
        self.rd_img = pg.ImageItem()
        self.rd_img.setColorMap(pg.colormap.get("inferno"))
        self.p_rd.addItem(self.rd_img)
        self.ang_h = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(H_COL.name()), pen=None, name="horizontal pair")
        self.ang_v = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(V_COL.name()), pen=None, name="vertical pair")
        self.p_rd.addItem(self.ang_h)
        self.p_rd.addItem(self.ang_v)
        w.ci.layout.setRowStretchFactor(0, 2)
        w.ci.layout.setRowStretchFactor(1, 3)
        return w

    # ------------------------------------------------------------------ settings
    def _load_settings(self):
        s = self.settings
        if s.value("link") is not None and self.link.currentIndex() != 2:
            self.link.setCurrentIndex(int(s.value("link")))
        for combo, key in ((self.ctrl, "ctrl"), (self.data, "data")):
            v = s.value(key)
            if v:
                combo.setCurrentText(str(v))
        if s.value("preset") in presets():
            self.preset.setCurrentText(str(s.value("preset")))
        if s.value("env"):
            self.env.setCurrentText(str(s.value("env")))
        if s.value("step2d") in presets():
            self.step2d.setCurrentText(str(s.value("step2d")))

    def _save_settings(self):
        s = self.settings
        s.setValue("link", self.link.currentIndex())
        s.setValue("ctrl", self.ctrl.currentText())
        s.setValue("data", self.data.currentText())
        s.setValue("preset", self.preset.currentText())
        s.setValue("env", self.env.currentText())
        s.setValue("step2d", self.step2d.currentText())

    # ------------------------------------------------------------------ link / config
    def _fill_ports(self):
        self.ctrl.refresh()
        self.data.refresh()

    def check_link(self):
        """Run the SPI probe on the chosen port in the background; the result shows under the Start button."""
        if self.eng.session is not None or self.seq is not None:
            QMessageBox.information(self, "Stop first", "Stop the stream before checking the link.")
            return
        if getattr(self, "_probe", None) is not None:
            return
        kind = self._link_kind()
        if kind == "uart":
            QMessageBox.information(self, "UART", "For the UART link just press Start: the frame rate and dropped "
                                    "frames above the plots tell whether it works.")
            return
        ctrl = _port_name(self.ctrl.currentText())
        if kind == "spi" and not ctrl:
            QMessageBox.warning(self, "No port", "Pick the control port first.")
            return
        import types

        from ..cli_spi import spi_probe
        a = types.SimpleNamespace(ctrl=ctrl, preset="tcc-1d-40mm", spi_clock=30e6, spi_dev=0, ch347_dll=None,
                                  int_bit="auto", gpio_input=None, seconds=3.0, watch=1.0, sim=(kind == "sim"))
        self._probe = {"done": False, "rc": None}

        def run():
            try:
                self._probe["rc"] = spi_probe(a)
            except Exception as e:  # pragma: no cover
                self._probe["rc"] = repr(e)
            self._probe["done"] = True

        self.btn_check.setEnabled(False)
        self.btn_stream.setEnabled(False)
        self.state.setText("checking the SPI link (~10 s) …")
        threading.Thread(target=run, daemon=True).start()

    def _probe_tick(self):
        pr = getattr(self, "_probe", None)
        if pr is None or not pr["done"]:
            return
        self._probe = None
        self.btn_check.setEnabled(True)
        self.btn_stream.setEnabled(True)
        reps = sorted(self.runs_dir.glob("SPIPROBE_*.md"))
        line = ""
        if reps:
            for ln in reps[-1].read_text(encoding="utf-8").splitlines():
                if ln.startswith("**Result:"):
                    line = ln.strip("*").replace("Result: ", "")
                    break
        ok = pr["rc"] == 0
        col = "#2e7d32" if ok else "#c62828"
        msg = (f"<span style='color:{col}'><b>SPI link {'OK' if ok else 'NOT OK'}</b></span> — {line or pr['rc']}"
               + ("" if ok else "<br>Check: MODE SEL ON, DIP 1 2 5 6 7 9 on, CH341PAR driver, the right port; "
                  "unplug/replug the USB; driver: drivers/wch/CH341PAR.EXE."))
        self.state.setText(msg)

    def _link_changed(self, *_):
        uart = self.link.currentIndex() == 1
        sim = self.link.currentIndex() == 2
        self.data.setVisible(uart)
        self.data_lbl.setVisible(uart)
        self.ctrl.setEnabled(not sim)

    def on_preset(self, name: str):
        if name not in presets():
            return
        self.cfg = presets()[name]
        self.form.set_config(self.cfg)
        self._show_summary()

    def _form_changed(self):
        cfg, errors = self.form.get_config()
        if errors:
            self.summary.setPlainText("Parameter errors:\n" + "\n".join(errors))
            return
        self.cfg = cfg
        self._show_summary()

    def _show_summary(self):
        lines = [self.cfg.name + (f": {self.cfg.notes}" if self.cfg.notes else "")]
        try:
            lines += self.cfg.derive().summary_lines()
        except Exception as e:  # pragma: no cover
            lines.append(f"derive failed: {e}")
        issues = self.cfg.validate()
        lines += [str(i) for i in issues if i.level != "info"] or ["no validation warnings"]
        self.summary.setPlainText("\n".join(lines))

    def _link_kind(self) -> str:
        return ("spi", "uart", "sim")[self.link.currentIndex()]

    def start_stream(self, cfg: RadarConfig | None = None, label: str = "") -> bool:
        cfg = cfg or self.cfg
        kind = self._link_kind()
        if kind == "uart" and cfg.link == "spi":
            QMessageBox.warning(self, "SPI-only config", f"{cfg.name} needs the SPI link (cube too big for the UART).")
            return False
        if has_errors(cfg.validate()):
            QMessageBox.warning(self, "Config not valid", "\n".join(str(i) for i in cfg.validate() if i.level == "error"))
            return False
        self._wf = None
        self._peak = None
        self._last_seq_seen = -1
        self._bytes_prev = None
        try:
            if kind == "sim":
                self.eng.start_sim(cfg, record=False)
            elif kind == "spi":
                ctrl = _port_name(self.ctrl.currentText())
                if not ctrl:
                    QMessageBox.warning(self, "No port", "Pick the control port (the single COM port in SPI mode).")
                    return False
                self.eng.start_spi(cfg, ctrl, record=False, label=label)
            else:
                ctrl, data = _port_name(self.ctrl.currentText()), _port_name(self.data.currentText())
                if not (ctrl and data):
                    QMessageBox.warning(self, "No port", "Pick both ports (control = higher COM number).")
                    return False
                self.eng.start_serial(cfg.copy(link="uart"), ctrl, data, record=False, label=label)
        except Exception as e:
            log.exception("start failed")
            QMessageBox.critical(self, "Start failed", repr(e))
            return False
        self._save_settings()
        self.btn_stream.setText("Stop  (F5)")
        return True

    def stop_stream(self):
        if self.rec is not None:
            self.stop_record()
        self.eng.stop("stopped in the recorder")
        self.btn_stream.setText("Start  (F5)")
        self.state.setText("stopped")

    def toggle_stream(self):
        if self.seq is not None or self.guided.busy:
            return
        if self.eng.session is None:
            self.start_stream()
        else:
            self.stop_stream()

    def apply_config(self):
        if self.seq is not None or self.guided.busy:
            return
        self._form_changed()
        if self.eng.session is not None:
            self.stop_stream()
            self.start_stream()

    # ------------------------------------------------------------------ protocol steps
    def _fill_steps(self, *_):
        self.steps: list[Step] = tcc_plan(self.env.currentText(), t3=True, bonus=True, walk=True)
        self.step.blockSignals(True)
        self.step.clear()
        self.step.addItem("(none: free recording)")
        self.step.addItems([s.sid for s in self.steps])
        self.step.blockSignals(False)
        self._step_changed()

    def _current_step(self) -> Step | None:
        i = self.step.currentIndex()
        return self.steps[i - 1] if i >= 1 else None

    def _step_changed(self, *_):
        st = self._current_step()
        if st is None:
            self.step_text.setText("Free recording: the current config, with the label below.")
            return
        extra = f"  ({st.reps} reps per config)" if st.reps > 1 else ""
        self.step_text.setText(f"{st.text}{extra}")
        self.secs.setValue(st.seconds)

    # ------------------------------------------------------------------ session
    def _meta(self) -> dict:
        m = {k: e.text().strip() for k, e in self.meta_edits.items()}
        m["radar_rotation_deg"] = str(self._rotation())
        m["phone_orientation"] = self.phone_combo.currentText()
        h, v = self.pairs["horizontal"], self.pairs["vertical"]
        m["rx_pair_horizontal"], m["rx_pair_vertical"] = f"{h[0]}-{h[1]}", f"{v[0]}-{v[1]}"
        return m

    def _meta_edited(self):
        self.session_id = ""          # a changed measurement starts a new session file at the next recording
        self.session_lbl.setText("changed: saved at the next recording")
        if hasattr(self, "guided"):
            self.guided._refresh()

    def _ensure_session(self):
        if not self.session_id:
            self.session_id = save_session_meta(self.runs_dir, self.env.currentText(), self._meta())
            self.session_lbl.setText(f"saved: TCC_SESSION_{self.session_id}.json")

    # ------------------------------------------------------------------ recording
    def toggle_record(self):
        if self.guided.busy or (self.tabs.currentIndex() == 0 and self.rec is None and self.seq is None):
            self.guided.f9()                  # Simple tab: F9 = start the take / next config
            return
        if self.seq is not None:
            return
        if self.rec is None:
            self.start_record()
        else:
            self.stop_record()

    def start_record(self, step: Step | None = None, rep: int = 1, label: str | None = None, env: str | None = None,
                     text: str | None = None) -> bool:
        ses = self.eng.session
        if ses is None:
            if not self.start_stream():
                return False
            ses = self.eng.session
        self._ensure_session()
        st = step if step is not None else self._current_step()
        label = label or self.label.text().strip() or (st.sid if st else "free")
        xy = st.xy() if st else None
        meta = {"tcc_step": st.sid if st else label, "label": label, "test": st.test if st else "manual",
                "env": env or self.env.currentText(), "target": st.target if st else "", "r_m": st.r_m if st else None,
                "theta_deg": st.theta_deg if st else None, "x_m": xy[0] if xy else None, "y_m": xy[1] if xy else None,
                "rep": rep, "config": self.eng.cfg.name if self.eng.cfg else "", "session": self.session_id,
                "session_meta": self._meta(), **(st.extra if st else {}),
                "text": text if text is not None else (st.text if st else self.label.text().strip())}
        self.eng.note("tcc " + json.dumps(meta), kind="tcc_step")
        ses.start_recording()
        self.eng.note("tcc start beep", kind="tcc_beep")
        tb0 = time.time()
        beep(1)
        ss = (ses.source_stats or {})
        self.rec = {"meta": meta, "step": st, "rep": rep, "t0": tb0, "t0m": time.monotonic(), "frames0": ses.frames,
                    "gaps0": ss.get("index_gaps", 0) or 0, "session": ses}
        self.btn_rec.setText("■ Stop recording  (F9)")
        self.btn_rec.setStyleSheet("font-weight:bold; padding:6px; background:#b71c1c; color:white;")
        return True

    def stop_record(self) -> dict | None:
        if self.rec is None:
            return None
        r, self.rec = self.rec, None
        ses = r["session"]
        tb1 = time.time()
        beep(2)
        ses.stop_recording()
        self.eng.note("tcc end beep", kind="tcc_beep")
        dur = max(1e-3, time.monotonic() - r["t0m"])
        n = ses.frames - r["frames0"]
        g = max(0, ((ses.source_stats or {}).get("index_gaps", 0) or 0) - r["gaps0"])
        loss = 100.0 * g / max(1, n + g)
        status = "ok" if n > 0 and loss <= 15.0 else ("no_frames" if n == 0 else "high_loss")
        m, st = r["meta"], r["step"]
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "session": self.session_id, "env": m["env"],
               "step": m["label"], "test": m["test"], "target": m["target"], "r_m": m["r_m"],
               "theta_deg": m["theta_deg"], "x_m": m["x_m"] if m["x_m"] is not None else "",
               "y_m": m["y_m"] if m["y_m"] is not None else "", "config": m["config"], "rep": r["rep"],
               "seconds": round(dur, 1), "frames": n, "fps": round(n / dur, 1), "index_gaps": g,
               "loss_pct": round(loss, 1), "status": status, "t_beep_start": round(r["t0"], 3),
               "t_beep_end": round(tb1, 3), "run_dir": str(ses.dir),
               "text": m.get("text", "")}
        append_take_row(self.runs_dir / "TCC_TAKES.csv", row)
        self.btn_rec.setText("● Record  (F9)")
        self.btn_rec.setStyleSheet("font-weight:bold; padding:6px;")
        col = "#81c784" if status == "ok" else "#ff5252"
        self.rec_status.setText(f"<span style='color:{col}'>{m['label']} / {m['config']}: {status}, {n} frames, "
                                f"{row['fps']} fps, loss {row['loss_pct']} %</span>")
        return row

    def add_marker(self):
        txt = self.marker.text().strip() or "marker"
        if self.eng.session is not None:
            self.eng.note(txt, kind="marker")
            self.rec_status.setText(f"marker: {txt}")
        self.marker.clear()

    # ------------------------------------------------------------------ "record step": all TCC configs in a row
    def record_step(self):
        if self.guided.busy:
            return
        st = self._current_step()
        if st is None:
            QMessageBox.information(self, "No step", "Pick a protocol step first (or use ● Record for a free recording).")
            return
        if self.seq is not None:          # second click = abort
            self._seq_finish("aborted")
            return
        if self.rec is not None:
            self.stop_record()
        queue = []
        for rep in range(1, st.reps + 1):
            cfgs = (CONFIGS[0], self.step2d.currentText() or CONFIGS[1])
            order = cfgs if rep % 2 == 1 else tuple(reversed(cfgs))           # ABBA across repetitions
            queue += [(c, rep) for c in order]
        self.seq = {"step": st, "queue": queue, "phase": "next", "rows": [], "t": 0.0}
        self.btn_step.setText("■ Abort step")
        for wdg in (self.btn_stream, self.btn_rec, self.btn_apply, self.step, self.preset):
            wdg.setEnabled(False)

    def _seq_tick(self):
        s = self.seq
        now = time.monotonic()
        st: Step = s["step"]
        if s["phase"] == "next":
            if not s["queue"]:
                bad = [r for r in s["rows"] if r["status"] != "ok"]
                self._seq_finish("done" if not bad else f"{len(bad)} segment(s) failed: repeat the step")
                if not bad and self.step.currentIndex() < self.step.count() - 1:
                    self.step.setCurrentIndex(self.step.currentIndex() + 1)
                return
            s["cfg"], s["rep"] = s["queue"].pop(0)
            s["phase"], s["t"] = ("countdown" if st.moving else "start"), now
        if s["phase"] == "countdown":
            left = 10 - (now - s["t"])
            self.rec_status.setText(f"<b>{s['cfg']} rep {s['rep']}</b>: go to the start position; BEEP in "
                                    f"~{max(0, left):.0f} s + radar start-up")
            if left <= 0:
                s["phase"] = "start"
        if s["phase"] == "start":
            label = f"tcc_{self.env.currentText()}_{st.sid}_{s['cfg']}" + (f"_r{s['rep']}" if st.reps > 1 else "")
            if not self.start_stream(presets()[s["cfg"]], label=label):
                self._seq_finish("could not start the stream")
                return
            s["phase"], s["t"] = "wait_first", now
            self.rec_status.setText(f"{s['cfg']}: starting the radar …")
        elif s["phase"] == "wait_first":
            ses = self.eng.session
            if ses is not None and ses.frames > 0:
                self.start_record(step=st, rep=s["rep"])
                s["phase"], s["t"] = "recording", now
            elif now - s["t"] > 15 or self.eng.state == "error":
                row = {"status": "no_frames"}
                s["rows"].append(row)
                self.stop_stream()
                self.rec_status.setText("<span style='color:#ff5252'>no frames: unplug/replug the radar USB, then "
                                        "repeat the step</span>")
                s["phase"] = "next"
        elif s["phase"] == "recording":
            self.rec_status.setText(f"<b>recording {s['cfg']}</b> rep {s['rep']}: {now - s['t']:.0f} / "
                                    f"{self.secs.value():.0f} s")
            if now - s["t"] >= self.secs.value():
                s["rows"].append(self.stop_record() or {"status": "?"})
                self.stop_stream()
                s["phase"] = "next"

    def _seq_finish(self, msg: str):
        if self.rec is not None:
            self.stop_record()
        if self.eng.session is not None:
            self.stop_stream()
        rows = self.seq["rows"] if self.seq else []
        self.seq = None
        self.btn_step.setText("▶ Record step (all TCC configs)")
        for wdg in (self.btn_stream, self.btn_rec, self.btn_apply, self.step, self.preset):
            wdg.setEnabled(True)
        summ = ", ".join(f"{r.get('config', '?')}: {r.get('status')} ({r.get('loss_pct', '-')} % loss)" for r in rows)
        col = "#81c784" if msg == "done" else "#ff5252"
        self.rec_status.setText(f"<span style='color:{col}'><b>step {msg}</b></span><br>{summ}")

    # ------------------------------------------------------------------ periodic update
    def tick(self):
        self._probe_tick()
        try:
            self.guided.tick()
        except Exception as e:  # pragma: no cover
            log.exception("guided take failed")
            self.guided.finish(f"error {e!r}")
        if self.seq is not None:
            try:
                self._seq_tick()
            except Exception as e:  # pragma: no cover
                log.exception("step sequence failed")
                self._seq_finish(f"error {e!r}")
        if self.rec is not None and self.seq is None and not self.guided.busy and self.secs.value() > 0 and \
                time.monotonic() - self.rec["t0m"] >= self.secs.value():
            self.stop_record()
        self._update_stats()
        fr = self.eng.latest_frame
        if fr is None or self.eng.session is None or fr.seq == self._last_seq_seen:
            return
        self._last_seq_seen = fr.seq
        try:
            self._update_plots(fr)
        except Exception as e:  # pragma: no cover
            log.warning("plot update failed: %s", e)

    def _update_plots(self, fr):
        cfg = self.eng.cfg or self.cfg
        v = frame_views(fr.cube, cfg, getattr(self, "pairs", None))
        r = v["r"]
        for i, c in enumerate(self.curves):
            if i < v["prof_rx"].shape[0]:
                c.setData(r, v["prof_rx"][i])
            else:
                c.setData([], [])
        self.floor_line.setValue(v["floor"])
        self.peak_line.setValue(v["peak_r"])
        near = r < MIN_RANGE_M                       # TX leakage: hidden in the images so it does not set the colours
        row = v["tot_db"].astype(np.float32)
        row[near] = v["floor"]
        if self._wf is None or self._wf.shape[1] != row.size:
            self._wf = np.full((WATERFALL_ROWS, row.size), v["floor"], dtype=np.float32)
        self._wf = np.roll(self._wf, 1, axis=0)
        self._wf[0] = row
        lo, hi = np.percentile(self._wf, [5, 99.7])
        self.wf_img.setImage(self._wf.T, autoLevels=False, levels=(lo, max(hi, lo + 1)))
        dx = r[1] - r[0] if r.size > 1 else 1.0
        self.wf_img.setRect(0, 0, dx * row.size, WATERFALL_ROWS)
        if v["rd"] is not None:
            self.ang_h.setData([], [])
            self.ang_v.setData([], [])
            self.rd_img.setVisible(True)
            rd = v["rd"].copy()
            rd[:, near] = np.median(rd)
            lo, hi = np.percentile(rd, [20, 99.9])
            self.rd_img.setImage(rd.T, autoLevels=False, levels=(lo, max(hi, lo + 1)))
            va = v["v_axis"]
            dv = va[1] - va[0] if va.size > 1 else 1.0
            self.rd_img.setRect(0, va[0] - dv / 2, dx * rd.shape[1], dv * rd.shape[0])
            self.p_rd.setYRange(float(va[0] - dv / 2), float(va[0] - dv / 2 + dv * rd.shape[0]), padding=0)
            self.p_rd.setXRange(0, float(dx * rd.shape[1]), padding=0)
            self.p_rd.setTitle("Range-Doppler (dBFS)")
            self.p_rd.setLabel("left", "velocity", "m/s")
        else:
            self.rd_img.setVisible(False)
            sel = (v["tot_db"] > v["floor"] + 12) & (r >= MIN_RANGE_M)
            for item, key in ((self.ang_h, "angle_h"), (self.ang_v, "angle_v")):
                item.setData(r[sel], v[key][sel]) if v[key] is not None else item.setData([], [])
            h, vv = self.pairs["horizontal"], self.pairs["vertical"]
            self.p_rd.setTitle(f"Angle of strong returns (uncalibrated): <span style='color:{H_COL.name()}'>"
                               f"{h[0]}-{h[1]} horizontal</span>, <span style='color:{V_COL.name()}'>"
                               f"{vv[0]}-{vv[1]} vertical</span>")
            self.p_rd.setLabel("left", "angle", "deg")
            self.p_rd.setYRange(-90, 90)
            self.p_rd.setXRange(0, float(r[-1]) if r.size else 1.0)
        self._peak = v

    def _update_stats(self):
        src, ses, eng = self.eng.source, self.eng.session, self.eng
        if ses is None:
            self.stats.setText("<b>idle</b>: choose the link and the port, check the link if you like, pick the "
                               "config, then Start (F5).")
            return
        cfg = eng.cfg or self.cfg
        d = cfg.derive()
        ss = src.stats.to_dict() if src is not None else {}
        ps = src.parser.stats.to_dict() if (src is not None and getattr(src, "parser", None) is not None) else {}
        now = time.monotonic()
        if src is not None and hasattr(src, "spi_stats") and now - self._spi_t > 1.0:
            try:
                self._spi_cache = src.spi_stats()
            except Exception:
                self._spi_cache = {}
            self._spi_t = now
        frames, gaps = ss.get("frames", 0), ss.get("index_gaps", 0)
        loss = 100.0 * gaps / max(1, frames + gaps)
        fps = ss.get("fps_recent", 0.0)
        b_in = ps.get("bytes_in", 0)
        kbs = 0.0
        if self._bytes_prev is not None and now > self._bytes_prev[0]:
            kbs = (b_in - self._bytes_prev[1]) / (now - self._bytes_prev[0]) / 1e3
        if self._bytes_prev is None or now - self._bytes_prev[0] > 1.0:
            self._bytes_prev = (now, b_in)
            self._kbs = kbs
        kbs = getattr(self, "_kbs", 0.0)
        err = sum((ps.get("errors") or {}).values())
        rep = ses.config_report or {}
        failed, mism = rep.get("failed", []), rep.get("check_mismatches", [])
        st = eng.state + (f": {eng.state_info.get('error')}" if eng.state_info.get("error") else "")
        self.state.setText(f"{st} — run {ses.dir.name}")

        def c(ok: bool, txt: str) -> str:
            return f"<span style='color:{'#81c784' if ok else '#ff5252'}'>{txt}</span>"

        fps_ok = fps >= 0.8 * d.expected_fps or frames < 20
        lines = [
            f"<b>{cfg.name}</b> &nbsp; state {c(eng.state == 'streaming', eng.state)} &nbsp; "
            f"fps {c(fps_ok, f'{fps:5.1f}')} / {d.expected_fps:.1f} expected &nbsp; frames {frames} &nbsp; "
            f"{kbs:6.1f} kB/s",
            f"dropped frames (radar index gaps) {c(loss <= BAD_LOSS_PCT, f'{gaps} ({loss:.1f} %)')} &nbsp; "
            f"incomplete {c(ps.get('frames_incomplete', 0) == 0, str(ps.get('frames_incomplete', 0)))} &nbsp; "
            f"parse errors {c(err == 0, str(err))} &nbsp; resyncs {ps.get('resyncs', 0)} &nbsp; "
            f"shape {ss.get('shape_used') or '-'}" + (c(False, f" MISMATCH {ss.get('shape_mismatch_frames')}")
                                                      if ss.get("shape_mismatch_frames") else ""),
            f"chip config: rejected {c(not failed, str(len(failed)))} &nbsp; read-back mismatches "
            f"{c(not mism, str(len(mism)))}" + (f" &nbsp; SPI resets {self._spi_cache.get('hif_resets', 0)}"
                                                if self._spi_cache else ""),
        ]
        pk = getattr(self, "_peak", None)
        if pk is not None:
            ang = ""
            if pk.get("peak_ang_h") is not None and pk.get("peak_ang_v") is not None:
                ang = (f", angle <span style='color:{H_COL.name()}'>horizontal {pk['peak_ang_h']:+.0f}°</span> "
                       f"<span style='color:{V_COL.name()}'>vertical {pk['peak_ang_v']:+.0f}°</span> (uncal.)")
            lines.append(f"strongest return {pk['peak_r']:.2f} m, {pk['peak_snr']:.0f} dB above the median floor{ang}")
        if self.rec is not None:
            rr = self.rec
            n = ses.frames - rr["frames0"]
            g = max(0, gaps - rr["gaps0"])
            lines.append(c(True, f"● REC {rr['meta']['label']} — {now - rr['t0m']:.0f} s, {n} frames, "
                                 f"{100.0 * g / max(1, n + g):.1f} % dropped"))
        self.stats.setText("<br>".join(lines))

    def closeEvent(self, ev):
        if self.guided.busy:
            self.guided.finish("aborted (window closed)")
        self.guided.save()
        if self.seq is not None:
            self._seq_finish("aborted (window closed)")
        self.stop_stream()
        self._save_settings()
        super().closeEvent(ev)


def run_recorder(argv=None, sim: bool = False) -> int:
    runs = default_runs_dir()
    setup_base(logging.WARNING, app_log_dir=runs)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    eng = Engine(runs_dir=runs, process=False)
    w = RecorderWindow(eng, runs, sim_default=sim)
    w.show()
    return app.exec()
