"""Radar lab GUI: live/sim/replay view, radar config editor with validation, processing
controls, user markers and a live log. Run with ``uv run ravidata gui``.
"""

from __future__ import annotations

import html
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from ..config import RadarConfig, has_errors, presets
from ..engine import Engine
from ..logsetup import get_logger, get_ring, setup_base
from ..processing.pipeline import Preprocess
from ..session import default_runs_dir
from ..sim import scene_presets
from .forms import ConfigForm, param_form, set_param_widget

log = get_logger("gui")

HELP = """<b>Plots</b><br>
<b>Range profile</b>: power per range bin (dB). Thin lines = each RX, thick = the combined detection
power (Processing → combine), dashed grey = tracked noise floor, dashed red = primary detector threshold,
dots = confirmed detections, vertical line = nearest obstacle. Shaded = cropped (ignored) ranges.<br>
<b>Waterfall</b>: combined power over time (newest at the top). Static obstacles you walk towards appear
as lines drifting to shorter range.<br>
<b>Range × height</b>: every bin above <i>show SNR</i> placed at the height implied by the RX-hi/RX-ref phase
(needs correct antenna roles, sign, phase calibration, sensor height and pitch). Bands: ground and the
head-height zone. Big markers = detections. In the simulator, crosses = true scatterers.<br>
<b>Range–Doppler / elevation</b>: range–Doppler image in 2D modes; otherwise elevation angle per bin.<br><br>
<b>Markers</b> (F1–F6) are written to the run log with the frame number. Use them: they are what lets
an agent line up your observations with the data.<br><br>
<b>Logs</b>: every Start creates <code>runs/&lt;time&gt;_&lt;source&gt;_&lt;preset&gt;/</code> with summary.log, verbose.log,
meta.json, events.jsonl, SESSION_REPORT.md and (if Record is on) cubes.h5. <code>runs/LATEST.txt</code> points to the
newest one."""

MARKERS = [("F1", "target placed"), ("F2", "approach start"), ("F3", "approach end"),
           ("F4", "MISSED obstacle"), ("F5", "FALSE alarm"), ("F6", "scene changed")]
# Distance markers for known-range holds (EXP-004 wall sweep). `ravidata compare` / `ravidata ab` score the
# frames from such a marker to the next marker (max 4 s) against the distance. "0" = moving / no hold.
# Ctrl+digit so typing numbers into parameter fields never creates markers.
DIST_MARKERS = [("Ctrl+1", "d=0.3 m"), ("Ctrl+2", "d=0.5 m"), ("Ctrl+3", "d=0.75 m"), ("Ctrl+4", "d=1.0 m"),
                ("Ctrl+5", "d=1.5 m"), ("Ctrl+6", "d=2.0 m"), ("Ctrl+7", "d=2.5 m"), ("Ctrl+8", "d=3.0 m"),
                ("Ctrl+0", "moving (no hold)")]
LEVEL_COLORS = {"DEBUG": "#888", "INFO": "#ddd", "WARNING": "#f0c040", "ERROR": "#ff6060", "CRITICAL": "#ff3030"}


class MainWindow(QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.settings = QSettings("RAVIDATA", "ravidata-lab")
        self.setWindowTitle("RAVIDATA lab — RS6130")
        self.resize(1700, 1000)
        self._last_counter = -1
        self._ring_pos = 0
        self._det_widgets: dict[str, dict] = {}
        self._pre_widgets: dict = {}
        self.cfg = presets()["coarse-1d-20cm"]
        pg.setConfigOptions(antialias=False, imageAxisOrder="row-major")
        self._build_plots()
        self._build_controls()
        self._build_log()
        self._load_settings()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)
        for key, text in MARKERS + DIST_MARKERS:
            QShortcut(QKeySequence(key), self, activated=lambda t=text: self._marker(t))
        QShortcut(QKeySequence("F7"), self, activated=self._take_start)
        QShortcut(QKeySequence("F8"), self, activated=self._take_end)

    # ================================================================== plots
    def _build_plots(self):
        central = QWidget()
        v = QVBoxLayout(central)
        self.banner = QLabel("idle")
        self.banner.setAlignment(Qt.AlignCenter)
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        self.banner.setFont(f)
        self.banner.setStyleSheet("background:#333;color:#ccc;padding:6px;border-radius:6px")
        v.addWidget(self.banner)
        self.det_line = QLabel("")
        self.det_line.setAlignment(Qt.AlignCenter)
        v.addWidget(self.det_line)
        grid = QGridLayout()
        v.addLayout(grid, 1)

        self.p_prof = pg.PlotWidget(title="Range profile (dB)")
        self.p_prof.setLabel("bottom", "range [m]")
        self.p_prof.showGrid(x=True, y=True, alpha=0.3)
        self.p_prof.addLegend(offset=(-10, 10))
        self.c_rx = [self.p_prof.plot(pen=pg.mkPen(c, width=1), name=f"RX{i}") for i, c in
                     enumerate(["#4c8bf5", "#f5a54c", "#58c96b"])]
        self.c_comb = self.p_prof.plot(pen=pg.mkPen("w", width=2.5), name="combined")
        self.c_floor = self.p_prof.plot(pen=pg.mkPen("#999", width=1, style=Qt.DashLine), name="floor")
        self.c_thr = self.p_prof.plot(pen=pg.mkPen("#ff5050", width=1.5, style=Qt.DashLine), name="threshold")
        self.s_det = pg.ScatterPlotItem(size=12, brush=pg.mkBrush("#ff3030"), pen=pg.mkPen("w"))
        self.p_prof.addItem(self.s_det)
        self.v_near = pg.InfiniteLine(angle=90, pen=pg.mkPen("#ff3030", width=2))
        self.p_prof.addItem(self.v_near)
        self.crop_near = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(80, 80, 80, 60))
        self.crop_far = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(80, 80, 80, 60))
        self.p_prof.addItem(self.crop_near)
        self.p_prof.addItem(self.crop_far)
        grid.addWidget(self.p_prof, 0, 0)

        self.p_wf = pg.PlotWidget(title="Waterfall: combined power over time (newest on top)")
        self.p_wf.setLabel("bottom", "range [m]")
        self.p_wf.setLabel("left", "frames ago")
        self.img_wf = pg.ImageItem()
        self.img_wf.setColorMap(pg.colormap.get("viridis"))
        self.p_wf.addItem(self.img_wf)
        grid.addWidget(self.p_wf, 0, 1)

        self.p_h = pg.PlotWidget(title="Range × height (from elevation phase)")
        self.p_h.setLabel("bottom", "range [m]")
        self.p_h.setLabel("left", "height [m]")
        self.p_h.showGrid(x=True, y=True, alpha=0.3)
        self.p_h.setYRange(-0.5, 2.8)
        self.band_ground = pg.LinearRegionItem(values=(0, 0.25), orientation="horizontal", movable=False,
                                               brush=pg.mkBrush(120, 90, 40, 50))
        self.band_head = pg.LinearRegionItem(values=(1.4, 2.0), orientation="horizontal", movable=False,
                                             brush=pg.mkBrush(200, 40, 40, 40))
        self.line_clear = pg.InfiniteLine(pos=2.1, angle=0, pen=pg.mkPen("#aaa", style=Qt.DotLine))
        self.line_sensor = pg.InfiniteLine(pos=0.9, angle=0, pen=pg.mkPen("#4c8bf5", style=Qt.DotLine))
        for it in (self.band_ground, self.band_head, self.line_clear, self.line_sensor):
            self.p_h.addItem(it)
        self.s_bins = pg.ScatterPlotItem(size=7, pen=None)
        self.s_hdet = pg.ScatterPlotItem(size=16, brush=pg.mkBrush("#ff3030"), pen=pg.mkPen("w", width=2))
        self.s_truth = pg.ScatterPlotItem(size=12, symbol="x", brush=pg.mkBrush("#00e5ff"), pen=pg.mkPen("#00e5ff"))
        for it in (self.s_bins, self.s_hdet, self.s_truth):
            self.p_h.addItem(it)
        grid.addWidget(self.p_h, 1, 0)

        self.p_rd = pg.PlotWidget(title="Range–Doppler (dB) / elevation per bin")
        self.p_rd.setLabel("bottom", "range [m]")
        self.img_rd = pg.ImageItem()
        self.img_rd.setColorMap(pg.colormap.get("plasma"))
        self.p_rd.addItem(self.img_rd)
        self.s_elev = pg.ScatterPlotItem(size=7, pen=None, brush=pg.mkBrush("#f5a54c"))
        self.p_rd.addItem(self.s_elev)
        grid.addWidget(self.p_rd, 1, 1)
        for p in (self.p_wf, self.p_h, self.p_rd):
            p.setXLink(self.p_prof)
        self._xmax = None
        self.setCentralWidget(central)

        self.status = QLabel("")
        self.statusBar().addPermanentWidget(self.status, 1)

    # ================================================================== controls
    def _build_controls(self):
        dock = QDockWidget("Controls", self)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        tabs = QTabWidget()
        tabs.addTab(self._tab_source(), "Source")
        tabs.addTab(self._scroll(self._tab_config()), "Radar config")
        tabs.addTab(self._scroll(self._tab_processing()), "Processing")
        tabs.addTab(self._scroll(self._tab_notes()), "Markers")
        hl = QTextEdit()
        hl.setReadOnly(True)
        hl.setHtml(HELP)
        tabs.addTab(hl, "Help")
        dock.setWidget(tabs)
        dock.setMinimumWidth(430)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _scroll(self, w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        return s

    def _tab_source(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.cb_kind = QComboBox()
        self.cb_kind.addItems(["Simulator", "Serial (radar module)", "Replay (recording)",
                               "SPI via CH347 (P1812, EXP-002)"])
        self.cb_kind.currentIndexChanged.connect(self._kind_changed)
        v.addWidget(QLabel("Source"))
        v.addWidget(self.cb_kind)

        self.g_serial = QGroupBox("Serial ports (dev board: higher COM = UART0)")
        fs = QFormLayout(self.g_serial)
        self.cb_ctrl = QComboBox()
        self.cb_ctrl.setEditable(True)
        self.cb_data = QComboBox()
        self.cb_data.setEditable(True)
        b_ref = QPushButton("Refresh ports")
        b_ref.clicked.connect(self._refresh_ports)
        fs.addRow("control (mmwc)", self.cb_ctrl)
        fs.addRow("data (cube)", self.cb_data)
        fs.addRow(b_ref)
        self.chk_start_on_err = QCheckBox("stream even if some config commands fail")
        self.chk_start_on_err.setChecked(True)
        fs.addRow(self.chk_start_on_err)
        v.addWidget(self.g_serial)

        self.g_spi = QGroupBox("SPI link (MODE SEL ON; DIP 1,2,5,6,7,9 ON)")
        fp = QFormLayout(self.g_spi)
        self.cb_spi_ctrl = QComboBox()
        self.cb_spi_ctrl.setEditable(True)
        self.sp_spi_dev = QSpinBox()
        self.sp_spi_dev.setRange(0, 15)
        self.cb_spi_clock = QComboBox()
        self.cb_spi_clock.addItems(["30000000", "15000000", "60000000", "7500000", "3750000"])
        self.cb_spi_clock.setEditable(True)
        self.cb_int_bit = QComboBox()
        self.cb_int_bit.addItems(["auto", "none", "0", "1", "2", "3", "4", "5", "6", "7"])
        fp.addRow("control (mmwc, single COM)", self.cb_spi_ctrl)
        fp.addRow("CH347 device index", self.sp_spi_dev)
        fp.addRow("SPI clock Hz", self.cb_spi_clock)
        fp.addRow("INT GPIO bit", self.cb_int_bit)
        v.addWidget(self.g_spi)

        self.g_sim = QGroupBox("Simulator")
        fm = QFormLayout(self.g_sim)
        self.cb_scene = QComboBox()
        self.cb_scene.addItems(list(scene_presets()))
        self.le_fail = QLineEdit()
        self.le_fail.setPlaceholderText("e.g. hpf; win   (simulate rejected commands)")
        self.le_over = QLineEdit()
        self.le_over.setPlaceholderText('chip runs other config, e.g. {"range_res_mm": 100}')
        fm.addRow("scene", self.cb_scene)
        fm.addRow("reject cmds", self.le_fail)
        fm.addRow("chip overrides", self.le_over)
        v.addWidget(self.g_sim)

        self.g_rep = QGroupBox("Replay")
        fr = QFormLayout(self.g_rep)
        self.le_path = QLineEdit()
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse)
        b_last = QPushButton("Latest run")
        b_last.clicked.connect(self._latest_run)
        hb = QHBoxLayout()
        hb.addWidget(b_browse)
        hb.addWidget(b_last)
        self.sp_speed = QDoubleSpinBox()
        self.sp_speed.setRange(0.05, 20)
        self.sp_speed.setValue(1.0)
        self.sp_speed.valueChanged.connect(self._speed)
        self.chk_loop = QCheckBox("loop")
        self.chk_loop.setChecked(True)
        self.b_pause = QPushButton("Pause")
        self.b_pause.setCheckable(True)
        self.b_pause.toggled.connect(self._pause)
        b_step = QPushButton("Step ▶")
        b_step.clicked.connect(lambda: self._step())
        self.sl_seek = QSlider(Qt.Horizontal)
        self.sl_seek.sliderReleased.connect(self._seek)
        fr.addRow("file / run dir", self.le_path)
        fr.addRow(hb)
        fr.addRow("speed", self.sp_speed)
        fr.addRow(self.chk_loop)
        hb2 = QHBoxLayout()
        hb2.addWidget(self.b_pause)
        hb2.addWidget(b_step)
        fr.addRow(hb2)
        fr.addRow("seek", self.sl_seek)
        v.addWidget(self.g_rep)

        self.chk_record = QCheckBox("Record raw cubes (cubes.h5)")
        self.chk_record.setChecked(True)      # cubes are small; a run without a recording cannot be re-analysed
        self.chk_record.toggled.connect(self._record_toggled)
        v.addWidget(self.chk_record)
        hb3 = QHBoxLayout()
        self.b_start = QPushButton("Start")
        self.b_start.setStyleSheet("font-weight:bold;padding:8px")
        self.b_start.clicked.connect(self._start)
        self.b_stop = QPushButton("Stop")
        self.b_stop.setStyleSheet("padding:8px")
        self.b_stop.clicked.connect(self._stop)
        hb3.addWidget(self.b_start)
        hb3.addWidget(self.b_stop)
        v.addLayout(hb3)
        self.lbl_run = QLabel("")
        self.lbl_run.setWordWrap(True)
        self.lbl_run.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.lbl_run)
        v.addStretch(1)
        self._refresh_ports()
        self._kind_changed()
        return w

    def _tab_config(self):
        w = QWidget()
        v = QVBoxLayout(w)
        hb = QHBoxLayout()
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(list(presets()))
        self.cb_preset.currentTextChanged.connect(self._preset_changed)
        b_load = QPushButton("Load…")
        b_load.clicked.connect(self._load_cfg)
        b_save = QPushButton("Save…")
        b_save.clicked.connect(self._save_cfg)
        hb.addWidget(QLabel("preset"))
        hb.addWidget(self.cb_preset, 1)
        hb.addWidget(b_load)
        hb.addWidget(b_save)
        v.addLayout(hb)
        self.lbl_valid = QLabel("")
        self.lbl_valid.setWordWrap(True)
        self.lbl_valid.setTextFormat(Qt.RichText)
        self.lbl_valid.setStyleSheet("background:#222;color:#ddd;padding:6px")
        v.addWidget(self.lbl_valid)
        self.cfg_form = ConfigForm(self._cfg_edited)
        v.addWidget(self.cfg_form)
        v.addWidget(QLabel("<i>Config changes apply on the next Start.</i>"))
        v.addStretch(1)
        self.cfg_form.set_config(self.cfg)
        self._cfg_edited()
        return w

    def _tab_processing(self):
        w = QWidget()
        v = QVBoxLayout(w)
        proc = self.engine.processor
        hb = QHBoxLayout()
        self.cb_primary = QComboBox()
        self.cb_primary.addItems(list(proc.detectors))
        self.cb_primary.setCurrentText(proc.primary)
        self.cb_primary.currentTextChanged.connect(self.engine.set_primary)
        hb.addWidget(QLabel("primary detector (drives banner)"))
        hb.addWidget(self.cb_primary, 1)
        # EXP-010: horizontal output FOV for all detectors (detection itself always uses the full data)
        self.cb_fov = QComboBox()
        self.cb_fov.addItems(["normal", "narrow", "laser"])
        self.cb_fov.setToolTip("Which confirmed obstacles may drive the output: normal = all; narrow = corridor "
                               "+-narrow_half_m ahead; laser = +-laser_half_m. Out-of-FOV detections stay visible "
                               "(grey). Needs a horizontal calibration (Calibrate phase with a target straight ahead).")
        self.cb_fov.setCurrentText(proc.detectors[proc.primary].p.get("fov", "normal"))
        self.cb_fov.currentTextChanged.connect(
            lambda lv: [self.engine.set_param(n, "fov", lv) for n in list(self.engine.processor.detectors)])
        hb.addWidget(QLabel("output FOV"))
        hb.addWidget(self.cb_fov)
        v.addLayout(hb)
        hb2 = QHBoxLayout()
        b_bg = QPushButton("Capture background (empty scene)")
        b_bg.setToolTip("Averages 40 frames with nothing in front; subtracts it afterwards (leakage + static scene).")
        b_bg.clicked.connect(lambda: self.engine.capture_background(40))
        b_ph = QPushButton("Calibrate phase (target at boresight)")
        b_ph.setToolTip("Put a reflector straight ahead at radar height; sets phase_offset_deg so it reads 0 deg.")
        b_ph.clicked.connect(self._cal_phase)
        hb2.addWidget(b_bg)
        hb2.addWidget(b_ph)
        v.addLayout(hb2)
        self.sp_show_snr = QDoubleSpinBox()
        self.sp_show_snr.setRange(0, 60)
        self.sp_show_snr.setValue(8)
        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0, 1)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.8)
        f = QFormLayout()
        f.addRow("show bins with SNR > (height plot)", self.sp_show_snr)
        f.addRow("... and angle coherence > (0-1)", self.sp_conf)
        v.addLayout(f)

        g = QGroupBox("Pre-processing (shared)")
        gl = QVBoxLayout(g)
        fw, self._pre_widgets = param_form(Preprocess.SPECS, proc.pre.p,
                                           lambda n, val: self.engine.set_param("preprocess", n, val))
        gl.addWidget(fw)
        v.addWidget(g)
        for name, det in proc.detectors.items():
            g = QGroupBox(f"Detector: {name}")
            g.setToolTip(det.description)
            gl = QVBoxLayout(g)
            gl.addWidget(QLabel(f"<i>{det.description}</i>"))
            fw, wd = param_form(det.SPECS, det.p, lambda n, val, d=name: self.engine.set_param(d, n, val))
            gl.addWidget(fw)
            self._det_widgets[name] = wd
            v.addWidget(g)
        v.addStretch(1)
        return w

    def _tab_notes(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Markers go into the run log with the current frame number."))
        for key, text in MARKERS:
            b = QPushButton(f"[{key}] {text}")
            b.clicked.connect(lambda _=False, t=text: self._marker(t))
            v.addWidget(b)
        v.addWidget(QLabel("<b>Distance holds</b> (Ctrl+1…Ctrl+8, Ctrl+0 = moving): press when the radar is at the mark"))
        hb = QHBoxLayout()
        for key, text in DIST_MARKERS:
            b = QPushButton(f"{text.replace('d=', '')}")
            b.setToolTip(key)
            b.clicked.connect(lambda _=False, t=text: self._marker(t))
            hb.addWidget(b)
        v.addLayout(hb)
        # ---- ground-truth takes (GT-001, docs/protocols/GT-001-ground-truth-session.md)
        from ..groundtruth import SCENES
        g = QGroupBox("Ground-truth take (GT-001): F7 start, F8 end; inside a walk take Ctrl+digit = passing a floor mark")
        fl = QFormLayout(g)
        self.cb_scene_gt = QComboBox()
        for sc in SCENES:
            self.cb_scene_gt.addItem(f"{sc['id']}  {sc['title']}", sc["id"])
        self.sp_rep = QSpinBox()
        self.sp_rep.setRange(1, 99)
        self.le_obj = QLineEdit()
        self.le_obj.setPlaceholderText("what the obstacle is, e.g. cardboard 30x40 on a string")
        self.sp_h = QDoubleSpinBox()
        self.sp_h.setRange(0.0, 3.0)
        self.sp_h.setSingleStep(0.05)
        self.sp_h.setSuffix(" m (bottom edge)")
        self.sp_lat = QDoubleSpinBox()
        self.sp_lat.setRange(-2.0, 2.0)
        self.sp_lat.setSingleStep(0.05)
        self.sp_lat.setSuffix(" m sideways")
        self.cb_scene_gt.currentIndexChanged.connect(self._scene_changed)
        fl.addRow("scene", self.cb_scene_gt)
        fl.addRow("repetition", self.sp_rep)
        fl.addRow("obstacle", self.le_obj)
        fl.addRow("height", self.sp_h)
        fl.addRow("lateral", self.sp_lat)
        hb3 = QHBoxLayout()
        b1 = QPushButton("[F7] Start take")
        b1.clicked.connect(self._take_start)
        b2 = QPushButton("[F8] End take")
        b2.clicked.connect(self._take_end)
        hb3.addWidget(b1)
        hb3.addWidget(b2)
        fl.addRow(hb3)
        self.lbl_take = QLabel("no take open")
        fl.addRow(self.lbl_take)
        v.addWidget(g)
        self._take_open = None
        self._scene_changed()
        v.addWidget(QLabel("<b>Phase check</b> (reflector ~1.2 m, radar still; then <code>ravidata phase-check RUN</code>)"))
        hb2 = QHBoxLayout()
        for k in ("boresight", "up", "down", "left", "right"):
            b = QPushButton(k)
            b.clicked.connect(lambda _=False, t=f"phase: {k}": self._marker(t))
            hb2.addWidget(b)
        v.addLayout(hb2)
        self.te_note = QPlainTextEdit()
        self.te_note.setPlaceholderText("Free note: what is in front of the radar, distances, what you expected…")
        v.addWidget(self.te_note)
        b = QPushButton("Add note")
        b.clicked.connect(self._add_note)
        v.addWidget(b)
        v.addStretch(1)
        return w

    def _build_log(self):
        dock = QDockWidget("Log (summary level)", self)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(3000)
        self.log_view.setStyleSheet("background:#111;color:#ddd;font-family:Consolas,monospace;font-size:9pt")
        dock.setWidget(self.log_view)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    # ================================================================== actions
    def _kind_changed(self, *_):
        k = self.cb_kind.currentIndex()
        self.g_sim.setVisible(k == 0)
        self.g_serial.setVisible(k == 1)
        self.g_rep.setVisible(k == 2)
        self.g_spi.setVisible(k == 3)

    def _refresh_ports(self):
        try:
            from serial.tools import list_ports
            ports = [(p.device, p.description) for p in list_ports.comports()]
        except Exception as e:  # pragma: no cover
            log.warning("could not list serial ports: %s", e)
            ports = []
        for cb in (self.cb_ctrl, self.cb_data, self.cb_spi_ctrl):
            cur = cb.currentText()
            cb.clear()
            for dev, desc in ports:
                cb.addItem(f"{dev}", dev)
                cb.setItemData(cb.count() - 1, desc, Qt.ToolTipRole)
            if cur:
                cb.setCurrentText(cur)
        log.info("serial ports: %s", ", ".join(f"{d} ({s})" for d, s in ports) or "none found")

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open recording", str(default_runs_dir()),
                                           "Recordings (*.h5 *.npz);;All (*)")
        if p:
            self.le_path.setText(p)

    def _latest_run(self):
        runs = default_runs_dir()
        cands = sorted([p for p in runs.glob("*/cubes*.h5")], key=lambda p: p.stat().st_mtime)
        if cands:
            self.le_path.setText(str(cands[-1]))
        else:
            QMessageBox.information(self, "No recordings", f"No cubes.h5 under {runs}")

    def _preset_changed(self, name):
        if name in presets():
            self.cfg = presets()[name]
            self.cfg_form.set_config(self.cfg)
            self._cfg_edited()

    def _cfg_edited(self):
        cfg, errs = self.cfg_form.get_config()
        if not errs:
            self.cfg = cfg
        lines = [f"<span style='color:#ff6060'>form: {html.escape(e)}</span>" for e in errs]
        try:
            d = self.cfg.derive()
            lines += [html.escape(s) for s in d.summary_lines()]
            iss = self.cfg.validate()
            col = {"error": "#ff6060", "warning": "#f0c040", "info": "#8ab4f8"}
            lines += [f"<span style='color:{col[i.level]}'>{html.escape(str(i))}</span>" for i in iss]
            if not iss:
                lines.append("<span style='color:#6c6'>no validation issues</span>")
        except Exception as e:
            lines.append(f"<span style='color:#ff6060'>derive failed: {html.escape(str(e))}</span>")
        self.lbl_valid.setText("<br>".join(lines))

    def _load_cfg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load config", str(Path.cwd()), "JSON (*.json)")
        if p:
            self.cfg = RadarConfig.load(Path(p))
            self.cfg_form.set_config(self.cfg)
            self._cfg_edited()

    def _save_cfg(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save config", f"{self.cfg.name}.json", "JSON (*.json)")
        if p:
            self.cfg.save(Path(p))

    def _start(self):
        k = self.cb_kind.currentIndex()
        try:
            if k == 2:
                path = self.le_path.text().strip()
                if not path:
                    QMessageBox.warning(self, "Replay", "Choose a recording first")
                    return
                self.engine.start_replay(path, speed=self.sp_speed.value(), loop=self.chk_loop.isChecked())
                src = self.engine.source
                self.sl_seek.setRange(0, max(0, src.rec.n - 1))
            else:
                cfg, errs = self.cfg_form.get_config()
                if errs:
                    QMessageBox.warning(self, "Config", "\n".join(errs))
                    return
                iss = cfg.validate()
                if has_errors(iss):
                    msg = "\n".join(str(i) for i in iss if i.level == "error")
                    r = QMessageBox.question(self, "Config has validation errors",
                                             f"{msg}\n\nSend it anyway? (useful to test what the chip does)")
                    if r != QMessageBox.Yes:
                        return
                if k == 0:
                    fails = {c.strip() for c in self.le_fail.text().split(";") if c.strip()}
                    over = json.loads(self.le_over.text()) if self.le_over.text().strip() else None
                    self.engine.start_sim(cfg, self.cb_scene.currentText(), record=self.chk_record.isChecked(),
                                          fail_commands=fails, chip_overrides=over)
                elif k == 3:
                    from ..cli_spi import int_bit_arg
                    ctrl = self.cb_spi_ctrl.currentText().strip()
                    if not ctrl:
                        QMessageBox.warning(self, "Ports", "Choose the control port")
                        return
                    self.engine.start_spi(cfg.copy(spi_clock_hz=int(float(self.cb_spi_clock.currentText()))), ctrl,
                                          self.sp_spi_dev.value(), int_bit_arg(self.cb_int_bit.currentText()),
                                          record=self.chk_record.isChecked())
                else:
                    ctrl, data = self.cb_ctrl.currentText().strip(), self.cb_data.currentText().strip()
                    if not ctrl or not data or ctrl == data:
                        QMessageBox.warning(self, "Ports", "Choose two different ports (control and data)")
                        return
                    self.engine.start_serial(cfg, ctrl, data, record=self.chk_record.isChecked(),
                                             start_on_config_errors=self.chk_start_on_err.isChecked())
            self.lbl_run.setText(f"run dir: {self.engine.session.dir}")
            self._save_settings()
            self._sync_param_widgets()
        except Exception as e:
            log.exception("start failed: %s", e)
            QMessageBox.critical(self, "Start failed", str(e))

    def _stop(self):
        d = self.engine.session.dir if self.engine.session else None
        self.engine.stop()
        if d:
            self.lbl_run.setText(f"last run: {d}\n(see SESSION_REPORT.md)")

    def _record_toggled(self, on):
        if self.engine.session and self.engine.source and self.engine.source.kind != "replay":
            self.engine.set_recording(on)

    def _pause(self, on):
        src = self.engine.source
        if src is not None and hasattr(src, "paused"):
            src.paused = on
            self.b_pause.setText("Resume" if on else "Pause")

    def _step(self):
        src = self.engine.source
        if src is not None and hasattr(src, "step"):
            if not src.paused:
                self.b_pause.setChecked(True)
            src.step(1)

    def _seek(self):
        src = self.engine.source
        if src is not None and hasattr(src, "seek"):
            src.seek(self.sl_seek.value())

    def _speed(self, v):
        src = self.engine.source
        if src is not None and hasattr(src, "speed"):
            src.speed = v

    def _scene_changed(self, *_):
        from ..groundtruth import SCENE_BY_ID
        sc = SCENE_BY_ID.get(self.cb_scene_gt.currentData(), {})
        self.sp_h.setValue(float(sc.get("h_m", 0.0)))
        self.sp_lat.setValue(float(sc.get("lat_m", 0.0)))

    def _take_start(self):
        from ..groundtruth import take_start_text
        if not self.engine.session:
            self.statusBar().showMessage("Start a run first; takes are stored per run", 3000)
            return
        if self._take_open:
            self._take_end()
        sid = self.cb_scene_gt.currentData()
        txt = take_start_text(sid, self.sp_rep.value(), obj=self.le_obj.text().strip(), h_m=self.sp_h.value(),
                              lat_m=self.sp_lat.value(), preset=self.cfg.name)
        self.engine.note(txt, kind="marker")
        self._take_open = (sid, self.sp_rep.value())
        self.lbl_take.setText(f"<b style='color:#ff5050'>TAKE OPEN: {sid} rep {self.sp_rep.value()}</b>")
        self.statusBar().showMessage(f"take start {sid} rep {self.sp_rep.value()}", 3000)

    def _take_end(self):
        if not self.engine.session or not self._take_open:
            return
        self.engine.note("take end", kind="marker")
        sid, rep_ = self._take_open
        self._take_open = None
        self.sp_rep.setValue(rep_ + 1)
        self.lbl_take.setText(f"last take: {sid} rep {rep_} (closed)")
        self.statusBar().showMessage("take end", 2000)

    def _cal_phase(self):
        self.engine.calibrate_phase()
        QTimer.singleShot(500, self._sync_param_widgets)

    def _marker(self, text):
        if not self.engine.session:
            self.statusBar().showMessage("Start a run first; markers are stored per run", 3000)
            return
        self.engine.note(text, kind="marker")
        self.statusBar().showMessage(f"marker: {text}", 2000)

    def _add_note(self):
        t = self.te_note.toPlainText().strip()
        if t and self.engine.session:
            self.engine.note(t)
            self.te_note.clear()

    def _sync_param_widgets(self):
        proc = self.engine.processor
        if not proc:
            return
        for n, e in self._pre_widgets.items():
            set_param_widget(e, proc.pre.p[n])
        for det, wd in self._det_widgets.items():
            for n, e in wd.items():
                set_param_widget(e, proc.detectors[det].p[n])

    # ================================================================== refresh
    def _tick(self):
        self._pull_logs()
        eng = self.engine
        res, fr, hist, counter = eng.snapshot()
        src = eng.source
        self._update_status(src)
        if res is None or counter == self._last_counter:
            return
        self._last_counter = counter
        try:
            self._draw(res, fr, hist)
        except Exception as e:  # pragma: no cover
            log.exception("draw failed: %s", e)
        if src is not None and src.kind == "replay" and not self.sl_seek.isSliderDown():
            self.sl_seek.setValue(src.pos)

    def _update_status(self, src):
        eng = self.engine
        if eng.state in ("error", "connecting", "configuring", "configured"):
            txt = {"error": f"ERROR: {eng.state_info.get('error', '?')}  (see log / SESSION_REPORT.md)",
                   "connecting": "connecting…", "configuring": "configuring radar…",
                   "configured": "configured, starting…"}[eng.state]
            col = "#b71c1c" if eng.state == "error" else "#455a64"
            self.banner.setText(txt)
            self.banner.setStyleSheet(f"background:{col};color:white;padding:6px;border-radius:6px")
        if src is None:
            self.status.setText(f"state: {eng.state}")
            return
        st = src.stats
        cfg = eng.cfg
        d = cfg.derive() if cfg else None
        parts = [f"state: <b>{eng.state}</b>", f"{st.fps():.1f} fps" + (f" / {d.expected_fps:.0f}" if d else ""),
                 f"frames {st.frames}", f"idx gaps {st.index_gaps}"]
        ps = getattr(src, "parser", None)
        if ps is not None:
            e = dict(ps.stats.errors)
            parts.append(f"parse err {sum(e.values())}")
            parts.append(f"incomplete {ps.stats.frames_incomplete}")
        if st.shape_mismatch:
            parts.append(f"<span style='color:#ff6060'>SHAPE {st.shape_used}</span>")
        if d:
            parts.append(f"link {100*d.link_utilisation:.0f}%")
        if eng.latest:
            parts.append(f"proc {eng.latest.proc_ms:.1f} ms")
        if eng.session and eng.session.recorder:
            parts.append(f"<span style='color:#ff5050'>● REC {eng.session.recorder.n}</span>")
        self.status.setText(" | ".join(parts))

    def _pull_logs(self):
        ring = get_ring()
        new = ring.since(self._ring_pos)
        if not new:
            return
        self._ring_pos = new[-1][0]
        for _, line in new:
            parts = line.split()
            lvl = parts[1] if len(parts) > 1 else "INFO"
            col = LEVEL_COLORS.get(lvl, "#ddd")
            self.log_view.appendHtml(f"<span style='color:{col}'>{html.escape(line)}</span>")

    def _draw(self, res, fr, hist):
        f = res.features
        r = f.r_m
        proc = self.engine.processor
        for i, c in enumerate(self.c_rx):
            if i < f.p_rx_db.shape[0]:
                c.setData(r, f.p_rx_db[i])
            else:
                c.clear()
        self.c_comb.setData(r, f.p_db)
        self.c_floor.setData(r, f.floor_db)
        out = res.primary_out
        if out.threshold_db is not None:
            self.c_thr.setData(r, out.threshold_db)
        else:
            self.c_thr.clear()
        if out.detections:
            # EXP-010: obstacles outside the output FOV are shown grey (detected, not alerted)
            self.s_det.setData([d.range_m for d in out.detections], [f.p_db[d.bin] for d in out.detections],
                               brush=[pg.mkBrush("#ff3030" if getattr(d, "in_fov", True) else "#808080")
                                      for d in out.detections])
        else:
            self.s_det.clear()
        pre = proc.pre.p
        rmax = r[-1] if len(r) else 1
        if self._xmax != rmax:
            self._xmax = rmax
            for p in (self.p_prof, self.p_wf, self.p_h, self.p_rd):
                p.enableAutoRange(x=False)
            self.p_prof.setXRange(0, rmax + f.bin_m, padding=0.01)
        self.crop_near.setRegion((0, pre["near_crop_m"]))
        self.crop_far.setRegion((rmax * (1 - pre["far_crop_frac"]), rmax))

        near = out.nearest
        if near:
            self.v_near.setVisible(True)
            self.v_near.setValue(near.range_m)
            hs = f" · height {near.height_m:.2f} m ({near.elev_deg:+.0f}°)" if math.isfinite(near.height_m) else ""
            if math.isfinite(getattr(near, "lateral_m", float("nan"))):
                hs += f" · side {near.lateral_m:+.2f}±{near.lat_sigma_m:.2f} m"
            self.banner.setText(f"OBSTACLE {near.range_m:.2f} m{hs}")
            closeness = near.range_m / max(0.1, self.engine.processor.detectors[res.primary].p["max_range_m"])
            col = "#c62828" if closeness < 0.5 else "#ef6c00"
            self.banner.setStyleSheet(f"background:{col};color:white;padding:6px;border-radius:6px")
        else:
            self.v_near.setVisible(False)
            self.banner.setText("clear")
            self.banner.setStyleSheet("background:#2e7d32;color:white;padding:6px;border-radius:6px")
        items = []
        for name, o in res.outputs.items():
            s = f"{o.nearest.range_m:.2f} m" if o.nearest else "—"
            items.append(f"<b>{name}</b>{' (primary)' if name == res.primary else ''}: {s}")
        tr = ""
        if res.truth:
            vis = [v for v in res.truth.get("visible", []) if v["label"] != "ground"]
            if vis:
                t = min(vis, key=lambda v: v["r"])
                tr = f" &nbsp;|&nbsp; <span style='color:#00e5ff'>truth: {t['label']} at {t['r']:.2f} m, el {t['el_deg']:+.0f}°</span>"
        self.det_line.setText(" &nbsp;&nbsp; ".join(items) + tr)

        # waterfall
        if hist:
            H = np.array(hist[::-1])
            self.img_wf.setImage(H, autoLevels=False, levels=(np.percentile(H, 5), np.percentile(H, 99.5) + 1e-3))
            self.img_wf.setRect(0, 0, rmax + f.bin_m, H.shape[0])

        # range x height
        thr = self.sp_show_snr.value()
        m = f.valid & (f.snr_db > thr) & np.isfinite(f.height_m)
        if getattr(f, "el_conf", None) is not None:
            m &= f.el_conf >= self.sp_conf.value()     # D-024: hide bins whose angle is not stable
        if m.any():
            snr = np.clip((f.snr_db[m] - thr) / 30.0, 0, 1)
            brushes = [pg.mkBrush(int(255 * s), int(180 * (1 - s)) + 60, 255 - int(200 * s), 220) for s in snr]
            self.s_bins.setData(r[m], f.height_m[m], brush=brushes)
        else:
            self.s_bins.clear()
        if out.detections:
            self.s_hdet.setData([d.range_m for d in out.detections], [d.height_m for d in out.detections])
        else:
            self.s_hdet.clear()
        self.line_sensor.setValue(pre["sensor_height_m"])
        dp = proc.detectors[res.primary].p
        self.band_ground.setRegion((0, dp["ground_max_m"]))
        self.line_clear.setValue(dp["head_clear_m"])
        if res.truth:
            vis = [v for v in res.truth.get("visible", []) if v["label"] != "ground"]
            self.s_truth.setData([v["r"] for v in vis], [v["z"] for v in vis])
        else:
            self.s_truth.clear()

        # range-doppler or elevation
        if f.rd_db is not None:
            self.s_elev.clear()
            self.img_rd.setVisible(True)
            rd = f.rd_db
            self.img_rd.setImage(rd, autoLevels=False, levels=(np.percentile(rd, 10), rd.max() + 1e-3))
            nd = rd.shape[0]
            self.img_rd.setRect(0, -nd / 2, rmax + f.bin_m, nd)
            self.p_rd.setLabel("left", "Doppler bin (0 = DC)")
            self.p_rd.setTitle("Range–Doppler (dB, mean over RX)")
        else:
            self.img_rd.setVisible(False)
            mm = f.valid & (f.snr_db > thr)
            self.s_elev.setData(r[mm], f.elev_deg[mm])
            self.p_rd.setLabel("left", "elevation [deg]")
            self.p_rd.setTitle("Elevation per bin (1D mode; bins above show-SNR)")
            self.p_rd.setYRange(-90, 90)

    # ================================================================== settings
    def _save_settings(self):
        s = self.settings
        s.setValue("kind", self.cb_kind.currentIndex())
        s.setValue("ctrl", self.cb_ctrl.currentText())
        s.setValue("data", self.cb_data.currentText())
        s.setValue("scene", self.cb_scene.currentText())
        s.setValue("replay", self.le_path.text())
        s.setValue("config_json", json.dumps(self.cfg.to_dict()))
        s.setValue("preset", self.cb_preset.currentText())
        if self.engine.processor:
            s.setValue("params_json", json.dumps(self.engine.processor.params_snapshot(), default=str))

    def _load_settings(self):
        s = self.settings
        try:
            self.cb_kind.setCurrentIndex(int(s.value("kind", 0)))
            if s.value("ctrl"):
                self.cb_ctrl.setCurrentText(s.value("ctrl"))
            if s.value("data"):
                self.cb_data.setCurrentText(s.value("data"))
            if s.value("scene"):
                self.cb_scene.setCurrentText(s.value("scene"))
            self.le_path.setText(s.value("replay", "") or "")
            if s.value("preset"):
                self.cb_preset.blockSignals(True)
                self.cb_preset.setCurrentText(s.value("preset"))
                self.cb_preset.blockSignals(False)
            if s.value("config_json"):
                self.cfg = RadarConfig.from_dict(json.loads(s.value("config_json")))
                self.cfg_form.set_config(self.cfg)
                self._cfg_edited()
            if s.value("params_json"):
                self.engine._saved_params = json.loads(s.value("params_json"))
                self.engine._apply_params(self.engine._saved_params)
                self._sync_param_widgets()
                self.cb_primary.blockSignals(True)          # restoring is not a user choice (mode default applies)
                self.cb_primary.setCurrentText(self.engine.processor.primary)
                self.cb_primary.blockSignals(False)
        except Exception as e:  # corrupted settings must never block the app
            log.warning("could not restore GUI settings: %s", e)

    def closeEvent(self, ev):
        self._save_settings()
        self.engine.stop("window closed")
        super().closeEvent(ev)


def run_gui(argv=None):
    runs = default_runs_dir()
    setup_base(logging.INFO, app_log_dir=runs)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    pg.setConfigOption("background", "#1a1a1a")
    eng = Engine(runs_dir=runs)
    w = MainWindow(eng)
    w.show()
    log.info("GUI started; runs are written to %s", runs)
    return app.exec()
