"""Alignment viewer: the phone video, the rig's 3D attitude (phone IMU) and the radar, on one clock.

Start with ``uv run ravidata align`` (or ``make align``).

1. Open the folder of a phone recording (the video + the Sensor Logger zip), or pick the files one by one.
2. "Load + auto-align": the video is matched to the radar takes by the recorder's beeps in its sound, the IMU by
   correlating the gyro with the motion in the video (see ``ravidata/align.py``).
3. Scrub or play. The timeline at the bottom shows the video motion and the gyro rate (their peaks must coincide),
   the radar takes (shaded) with their beeps (red) and the beeps heard in the video (cyan, dashed: they must sit on
   the red ones). Correct by hand with the two offsets if needed.
4. "Export pack" writes one ``.tccpack.h5`` with everything (``ravidata.align.load_pack`` reads it).
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF, QTransform
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSizePolicy, QSlider, QSplitter, QVBoxLayout, QWidget)

from .. import align as A
from .. import orientation
from ..logsetup import get_logger, setup_base
from ..session import default_runs_dir

log = get_logger("aligner")
H_COL, V_COL = QColor("#1e88e5"), QColor("#fb8c00")


# =============================================================================================== video frames
class VideoReader:
    """Frames on demand: sequential decoding while playing, seek + decode when jumping. Small LRU cache."""

    def __init__(self, path: Path, pts: np.ndarray, max_h: int = 560):
        import av
        self.av = av
        self.path, self.pts = path, pts
        self.c = av.open(str(path))
        self.vs = self.c.streams.video[0]
        self.vs.thread_type = "AUTO"
        h = self.vs.codec_context.height
        w = self.vs.codec_context.width
        s = min(1.0, max_h / max(h, w))
        self.out_w, self.out_h = max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)
        self.cache: OrderedDict[int, QImage] = OrderedDict()
        self._it = None
        self._next = None            # index the sequential iterator will produce next

    def close(self):
        try:
            self.c.close()
        except Exception:
            pass

    def _to_img(self, fr) -> QImage:
        a = np.ascontiguousarray(fr.reformat(width=self.out_w, height=self.out_h, format="rgb24").to_ndarray())
        img = QImage(a.data, a.shape[1], a.shape[0], 3 * a.shape[1], QImage.Format_RGB888).copy()
        rot = int(round(getattr(fr, "rotation", 0) or 0))
        if rot:
            img = img.transformed(QTransform().rotate(-rot))
        return img

    def _index(self, t: float) -> int:
        k = int(np.searchsorted(self.pts, t - 1e-4))
        return min(max(k, 0), self.pts.size - 1)

    def frame(self, i: int) -> QImage | None:
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        try:
            if self._it is None or self._next is None or not (self._next <= i <= self._next + 8):
                target = float(self.pts[i])
                self.c.seek(int(max(0.0, target - 0.05) / float(self.vs.time_base)), stream=self.vs, backward=True)
                self._it = self.c.decode(video=0)
                self._next = None
            img = None
            for fr in self._it:
                k = self._index(fr.time if fr.time is not None else 0.0)
                self._next = k + 1
                if k >= i - 0 and k >= i:
                    img = self._to_img(fr)
                    self._store(k, img)
                    break
                if k >= i - 2:                       # keep the frames just before for stepping back
                    self._store(k, self._to_img(fr))
            return img
        except Exception as e:  # pragma: no cover - broken file
            log.warning("video decode failed at frame %d: %s", i, e)
            self._it = None
            return None

    def _store(self, k: int, img: QImage):
        self.cache[k] = img
        while len(self.cache) > 48:
            self.cache.popitem(last=False)


# =============================================================================================== 3D rig view
class RigView(QWidget):
    """Wireframe 3D view of the rig in the world (z up). Drag with the mouse to turn the view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.view_az, self.view_el = -35.0, 18.0
        self.R = np.eye(3)              # device -> world at the current time
        self.yaw0 = 0.0                 # world turned so the rig's mean heading points away from the viewer
        self.axes = {"up": np.array([0, 1.0, 0]), "back": np.array([0, 0, 1.0]), "right": np.array([1.0, 0, 0])}
        self.height_m = 0.9
        self.offset = np.zeros(3)       # radar position relative to the phone lens, in rig axes (right, up, back), m
        self.rotation = 0
        self.info = ""
        self._drag = None
        self.setToolTip("Drag to turn the view. Blue/orange on the radar: horizontal/vertical RX pair. "
                        "Lines from the radar: boresight and the ±60° vertical field of view.")

    def set_state(self, R: np.ndarray | None, info: str):
        self.R = np.eye(3) if R is None or not np.isfinite(R).all() else R
        self.info = info
        self.update()

    def mousePressEvent(self, e):
        self._drag = (e.position().x(), e.position().y(), self.view_az, self.view_el)

    def mouseMoveEvent(self, e):
        if self._drag:
            x0, y0, az, el = self._drag
            self.view_az = az + (e.position().x() - x0) * 0.4
            self.view_el = float(np.clip(el + (e.position().y() - y0) * 0.3, -10, 85))
            self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _proj(self, P: np.ndarray, cx: float, cy: float, s: float) -> list[QPointF]:
        P = np.atleast_2d(P)
        c, sn = math.cos(math.radians(self.yaw0)), math.sin(math.radians(self.yaw0))
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])          # mean heading of the rig -> +y (into the screen)
        P = P @ Rz.T
        a, e = math.radians(self.view_az), math.radians(self.view_el)
        Va = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        P = P @ Va.T
        # camera behind and above, looking along +y and tilted down by e: far points rise on the screen
        sx = P[:, 0]
        sy = -(P[:, 2] * math.cos(e) + P[:, 1] * math.sin(e))
        return [QPointF(cx + s * x, cy + s * y) for x, y in zip(sx, sy)]

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#16181b"))
        w, h = self.width(), self.height()
        cx, cy, s = w * 0.42, h * 0.45, min(w * 0.55, h) * 0.42
        H = self.height_m
        grey, dim = QColor("#5f6368"), QColor("#30343a")
        # ground grid (radar at the origin, ground at -H)
        p.setPen(QPen(dim, 1))
        for v in np.arange(-0.5, 2.51, 0.5):
            a, b = self._proj(np.array([[-1.0, v, -H], [1.0, v, -H]]), cx, cy, s)
            p.drawLine(a, b)
        for u in np.arange(-1.0, 1.01, 0.5):
            a, b = self._proj(np.array([[u, -0.5, -H], [u, 2.5, -H]]), cx, cy, s)
            p.drawLine(a, b)
        # world up
        o = np.zeros(3)
        p.setPen(QPen(grey, 1, Qt.DashLine))
        a, b = self._proj(np.array([o, [0, 0, 0.6]]), cx, cy, s)
        p.drawLine(a, b)
        p.drawText(b + QPointF(4, 0), "up")
        R = self.R
        up, right, back = (R @ self.axes[k] for k in ("up", "right", "back"))
        fwd = -back
        rad = R @ (self.axes["right"] * self.offset[0] + self.axes["up"] * self.offset[1] + self.axes["back"] * self.offset[2])
        k = 3.0                                                       # draw the hardware 3x larger than life

        def box(c, hw, hh, hd, col):
            pts = [c + sx * hw * right + sy * hh * up + sz * hd * back for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
            P = self._proj(np.array(pts), cx, cy, s)
            fill = QColor(col)
            fill.setAlpha(70)
            p.setPen(Qt.NoPen)
            p.setBrush(fill)
            p.drawPolygon(QPolygonF([P[0], P[2], P[3], P[1]]))                 # front face (towards the scene)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 1.5))
            for i, j in ((0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)):
                p.drawLine(P[i], P[j])

        # phone at the lens (the radar is at the origin); its long side follows the phone's own y axis
        lens = -k * rad
        portrait = abs(self.axes["up"][1]) > 0.5
        box(lens, (0.037 if portrait else 0.08) * k, (0.08 if portrait else 0.037) * k, 0.004 * k, QColor("#cfd8dc"))
        # radar module (at the origin)
        box(o, 0.009 * k * 2.5, 0.006 * k * 2.5, 0.002 * k, QColor("#ef6c4a"))
        # RX pairs on the radar face: horizontal = rig right, vertical = rig up
        for vec, col in ((right, H_COL), (up, V_COL)):
            a, b = self._proj(np.array([o - 0.03 * vec, o + 0.03 * vec]), cx, cy, s)
            p.setPen(QPen(col, 3))
            p.drawLine(a, b)
        # boresight and the vertical field of view (±60° about the rig's right axis)
        a, b = self._proj(np.array([o, o + 2.5 * fwd]), cx, cy, s)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(a, b)
        for ang in (60, -60):
            t = math.radians(ang)
            d = math.cos(t) * fwd + math.sin(t) * up
            a, b = self._proj(np.array([o, o + 1.8 * d]), cx, cy, s)
            p.setPen(QPen(V_COL, 1, Qt.DashLine))
            p.drawLine(a, b)
        p.setPen(QColor("#e0e0e0"))
        f = p.font()
        f.setPointSize(9)
        p.setFont(f)
        p.drawText(QRectF(8, h - 58, w - 16, 54), Qt.AlignLeft | Qt.AlignBottom, self.info)
        p.end()


# =============================================================================================== main window
class AlignWindow(QMainWindow):
    def __init__(self, runs_dir: Path):
        super().__init__()
        self.runs_dir = Path(runs_dir)
        self.settings = QSettings("RAVIDATA", "ravidata-align")
        self.setWindowTitle("ravidata align — video + IMU + radar")
        self.resize(1500, 920)
        self.scan: A.VideoScan | None = None
        self.imu: A.Imu | None = None
        self.al = A.Alignment()
        self.takes: list[A.Take] = []
        self.radar: dict[str, dict | None] = {}
        self.reader: VideoReader | None = None
        self.t_ref = 0.0
        self.t = 0.0                      # current time, host clock, relative to t_ref
        self.t_lo, self.t_hi = 0.0, 1.0
        self.playing = False
        self._job = None
        self._wf_take = None
        self._last_frame = -1

        top = QWidget()
        tl = QGridLayout(top)
        self.video_edit, self.imu_edit = QLabel("-"), QLabel("-")
        for lab in (self.video_edit, self.imu_edit):
            lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
            lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        b_folder = QPushButton("Open phone recording folder…")
        b_folder.clicked.connect(self.pick_folder)
        b_vid, b_imu = QPushButton("video…"), QPushButton("IMU zip…")
        b_vid.clicked.connect(self.pick_video)
        b_imu.clicked.connect(self.pick_imu)
        self.b_load = QPushButton("Load + auto-align")
        self.b_load.setStyleSheet("font-weight:bold; padding:5px;")
        self.b_load.clicked.connect(self.load)
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(220)
        self.prog.setVisible(False)
        tl.addWidget(b_folder, 0, 0, 2, 1)
        tl.addWidget(b_vid, 0, 1)
        tl.addWidget(self.video_edit, 0, 2)
        tl.addWidget(b_imu, 1, 1)
        tl.addWidget(self.imu_edit, 1, 2)
        tl.addWidget(self.b_load, 0, 3, 2, 1)
        tl.addWidget(self.prog, 0, 4, 2, 1)
        tl.setColumnStretch(2, 1)

        # video
        self.video = QLabel("open a phone recording")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(300, 400)
        self.video.setStyleSheet("background:#000; color:#888;")
        self.video.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        ctl = QHBoxLayout()
        self.b_play = QPushButton("▶")
        self.b_play.setFixedWidth(40)
        self.b_play.clicked.connect(self.toggle_play)
        b_prev, b_next = QPushButton("◀|"), QPushButton("|▶")
        for b_, d in ((b_prev, -1), (b_next, 1)):
            b_.setFixedWidth(36)
            b_.clicked.connect(lambda _=False, d=d: self.step_frame(d))
        self.speed = QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×"])
        self.speed.setCurrentText("1×")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.valueChanged.connect(self._slider_moved)
        self.tlabel = QLabel("t = -")
        self.tlabel.setMinimumWidth(150)
        for wdg in (self.b_play, b_prev, b_next, self.speed):
            ctl.addWidget(wdg)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.tlabel)
        vbox = QWidget()
        vl = QVBoxLayout(vbox)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self.video, 1)
        vl.addLayout(ctl)

        # 3D + radar
        self.rig = RigView()
        pg.setConfigOptions(antialias=False, background="#151515")
        self.rw = pg.GraphicsLayoutWidget()
        self.p_wf = self.rw.addPlot(row=0, col=0, title="Radar: range over time (selected take)")
        self.p_wf.setLabel("bottom", "range", "m")
        self.p_wf.setLabel("left", "t", "s")
        self.wf_img = pg.ImageItem()
        self.wf_img.setColorMap(pg.colormap.get("viridis"))
        self.p_wf.addItem(self.wf_img)
        self.wf_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#ff5252", width=1.5))
        self.p_wf.addItem(self.wf_line)
        self.p_now = self.rw.addPlot(row=0, col=1, title="Radar frame now")
        self.p_now.setLabel("bottom", "range", "m")
        self.prof = [self.p_now.plot(pen=pg.mkPen(c, width=1.3)) for c in ("#4fc3f7", "#ffb74d", "#aed581")]
        self.rd_img = pg.ImageItem()
        self.rd_img.setColorMap(pg.colormap.get("inferno"))
        self.p_rd = self.rw.addPlot(row=1, col=1, title="Range-Doppler now (2D)")
        self.p_rd.addItem(self.rd_img)
        self.p_rd.setLabel("bottom", "range", "m")
        self.p_rd.setLabel("left", "v", "m/s")
        self.rw.ci.layout.setColumnStretchFactor(0, 3)
        self.rw.ci.layout.setColumnStretchFactor(1, 2)
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.rig)
        right.addWidget(self.rw)
        right.setSizes([330, 420])

        # timeline + alignment controls
        self.tp = pg.PlotWidget(title="Alignment: video motion (green) and gyro rate (orange) must peak together; "
                                      "video beeps (cyan, dashed) must sit on the radar beeps (red)")
        self.tp.setLabel("bottom", "t", "s (0 = first take's start beep)")
        self.tp.setMinimumHeight(170)
        self.c_motion = self.tp.plot(pen=pg.mkPen("#66bb6a", width=1.2))
        self.c_gyro = self.tp.plot(pen=pg.mkPen("#ffa726", width=1.2))
        self.cursor = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen("#ffffff", width=1.5))
        self.cursor.sigPositionChanged.connect(lambda: self.set_time(self.cursor.value(), from_cursor=True))
        self.tp.addItem(self.cursor)
        self._marks: list = []
        ctrl = QGroupBox("Alignment")
        cg = QGridLayout(ctrl)
        self.sp_video = QDoubleSpinBox()
        self.sp_imu = QDoubleSpinBox()
        for sp in (self.sp_video, self.sp_imu):
            sp.setRange(-10000, 10000)
            sp.setDecimals(0)
            sp.setSingleStep(10)
            sp.setSuffix(" ms")
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._manual_changed)
        self.sp_video.setToolTip("Added to the video time (positive = the video happens later on the radar clock).")
        self.sp_imu.setToolTip("Added to the IMU time (positive = the IMU happens later on the radar clock).")
        self.take_combo = QComboBox()
        self.take_combo.currentIndexChanged.connect(self._take_selected)
        b_auto = QPushButton("Re-run auto")
        b_auto.clicked.connect(self.realign)
        b_zero = QPushButton("Reset manual")
        b_zero.clicked.connect(lambda: (self.sp_video.setValue(0), self.sp_imu.setValue(0)))
        self.embed = QCheckBox("embed the video in the pack")
        self.embed.setChecked(True)
        self.b_export = QPushButton("Export pack…")
        self.b_export.setStyleSheet("font-weight:bold; padding:5px;")
        self.b_export.clicked.connect(self.export)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setStyleSheet("font-family:Consolas,monospace; font-size:9pt;")
        cg.addWidget(QLabel("radar take"), 0, 0)
        cg.addWidget(self.take_combo, 0, 1, 1, 2)
        cg.addWidget(QLabel("video offset"), 1, 0)
        cg.addWidget(self.sp_video, 1, 1)
        cg.addWidget(QLabel("IMU offset"), 2, 0)
        cg.addWidget(self.sp_imu, 2, 1)
        cg.addWidget(b_auto, 1, 2)
        cg.addWidget(b_zero, 2, 2)
        cg.addWidget(self.summary, 3, 0, 1, 3)
        cg.addWidget(self.embed, 4, 0, 1, 2)
        cg.addWidget(self.b_export, 4, 2)
        ctrl.setMaximumWidth(430)
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.tp, 1)
        bl.addWidget(ctrl)

        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(vbox)
        mid.addWidget(right)
        mid.setSizes([520, 900])
        vert = QSplitter(Qt.Vertical)
        vert.addWidget(mid)
        vert.addWidget(bottom)
        vert.setSizes([640, 260])
        root = QWidget()
        rl = QVBoxLayout(root)
        rl.addWidget(top)
        rl.addWidget(vert, 1)
        self.setCentralWidget(root)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._last_tick = time.monotonic()
        last = self.settings.value("folder")
        if last and Path(str(last)).exists():
            self._use_folder(Path(str(last)))

    # ------------------------------------------------------------------ files
    def _use_folder(self, folder: Path):
        v, z = A.find_phone_files(folder)
        self.video_path, self.imu_path = v, z
        self.video_edit.setText(str(v) if v else "(no video in the folder)")
        self.imu_edit.setText(str(z) if z else "(no Sensor Logger zip in the folder)")

    def pick_folder(self):
        start = str(self.settings.value("folder") or self.runs_dir)
        d = QFileDialog.getExistingDirectory(self, "Folder with the phone video and the Sensor Logger zip", start)
        if d:
            self.settings.setValue("folder", d)
            self._use_folder(Path(d))

    def pick_video(self):
        f, _ = QFileDialog.getOpenFileName(self, "Phone video", str(self.settings.value("folder") or self.runs_dir),
                                           "Video (*.mp4 *.mov *.3gp *.mkv)")
        if f:
            self.video_path = Path(f)
            self.video_edit.setText(f)

    def pick_imu(self):
        f, _ = QFileDialog.getOpenFileName(self, "Sensor Logger export", str(self.settings.value("folder") or self.runs_dir),
                                           "Sensor Logger zip (*.zip)")
        if f:
            self.imu_path = Path(f)
            self.imu_edit.setText(f)

    # ------------------------------------------------------------------ loading (background)
    def load(self, video: Path | None = None, imu: Path | None = None):
        video = video or getattr(self, "video_path", None)
        imu = imu or getattr(self, "imu_path", None)
        if not video and not imu:
            QMessageBox.information(self, "Nothing to load", "Open a phone recording folder (or pick the files).")
            return
        if self._job is not None:
            return
        self.video_path, self.imu_path = video, imu
        self.video_edit.setText(str(video) if video else "(no video)")
        self.imu_edit.setText(str(imu) if imu else "(no IMU)")
        self._job = {"done": False, "p": 0.0, "err": None}
        self.prog.setVisible(True)
        self.b_load.setEnabled(False)

        def run():
            try:
                job = self._job
                job["scan"] = A.scan_video(video, progress=lambda x: job.__setitem__("p", 0.8 * x)) if video else None
                job["imu"] = A.load_sensorlogger(imu) if imu else None
                job["all_takes"] = A.find_takes(self.runs_dir)
                al = A.auto_align(job["scan"], job["imu"], job["all_takes"])
                job["al"] = al
                job["takes"] = A.takes_in(job["all_takes"], al, job["scan"], job["imu"])
                job["radar"] = {t.run_id: A.load_take_radar(t) for t in job["takes"]}
                A.check_with_radar(al, job["scan"], job["radar"])
                job["p"] = 1.0
            except Exception as e:
                log.exception("load failed")
                self._job["err"] = repr(e)
            self._job["done"] = True

        threading.Thread(target=run, daemon=True).start()

    def _load_finished(self):
        job, self._job = self._job, None
        self.prog.setVisible(False)
        self.b_load.setEnabled(True)
        if job["err"]:
            QMessageBox.critical(self, "Load failed", job["err"])
            return
        if self.reader:
            self.reader.close()
        self.scan, self.imu, self.al = job["scan"], job["imu"], job["al"]
        self.takes, self.radar = job["takes"], job["radar"]
        self.reader = VideoReader(self.scan.path, self.scan.pts) if self.scan else None
        self.session_meta = A.session_meta_for(self.runs_dir, self.takes)
        self.sp_video.blockSignals(True)
        self.sp_imu.blockSignals(True)
        self.sp_video.setValue(0)
        self.sp_imu.setValue(0)
        self.sp_video.blockSignals(False)
        self.sp_imu.blockSignals(False)
        self._setup_rig()
        self.take_combo.blockSignals(True)
        self.take_combo.clear()
        for t in self.takes:
            have = "" if self.radar.get(t.run_id) else "  (cubes.h5 missing)"
            self.take_combo.addItem(f"{t.run_id} [{t.config}]{have}", t.run_id)
        self.take_combo.blockSignals(False)
        self.t_ref = self.takes[0].t_beep_start if self.takes else (
            float(self.al.video_to_host(self.scan.pts[0])) if self.scan else float(self.imu.t_range[0]))
        self._update_span()
        self._redraw_timeline()
        self._take_selected()
        self.set_time(0.0 if self.takes else self.t_lo)
        self._show_summary()

    def realign(self):
        if self.scan is None and self.imu is None:
            return
        self.al = A.auto_align(self.scan, self.imu, A.find_takes(self.runs_dir))
        A.check_with_radar(self.al, self.scan, self.radar)
        self.al.manual_video_ms, self.al.manual_imu_ms = self.sp_video.value(), self.sp_imu.value()
        self._redraw_timeline()
        self._show_summary()
        self.set_time(self.t)

    def _setup_rig(self):
        if self.imu is None:
            return
        ax = A.rig_axes(self.imu)
        self.rig.axes = {k: ax[k] for k in ("up", "back", "right")}
        meta = (self.session_meta or {}).get("meta", {})

        def num(k, scale, default):
            try:
                return float(meta.get(k, "")) * scale
            except (TypeError, ValueError):
                return default
        self.rig.height_m = num("radar_height_m", 1.0, 0.9)
        # radar relative to the lens = -(camera relative to the radar); right/up/back in metres
        self.rig.offset = -np.array([num("cam_dx_cm", 0.01, 0.0), num("cam_dy_cm", 0.01, 0.12), num("cam_dz_cm", 0.01, 0.02)])
        try:
            self.rig.rotation = int(meta.get("radar_rotation_deg", 0))
        except (TypeError, ValueError):
            self.rig.rotation = 0
        att = A.attitude(self.imu, self.al, self.al.imu_to_host(self.imu.sensors["Gyroscope"]["t"]))
        yaw = att["yaw_deg"]
        self.rig.yaw0 = float(np.degrees(np.angle(np.nanmean(np.exp(1j * np.radians(yaw[np.isfinite(yaw)])))))) \
            if np.isfinite(yaw).any() else 0.0
        self._phone_detected = ax["phone_orientation"]

    def _update_span(self):
        lo, hi = [], []
        if self.scan is not None:
            lo.append(float(self.al.video_to_host(self.scan.pts[0])))
            hi.append(float(self.al.video_to_host(self.scan.pts[-1])))
        if self.imu is not None:
            a, b = self.imu.t_range
            lo.append(float(self.al.imu_to_host(a)))
            hi.append(float(self.al.imu_to_host(b)))
        for t in self.takes:
            lo.append(t.t_beep_start)
            hi.append(t.t_beep_end)
        if lo:
            self.t_lo, self.t_hi = min(lo) - self.t_ref, max(hi) - self.t_ref

    # ------------------------------------------------------------------ alignment display
    def _manual_changed(self, *_):
        self.al.manual_video_ms, self.al.manual_imu_ms = self.sp_video.value(), self.sp_imu.value()
        self._update_span()
        self._redraw_timeline()
        self._show_summary()
        self._last_frame = -1
        self.set_time(self.t)

    def _redraw_timeline(self):
        for m in self._marks:
            self.tp.removeItem(m)
        self._marks = []

        def norm(x):
            x = np.asarray(x, float)
            return (x - np.percentile(x, 5)) / (np.percentile(x, 99.5) - np.percentile(x, 5) + 1e-9)
        if self.scan is not None:
            self.c_motion.setData(self.al.video_to_host(self.scan.pts) - self.t_ref, norm(self.scan.motion))
            for tv, kind in A.group_beeps(self.scan.tones):
                ln = pg.InfiniteLine(pos=float(self.al.video_to_host(tv)) - self.t_ref, angle=90,
                                     pen=pg.mkPen("#4dd0e1", width=2, style=Qt.DashLine))
                self._marks.append(ln)
        else:
            self.c_motion.setData([], [])
        if self.imu is not None:
            tg, wg = A.gyro_rate(self.imu)
            self.c_gyro.setData(self.al.imu_to_host(tg) - self.t_ref, norm(wg) - 1.1)
        else:
            self.c_gyro.setData([], [])
        for t in self.takes:
            reg = pg.LinearRegionItem((t.t_beep_start - self.t_ref, t.t_beep_end - self.t_ref), movable=False,
                                      brush=pg.mkBrush(30, 136, 229, 40))
            self._marks.append(reg)
            for tb in (t.t_beep_start, t.t_beep_end):
                self._marks.append(pg.InfiniteLine(pos=tb - self.t_ref, angle=90, pen=pg.mkPen("#ef5350", width=1.5)))
        for m in self._marks:
            self.tp.addItem(m)
        self.tp.setXRange(self.t_lo, self.t_hi, padding=0.01)

    def _show_summary(self):
        txt = A.fmt_alignment(self.al)
        det = getattr(self, "_phone_detected", None)
        meta = (getattr(self, "session_meta", None) or {}).get("meta", {})
        if det:
            txt += f"\nphone held {det} (from gravity)"
            if meta.get("phone_orientation") and meta.get("phone_orientation") != det:
                txt += f" — the session says {meta.get('phone_orientation')}!"
        if self.takes:
            miss = [t.run_id for t in self.takes if not self.radar.get(t.run_id)]
            txt += f"\ntakes in this recording: {len(self.takes)}"
            if miss:
                txt += f"\ncubes.h5 not on this computer: {', '.join(miss)} (copy them into runs/<run>/)"
        self.summary.setPlainText(txt)

    # ------------------------------------------------------------------ radar take
    def _take_selected(self, *_):
        rid = self.take_combo.currentData()
        r = self.radar.get(rid) if rid else None
        self._wf_take = (rid, r)
        if r is None:
            self.wf_img.clear()
            self.p_wf.setTitle("Radar: this take's cubes.h5 is not on this computer (copy runs/<run>/cubes.h5)"
                               if rid else "Radar: no take in this recording")
            return
        self.p_wf.setTitle("Radar: range over time (selected take)")
        p = np.abs(r["cube"]) ** 2
        db = 10 * np.log10(p.mean(axis=(1, 2)) + 1e-12)                  # (frames, range)
        near = r["range_m"] < 0.25
        db[:, near] = np.median(db)
        lo, hi = np.percentile(db, [5, 99.7])
        self.wf_img.setImage(db.T, autoLevels=False, levels=(lo, max(hi, lo + 1)))
        t = r["t_host"] - self.t_ref
        dr = float(r["range_m"][1] - r["range_m"][0]) if r["range_m"].size > 1 else 1.0
        self.wf_img.setRect(0, float(t[0]), dr * db.shape[1], float(t[-1] - t[0]) or 1.0)
        self.p_rd.setVisible(r["cube"].shape[1] > 1)
        tk = next((x for x in self.takes if x.run_id == rid), None)
        if tk is not None and not self.playing:
            self.set_time(tk.t_beep_start - self.t_ref)

    # ------------------------------------------------------------------ time
    def _slider_moved(self, v: int):
        if getattr(self, "_slider_lock", False):
            return
        self.set_time(self.t_lo + (self.t_hi - self.t_lo) * v / 1000.0)

    def set_time(self, t: float, from_cursor: bool = False):
        t = float(np.clip(t, self.t_lo, self.t_hi))
        self.t = t
        self._slider_lock = True
        self.slider.setValue(int(round(1000 * (t - self.t_lo) / max(1e-6, self.t_hi - self.t_lo))))
        self._slider_lock = False
        if not from_cursor:
            self.cursor.blockSignals(True)
            self.cursor.setValue(t)
            self.cursor.blockSignals(False)
        th = self.t_ref + t
        self.tlabel.setText(f"t = {t:+7.2f} s")
        self._show_video(th)
        self._show_rig(th)
        self._show_radar(th)

    def _show_video(self, th: float):
        if self.reader is None or self.scan is None:
            return
        tv = float(self.al.host_to_video(th))
        if tv < self.scan.pts[0] - 0.05 or tv > self.scan.pts[-1] + 0.05:
            self.video.setText("(outside the video)")
            self._last_frame = -1
            return
        i = int(np.clip(np.searchsorted(self.scan.pts, tv, side="right") - 1, 0, self.scan.n_frames - 1))
        if i == self._last_frame:
            return
        img = self.reader.frame(i)
        if img is not None:
            pm = QPixmap.fromImage(img).scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video.setPixmap(pm)
            self._last_frame = i

    def _show_rig(self, th: float):
        if self.imu is None:
            return
        att = A.attitude(self.imu, self.al, np.array([th]))
        q = att["quat"][0]
        R = A._quat_to_R(q) if np.isfinite(q).all() else None
        info = (f"pitch {att['pitch_deg'][0]:+5.1f}°   roll {att['roll_deg'][0]:+5.1f}°   "
                f"heading {att['yaw_deg'][0]:5.0f}°\nphone {getattr(self, '_phone_detected', '?')}, "
                f"radar pin-1 {orientation.pin1_corner(self.rig.rotation)}")
        self.rig.set_state(R, info)

    def _show_radar(self, th: float):
        rid, r = self._wf_take if self._wf_take else (None, None)
        if r is None:
            for c in self.prof:
                c.setData([], [])
            return
        self.wf_line.setValue(th - self.t_ref)
        t = r["t_host"]
        if th < t[0] - 0.1 or th > t[-1] + 0.1:
            for c in self.prof:
                c.setData([], [])
            self.p_now.setTitle("Radar frame now: (outside the take)")
            return
        k = int(np.clip(np.searchsorted(t, th), 0, t.size - 1))
        cube = r["cube"][k]
        prof = 10 * np.log10((np.abs(cube) ** 2).mean(axis=0) + 1e-12)
        for i, c in enumerate(self.prof):
            c.setData(r["range_m"], prof[i]) if i < prof.shape[0] else c.setData([], [])
        self.p_now.setTitle(f"Radar frame {k} (index {int(r['frame_index'][k])}), Δt {1000 * (t[k] - th):+.0f} ms")
        if cube.shape[0] > 1:
            rd = 10 * np.log10((np.abs(cube) ** 2).mean(axis=1) + 1e-12)
            rd[:, r["range_m"] < 0.25] = np.median(rd)
            lo, hi = np.percentile(rd, [20, 99.9])
            self.rd_img.setImage(rd.T, autoLevels=False, levels=(lo, max(hi, lo + 1)))
            va = r["v_axis"] if r["v_axis"] is not None else np.arange(cube.shape[0])
            dv = float(va[1] - va[0]) if va.size > 1 else 1.0
            dr = float(r["range_m"][1] - r["range_m"][0])
            self.rd_img.setRect(0, float(va[0]) - dv / 2, dr * rd.shape[1], dv * rd.shape[0])

    # ------------------------------------------------------------------ playback
    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.setText("❚❚" if self.playing else "▶")
        self._last_tick = time.monotonic()

    def step_frame(self, d: int):
        if self.scan is None:
            self.set_time(self.t + 0.033 * d)
            return
        tv = float(self.al.host_to_video(self.t_ref + self.t))
        i = int(np.clip(np.searchsorted(self.scan.pts, tv, side="right") - 1 + d, 0, self.scan.n_frames - 1))
        self.set_time(float(self.al.video_to_host(self.scan.pts[i])) - self.t_ref + 1e-4)

    def _tick(self):
        if self._job is not None:
            self.prog.setValue(int(100 * self._job["p"]))
            if self._job["done"]:
                self._load_finished()
            return
        now = time.monotonic()
        dt, self._last_tick = now - self._last_tick, now
        if self.playing:
            sp = float(self.speed.currentText().rstrip("×"))
            nt = self.t + dt * sp
            if nt >= self.t_hi:
                self.toggle_play()
                nt = self.t_hi
            self.set_time(nt)

    # ------------------------------------------------------------------ export
    def export(self):
        if self.scan is None and self.imu is None:
            QMessageBox.information(self, "Nothing loaded", "Load a phone recording first.")
            return
        stem = (self.takes[0].run_id if self.takes else (self.video_path or self.imu_path).stem).replace(" ", "_")
        default = self.runs_dir / "packs" / f"{stem}.tccpack.h5"
        default.parent.mkdir(parents=True, exist_ok=True)
        f, _ = QFileDialog.getSaveFileName(self, "Export pack", str(default), "TCC pack (*.tccpack.h5)")
        if not f:
            return
        self.export_to(Path(f))

    def export_to(self, path: Path) -> Path:
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            out = A.write_pack(path, self.scan, self.imu, self.al, self.takes, getattr(self, "session_meta", {}),
                               embed_video=self.embed.isChecked())
        finally:
            QApplication.restoreOverrideCursor()
        self.summary.appendPlainText(f"\nexported: {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return out

    def closeEvent(self, ev):
        if self.reader:
            self.reader.close()
        super().closeEvent(ev)


def run_aligner(argv=None) -> int:
    runs = default_runs_dir()
    setup_base(logging.WARNING, app_log_dir=runs)
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setStyle("Fusion")
    w = AlignWindow(runs)
    w.show()
    return app.exec()
