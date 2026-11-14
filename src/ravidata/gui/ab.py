"""Blind A/B review of two detectors on the same recording: ``uv run ravidata ab [RUN ...]``.

Both candidates are replayed offline on identical data. The range-time image (raw power) is shown with
each candidate's reported range as coloured dots, labelled only X and Y (which one is A is hidden
until you press Reveal). You step through fixed-length segments and vote; votes are appended to
``runs/AB_VOTES.jsonl`` and summarised by ``ravidata ab-report``.

Distance markers (Ctrl+1..8 in the lab GUI) are drawn as dashed lines at their range, so a vote can use
the known distance; the proxy metrics of ``ravidata compare`` are shown after Reveal.

Votes are a quick, human first filter. They do not replace scored runs with markers (EXP-004).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QMainWindow, QMessageBox, QPushButton, QSplitter, QTextEdit,
                               QVBoxLayout, QWidget)

from ..evaluate import metrics, run_offline
from ..processing.detectors import DETECTORS
from ..session import default_runs_dir

VOTES = "AB_VOTES.jsonl"
COL = {"X": "#ffa726", "Y": "#29b6f6"}


def list_runs(runs_dir: Path, min_frames: int = 100) -> list[Path]:
    out = []
    for d in sorted(runs_dir.iterdir(), reverse=True):
        h = d / "cubes.h5"
        if h.exists() and h.stat().st_size > 50_000:
            out.append(d)
    return out


class AbWindow(QMainWindow):
    def __init__(self, runs: list[Path], a: str, b: str, seg_s: float, blind: bool):
        super().__init__()
        self.setWindowTitle("ravidata — blind A/B review")
        self.resize(1600, 950)
        self.runs_dir = default_runs_dir()
        self.blind = blind
        self.rr = None
        self.map = {"X": "A", "Y": "B"}
        self.revealed = not blind
        self.seg = 0
        pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

        split = QSplitter()
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addWidget(QLabel("<b>Recording</b>"))
        self.lst = QListWidget()
        for d in runs:
            self.lst.addItem(d.name)
        lv.addWidget(self.lst, 1)
        self.cb = {}
        self.ed = {}
        for k, default in (("A", a), ("B", b)):
            lv.addWidget(QLabel(f"<b>Candidate {k}</b> (detector + JSON parameter overrides)"))
            cb = QComboBox()
            cb.addItems(sorted(DETECTORS))
            cb.setCurrentText(default)
            ed = QLineEdit()
            ed.setPlaceholderText('e.g. {"tau_s": 0.3, "report": "strongest"}')
            lv.addWidget(cb)
            lv.addWidget(ed)
            self.cb[k], self.ed[k] = cb, ed
        hb = QHBoxLayout()
        hb.addWidget(QLabel("segment [s]"))
        self.sp_seg = QDoubleSpinBox()
        self.sp_seg.setRange(2, 120)
        self.sp_seg.setValue(seg_s)
        hb.addWidget(self.sp_seg)
        lv.addLayout(hb)
        self.b_run = QPushButton("Run both on the selected recording")
        self.b_run.clicked.connect(self._run)
        lv.addWidget(self.b_run)
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        lv.addWidget(self.info, 2)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.title = QLabel("pick a recording and press Run")
        rv.addWidget(self.title)
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "time [s]")
        self.plot.setLabel("left", "range [m]")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("CET-L1"))
        self.plot.addItem(self.img)
        self.sc = {k: pg.ScatterPlotItem(size=4, pen=None, brush=pg.mkBrush(COL[k])) for k in ("X", "Y")}
        for k, s in self.sc.items():
            self.plot.addItem(s)
        self.region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(255, 255, 255, 25))
        self.plot.addItem(self.region)
        self.legend = self.plot.addLegend()
        rv.addWidget(self.plot, 1)
        vb = QHBoxLayout()
        for text, fn in (("◀ prev [PgUp]", lambda: self._go(-1)), ("next [PgDn] ▶", lambda: self._go(1))):
            b = QPushButton(text)
            b.clicked.connect(fn)
            vb.addWidget(b)
        for key, (text, vote) in {"F1": ("X better [F1]", "X"), "F2": ("Y better [F2]", "Y"),
                                  "F3": ("tie / both fine [F3]", "tie"), "F4": ("both bad [F4]", "both_bad")}.items():
            b = QPushButton(text)
            if vote in COL:
                b.setStyleSheet(f"background:{COL[vote]};color:black;font-weight:bold")
            b.clicked.connect(lambda _=False, v=vote: self._vote(v))
            vb.addWidget(b)
            QShortcut(QKeySequence(key), self, activated=lambda v=vote: self._vote(v))
        self.b_reveal = QPushButton("Reveal")
        self.b_reveal.clicked.connect(self._reveal)
        vb.addWidget(self.b_reveal)
        rv.addLayout(vb)
        QShortcut(QKeySequence("PgDown"), self, activated=lambda: self._go(1))
        QShortcut(QKeySequence("PgUp"), self, activated=lambda: self._go(-1))
        self.status = QLabel("")
        rv.addWidget(self.status)
        split.addWidget(right)
        split.setSizes([380, 1220])
        self.setCentralWidget(split)
        if self.lst.count():
            self.lst.setCurrentRow(0)

    # ------------------------------------------------------------------ run
    def _spec(self, k):
        try:
            extra = json.loads(self.ed[k].text() or "{}")
        except Exception as e:
            raise ValueError(f"candidate {k}: parameters are not valid JSON ({e})")
        return {"kind": self.cb[k].currentText(), **extra}

    def _run(self):
        it = self.lst.currentItem()
        if it is None:
            return
        try:
            specs = {"A": self._spec("A"), "B": self._spec("B")}
        except ValueError as e:
            QMessageBox.warning(self, "A/B", str(e))
            return
        self.status.setText("replaying… (both candidates on identical data)")
        QApplication.processEvents()
        t0 = time.time()
        try:
            self.rr = run_offline(self.runs_dir / it.text(), detectors=specs,
                                  progress=lambda i, n: (self.status.setText(f"replaying {i}/{n}"),
                                                         QApplication.processEvents()))
        except Exception as e:
            QMessageBox.critical(self, "A/B", f"replay failed: {e}")
            return
        self.specs = specs
        self.map = {"X": "A", "Y": "B"}
        if self.blind and random.random() < 0.5:
            self.map = {"X": "B", "Y": "A"}
        self.revealed = not self.blind
        self.seg = 0
        self._draw_all()
        self._go(0)
        self.status.setText(f"done in {time.time() - t0:.1f} s. Vote per segment; F1 = X better, F2 = Y better.")

    def _draw_all(self):
        rr = self.rr
        t = rr.t - rr.t[0]
        P = rr.power_db
        lo, hi = np.percentile(P, 5), np.percentile(P, 99.7)
        self.img.setImage(P.T, levels=(lo, hi))
        R = rr.r_m[-1] + (rr.r_m[1] - rr.r_m[0])
        self.img.setRect(0, 0, float(t[-1] if len(t) else 1), float(R))
        self.legend.clear()
        for k in ("X", "Y"):
            y = rr.outputs[self.map[k]]
            m = ~np.isnan(y)
            self.sc[k].setData(t[m], y[m])
            self.legend.addItem(self.sc[k], k if not self.revealed else f"{k} = {self._label(self.map[k])}")
        for it in getattr(self, "_mk_items", []):
            self.plot.removeItem(it)
        self._mk_items = []
        for mk in rr.markers:
            f = mk.get("frame")
            if f is None or f >= len(t):
                continue
            ln = pg.InfiniteLine(pos=float(t[int(f)]), angle=90, pen=pg.mkPen("#bbb", style=Qt.DotLine),
                                 label=str(mk["text"])[:18], labelOpts={"position": 0.95, "color": "#ddd"})
            self.plot.addItem(ln)
            self._mk_items.append(ln)
            if mk.get("dist") is not None:
                h = pg.PlotDataItem([t[int(f)], t[int(f)] + 4], [mk["dist"]] * 2, pen=pg.mkPen("#ff4081", width=2,
                                                                                             style=Qt.DashLine))
                self.plot.addItem(h)
                self._mk_items.append(h)
        self.plot.setYRange(0, min(3.5, R))
        self.title.setText(f"<b>{rr.run}</b> — {len(t)} frames, {t[-1]:.0f} s, preset {rr.cfg.name}")
        self._info()

    def _label(self, ab):
        s = dict(self.specs[ab])
        kind = s.pop("kind")
        return kind + (f" {json.dumps(s)}" if s else "")

    def _segments(self):
        t = self.rr.t - self.rr.t[0]
        L = self.sp_seg.value()
        n = max(1, int(np.ceil((t[-1] + 1e-9) / L)))
        return [(i * L, min((i + 1) * L, t[-1])) for i in range(n)]

    def _go(self, d):
        if self.rr is None:
            return
        segs = self._segments()
        self.seg = int(np.clip(self.seg + d, 0, len(segs) - 1))
        a, b = segs[self.seg]
        self.region.setRegion((a, b))
        self.plot.setXRange(max(0, a - 0.1 * (b - a)), b + 0.1 * (b - a), padding=0)
        voted = self._votes_for_run()
        v = voted.get(round(a, 2))
        self.status.setText(f"segment {self.seg + 1}/{len(segs)}  [{a:.1f}–{b:.1f} s]"
                            + (f"  — your vote: {v}" if v else ""))

    # ------------------------------------------------------------------ votes
    def _votes_for_run(self) -> dict:
        p = self.runs_dir / VOTES
        out = {}
        if not p.exists() or self.rr is None:
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                v = json.loads(line)
            except Exception:
                continue
            if v.get("run") == self.rr.run and v.get("a") == self.specs["A"] and v.get("b") == self.specs["B"]:
                out[round(v["seg_start"], 2)] = v["vote"]
        return out

    def _vote(self, vote):
        if self.rr is None:
            return
        a, b = self._segments()[self.seg]
        winner = {"X": self.map["X"], "Y": self.map["Y"]}.get(vote, vote)
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": self.rr.run, "preset": self.rr.cfg.name,
               "seg_start": round(a, 2), "seg_end": round(b, 2), "a": self.specs["A"], "b": self.specs["B"],
               "shown_as": self.map, "blind": self.blind and not self.revealed, "vote": vote, "winner": winner}
        with open(self.runs_dir / VOTES, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self._info()
        self._go(1)

    def _reveal(self):
        if self.rr is None:
            return
        self.revealed = True
        self._draw_all()
        self._go(0)

    def _info(self):
        if self.rr is None:
            return
        votes = [v for v in self._votes_for_run().values()]
        lines = [f"<b>votes on this run</b>: {len(votes)} segments"]
        if self.revealed:
            wins = {"A": 0, "B": 0, "tie": 0, "both_bad": 0}
            p = self.runs_dir / VOTES
            for line in p.read_text(encoding="utf-8").splitlines() if p.exists() else []:
                v = json.loads(line)
                if v.get("run") == self.rr.run and v.get("a") == self.specs["A"] and v.get("b") == self.specs["B"]:
                    wins[v["winner"]] = wins.get(v["winner"], 0) + 1
            lines.append(f"A = {self._label('A')}<br>B = {self._label('B')}")
            lines.append(f"A wins {wins['A']}, B wins {wins['B']}, ties {wins['tie']}, both bad {wins['both_bad']}")
            lines.append("<b>proxy metrics</b> (no ground truth unless distance markers exist):")
            for k in ("A", "B"):
                m = metrics(self.rr, k)
                lines.append(f"{k}: " + ", ".join(f"{n} {v:.0f}" if isinstance(v, float) else f"{n} {v}"
                                                   for n, v in m.items()))
        else:
            lines.append("identities hidden — press Reveal when done")
        self.info.setHtml("<br>".join(lines))


def run_ab(runs: list[str] | None = None, a: str = "track", b: str = "cfar", seg_s: float = 10.0, blind: bool = True):
    from ..processing.pipeline import DETECTORS
    a = a if a in DETECTORS else "threshold"          # the default "track" is a private add-on
    b = b if b in DETECTORS else "cfar"
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    pg.setConfigOption("background", "#1a1a1a")
    rd = default_runs_dir()
    paths = [Path(r) if Path(r).is_absolute() else rd / r for r in runs] if runs else list_runs(rd)
    w = AbWindow([p for p in paths if (p / "cubes.h5").exists()], a, b, seg_s, blind)
    w.show()
    return app.exec()


def ab_report(runs_dir: Path | None = None) -> str:
    p = (runs_dir or default_runs_dir()) / VOTES
    if not p.exists():
        return "no votes yet (runs/AB_VOTES.jsonl)"
    agg: dict[tuple, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            v = json.loads(line)
        except Exception:
            continue
        key = (json.dumps(v["a"], sort_keys=True), json.dumps(v["b"], sort_keys=True))
        d = agg.setdefault(key, {"A": 0, "B": 0, "tie": 0, "both_bad": 0, "runs": set()})
        d[v["winner"]] = d.get(v["winner"], 0) + 1
        d["runs"].add(v["run"])
    L = ["| A | B | A wins | B wins | ties | both bad | runs |", "|---|---|---|---|---|---|---|"]
    for (a, b), d in agg.items():
        L.append(f"| `{a}` | `{b}` | {d['A']} | {d['B']} | {d['tie']} | {d['both_bad']} | {len(d['runs'])} |")
    return "\n".join(L)
