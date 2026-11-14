"""Simple tab of the recorder: one guided *take* = one scene = one phone video.

Steps on screen: 1 scene, 2 configs + time per part, 3 checklist, 4 Start (F9). A scene has one or more parts (the
static scene: empty → static → moving). In each part every ticked config is recorded once per round, back to back;
between parts the recorder waits for F9 (time to place objects / get the broom), then counts down. A big banner
above the plots says what to do now, the radar checks the sync hand swipe, and each recording gets ✓ / ⚠ / ✗.
Each recording is a normal run + one row in runs/TCC_TAKES.csv: step = "<scene>@<HHMMSS>" (the take id),
test = the part (S0, S1, S2, D1 …), text = what was done + your notes + the sync lead-in and swipe times.

Sync: the beeps tie the video to the laptop clock (fitted over the whole video, so clock drift is included); the
hand swipe measures the constant sound-output delay once (EXP-014: 0.56 s, constant within ±24 ms over 3 min).
So it is done once per take, in a **sync lead-in** added in front of the first recording (its first N seconds
are the swipe, not the scene; the part's own time comes after it), and as a check at the end of the last one.

Changing config = restarting the radar link. The old link is fully closed before the next one opens (else the
COM port is still busy), configuring may take up to 30 s, and a config that gives no frames is retried once.

At the end a box asks for the take's name: ``runs/recs/<name>_<YYYYMMDD-HHMMSS>/`` is created with ``take.json``
(every recording of the take) and a README; copy the phone video + Sensor Logger zip there, then ``ravidata pack``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from .. import orientation
from ..collect import Step
from ..config import presets
from ..scenes import (COMPARE_CONFIGS, DEFAULT_CONFIGS, SCENES, SYNC, SYNC_END, SYNC_LEAD_S, SwipeDetector,
                      profile_lin, take_queue, verdict)

CHECKS = ["Sensor Logger is recording on the phone",
          "Camera is recording (original quality, 1:1, AE/AF locked)",
          "Scene ready: tags inside the picture, nobody in the way"]
COL = {"ok": "#81c784", "warn": "#ffb74d", "bad": "#ff5252"}
ICON = {"ok": "✓", "warn": "⚠", "bad": "✗"}
START_TIMEOUT_S = 30.0            # radar link open + config + first frame
FREE_TIMEOUT_S = 30.0             # wait for the previous link to close (a stuck config can take ~25 s)


def slug(name: str) -> str:
    """A folder-safe version of the take name (letters, digits, - and _)."""
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-_")
    return s or "take"


def describe(name: str) -> str:
    c = presets()[name]
    d = c.derive()
    kind = "2D" if c.fft_mode == 2 else "1D"
    return (f"{name}  —  {kind}, {d.bin_mm / 10:.1f} cm bins, {c.range_max_mm / 1000:.1f} m, "
            f"{d.expected_fps:.1f} fps")


def _thread_alive(src) -> bool:
    th = getattr(src, "_thread", None)
    return bool(th is not None and th.is_alive())


class GuidedPanel(QWidget):
    countdown_s = 5.0                 # after F9 between parts: time to get into position

    def __init__(self, win):
        super().__init__()
        self.win = win
        self.run: dict | None = None
        self.results: list[str] = []
        self.passes: dict[str, int] = {}      # scene id -> takes done in this window (default names)
        self.last_folder: Path | None = None
        self.part_secs: list[QDoubleSpinBox] = []
        v = QVBoxLayout(self)

        # 1. scene
        b1 = QGroupBox("1. Scene (one phone video)")
        f1 = QFormLayout(b1)
        self.scene = QComboBox()
        for s in SCENES:
            self.scene.addItem(s.title, s.sid)
        self.place = QComboBox()
        self.place.addItems(["closed", "open"])
        self.setup_lbl = QLabel()
        self.setup_lbl.setWordWrap(True)
        self.setup_lbl.setTextFormat(Qt.RichText)
        self.setup_lbl.setStyleSheet("color:#ffd54f;")
        f1.addRow("scene", self.scene)
        f1.addRow("place", self.place)
        f1.addRow(self.setup_lbl)
        v.addWidget(b1)

        # 2. configs + timing
        b2 = QGroupBox("2. Radar configs (ticked = recorded, one after the other, in every part) and time")
        f2 = QFormLayout(b2)
        self.cfg_list = QListWidget()
        self.cfg_list.setMaximumHeight(150)
        self.cfg_list.setToolTip("Each ticked config is one recording (its own beeps, its own run). In every part of "
                                 "the scene they are recorded back to back; rounds repeat the list in ABBA order.")
        for n in COMPARE_CONFIGS:
            if n in presets():
                it = QListWidgetItem(describe(n))
                it.setData(Qt.UserRole, n)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Unchecked)
                self.cfg_list.addItem(it)
        f2.addRow(self.cfg_list)
        self.time_box = QGroupBox("time per config in each part (0 = until F9)")
        self.time_form = QFormLayout(self.time_box)
        f2.addRow(self.time_box)
        self.lead = QDoubleSpinBox()
        self.lead.setRange(0, 30)
        self.lead.setDecimals(0)
        self.lead.setSuffix(" s")
        self.lead.setValue(SYNC_LEAD_S)
        self.lead.setToolTip("Added in front of the take's first recording for the hand swipe; the part's own time "
                             "starts after it. Marked in the data so the analysis skips it.")
        self.rounds = QSpinBox()
        self.rounds.setRange(1, 6)
        self.rounds.setToolTip("Repeat the config list (ABBA order): more, shorter segments per config.")
        self.notes = QLineEdit()
        self.notes.setPlaceholderText("notes: broom tags 20/21, objects, background …")
        f2.addRow("sync lead-in (1st recording)", self.lead)
        f2.addRow("rounds (A B, B A …)", self.rounds)
        f2.addRow("notes", self.notes)
        v.addWidget(b2)

        # 3. checklist
        b3 = QGroupBox("3. Before each take")
        v3 = QVBoxLayout(b3)
        self.checks = [QCheckBox(t) for t in CHECKS]
        for c in self.checks:
            c.toggled.connect(self._refresh)
            v3.addWidget(c)
        self.rig_lbl = QLabel()
        self.rig_lbl.setWordWrap(True)
        self.rig_lbl.setTextFormat(Qt.RichText)
        v3.addWidget(self.rig_lbl)
        v.addWidget(b3)

        # 4. go
        h = QHBoxLayout()
        self.btn = QPushButton()
        self.btn.setMinimumHeight(46)
        self.btn.clicked.connect(self.f9)
        self.btn_abort = QPushButton("■ Abort take")
        self.btn_abort.clicked.connect(lambda: self.finish("aborted"))
        self.btn_abort.setEnabled(False)
        h.addWidget(self.btn, 3)
        h.addWidget(self.btn_abort, 1)
        v.addLayout(h)
        self.res = QLabel("")
        self.res.setWordWrap(True)
        self.res.setTextFormat(Qt.RichText)
        v.addWidget(self.res)
        v.addStretch(1)

        # the banner: the window places it above the plots (readable from where you stand)
        self.now = QLabel("")
        self.now.setWordWrap(True)
        self.now.setTextFormat(Qt.RichText)
        fnt = QFont()
        fnt.setPointSize(15)
        self.now.setFont(fnt)
        self.now.setMinimumHeight(150)

        self.scene.currentIndexChanged.connect(self._scene_changed)
        self.cfg_list.itemChanged.connect(self._refresh)
        self._load()
        self._refresh()

    # ------------------------------------------------------------------ state / settings
    @property
    def busy(self) -> bool:
        return self.run is not None

    def _scene(self):
        return SCENES[self.scene.currentIndex()]

    def _scene_changed(self, *_):
        s = self._scene()
        parts = "".join(f"<br><b>{p.pid} {p.title}</b>: set up: {' '.join(p.setup)} Then: {' '.join(p.during)}"
                        for p in s.phases)
        self.setup_lbl.setText(f"Once per take, in the sync lead-in of the first recording: {SYNC}.{parts}"
                               + ("<br>F9 between parts." if len(s.phases) > 1 else ""))
        if s.env:
            self.place.setCurrentText(s.env)
        want = s.configs or DEFAULT_CONFIGS
        for i in range(self.cfg_list.count()):
            it = self.cfg_list.item(i)
            it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in want else Qt.Unchecked)
        self._build_times(s)
        self._refresh()

    def _build_times(self, s):
        """One time box per part of the scene, remembered per scene + part."""
        for sp in self.part_secs:
            self.time_form.removeRow(sp)
        self.part_secs = []
        st = self.win.settings
        for ph in s.phases:
            sp = QDoubleSpinBox()
            sp.setRange(0, 1800)
            sp.setDecimals(0)
            sp.setSuffix(" s")
            sp.setSpecialValueText("until F9")
            sp.setFocusPolicy(Qt.StrongFocus)
            nw = getattr(self.win, "_nowheel", None)
            if nw is not None:
                sp.installEventFilter(nw)
            saved = st.value(f"g_len/{s.sid}/{ph.pid}")
            try:
                sp.setValue(float(saved) if saved is not None else ph.seconds)
            except (TypeError, ValueError):
                sp.setValue(ph.seconds)
            sp.valueChanged.connect(lambda val, k=f"g_len/{s.sid}/{ph.pid}": st.setValue(k, val))
            self.time_form.addRow(f"{ph.pid} {ph.title}", sp)
            self.part_secs.append(sp)

    def lengths(self) -> list[float]:
        return [float(sp.value()) for sp in self.part_secs]

    def configs(self) -> list[str]:
        return [self.cfg_list.item(i).data(Qt.UserRole) for i in range(self.cfg_list.count())
                if self.cfg_list.item(i).checkState() == Qt.Checked]

    def _load(self):
        s = self.win.settings
        sid = s.value("g_scene")
        idx = next((i for i, x in enumerate(SCENES) if x.sid == sid), 0)
        self.scene.blockSignals(True)
        self.scene.setCurrentIndex(idx)
        self.scene.blockSignals(False)
        self._scene_changed()
        if s.value("g_place") in ("closed", "open") and not self._scene().env:
            self.place.setCurrentText(str(s.value("g_place")))
        try:
            self.lead.setValue(float(s.value("g_lead", SYNC_LEAD_S)))
        except (TypeError, ValueError):
            pass

    def save(self):
        s = self.win.settings
        s.setValue("g_scene", self._scene().sid)
        s.setValue("g_place", self.place.currentText())
        s.setValue("g_lead", self.lead.value())

    def _rig_text(self) -> str:
        w = self.win
        if not hasattr(w, "meta_edits"):
            return ""
        m = w._meta()
        rot = int(m.get("radar_rotation_deg") or 0)
        h = (m.get("radar_height_m") or "").strip()
        dx, dy, dz = (m.get(k) or "?" for k in ("cam_dx_cm", "cam_dy_cm", "cam_dz_cm"))
        locked = getattr(w, "rig_lock", None) is not None and w.rig_lock.isChecked()
        txt = (f"Rig: radar pin-1 {orientation.pin1_corner(rot)} ({rot}°), phone {m.get('phone_orientation')}, "
               f"camera dx/dy/dz {dx}/{dy}/{dz} cm, height {h + ' m' if h else '<b>not set</b>'}")
        col = "#81c784" if (locked and h) else "#ffb74d"
        tail = " — locked" if locked else " — <b>not locked</b> (Setup tab: check it, then tick 'Rig fixed')"
        return f"<span style='color:{col}'>{txt}{tail}</span>"

    def _refresh(self, *_):
        if self.run is not None:
            return
        missing = [c for c in self.checks if not c.isChecked()]
        cf = self.configs()
        self.btn.setEnabled(not missing and bool(cf))
        self.btn.setText("▶ Start take  (F9)" if not missing else f"tick the {len(missing)} box(es) above first")
        if not cf:
            self.btn.setText("tick at least one config")
        self.rig_lbl.setText(self._rig_text())
        if not self.now.text():
            self._say("Ready. Tick the boxes, then <b>Start take (F9)</b>. Keep the phone filming the whole take.")

    def _say(self, html: str, color: str = "#1e2a38"):
        self.now.setStyleSheet(f"background:{color}; color:#fff; padding:8px; border-radius:4px;")
        self.now.setText(html)

    def _lock(self, locked: bool):
        for wdg in (self.scene, self.place, self.cfg_list, self.lead, self.rounds, *self.part_secs, *self.checks):
            wdg.setEnabled(not locked)
        self.btn_abort.setEnabled(locked)

    def _btn(self, text: str, enabled: bool):
        self.btn.setText(text)
        self.btn.setEnabled(enabled)

    # ------------------------------------------------------------------ take control
    def f9(self):
        r = self.run
        if r is None:
            if self.btn.isEnabled():
                self.start()
        elif r["phase"] == "recording":
            if time.monotonic() - r["t"] < r["lead_now"]:
                r["lead_now"] = time.monotonic() - r["t"]       # F9 in the lead-in: the swipe is done, go on
            elif r["secs"] == 0:
                r["next"] = True
        elif r["phase"] == "ready":
            r["phase"], r["t"] = "countdown", time.monotonic()

    def start(self):
        w = self.win
        s = self._scene()
        if w.seq is not None or w.rec is not None:
            return
        take = f"{s.sid}@{time.strftime('%H%M%S')}"
        queue = take_queue(s, self.configs(), self.rounds.value(), self.lengths())
        w.env.setCurrentText(self.place.currentText())
        self.run = {"scene": s, "take": take, "queue": queue, "n": len(queue), "done": 0, "recorded": 0, "phase": "next", "t": 0.0,
                    "part": None, "rows": [], "next": False, "det": None, "cfg": None, "secs": 0.0, "item": None,
                    "retried": set(), "old_src": None, "lead": float(self.lead.value()), "lead_now": 0.0,
                    "started": time.strftime("%Y%m%d-%H%M%S"), "configs": self.configs(),
                    "rounds": self.rounds.value(), "lengths": dict(zip([p.pid for p in s.phases], self.lengths())),
                    "notes": self.notes.text().strip(), "place": self.place.currentText()}
        self.results = []
        self.res.setText("")
        self._lock(True)
        self._btn("…", False)
        self.save()

    def finish(self, msg: str):
        w = self.win
        r, self.run = self.run, None
        if r is not None and w.rec is not None:
            w.stop_record()
        self._lock(False)
        for c in self.checks[:2]:               # phone apps: confirm again for the next take
            c.setChecked(False)
        if r is None:
            self._refresh()
            return
        kinds = [e["kind"] for e in r["rows"]]
        first = next((e for e in r["rows"] if e["first"]), None)
        no_sync = first is not None and first["row"].get("frames") and not (first["swipes"] or [])
        if msg == "done" and kinds and all(k == "ok" for k in kinds):
            head, col = f"<b>TAKE OK</b>: {len(kinds)} recordings.", "#1b5e20"
        elif msg == "done" and "bad" not in kinds:
            head, col = "<b>TAKE usable with warnings</b>: see the list.", "#8d6e00"
        else:
            bad_cfgs = {e["cfg"] for e in r["rows"] if e["kind"] == "bad"}
            if msg == "done" and bad_cfgs:      # leave only the failed configs ticked: "repeat" is one F9
                for i in range(self.cfg_list.count()):
                    it = self.cfg_list.item(i)
                    it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in bad_cfgs else Qt.Unchecked)
            head = (f"<b>TAKE {msg.upper()}</b>: {len(bad_cfgs) or 'some'} config(s) failed; only those are still "
                    "ticked.")
            col = "#7f0000"
        sync = " <b>No sync swipe seen</b> (beeps only: ±0.6 s)." if no_sync else ""
        folder = self.save_take(r, msg) if r["rows"] else None
        where = f"<br>Folder <b>{folder.name}</b>: put the phone video + Sensor Logger zip there." if folder else ""
        self._say(f"{head}{sync}{where}<br>Stop the camera, then Sensor Logger. Another pass: change the scene, "
                  "restart the phone, tick, F9.", col)
        self._refresh()

    # ------------------------------------------------------------------ take folder
    def ask_name(self, default: str) -> str | None:
        """The take's name (None = cancelled: the default is used). Tests replace this."""
        text, ok = QInputDialog.getText(self, "Name this take",
                                        "Name of the take (a folder runs/recs/<name>_<date-time> is created for the "
                                        "phone video and the Sensor Logger zip):", text=default)
        return text if ok else None

    def save_take(self, r: dict, msg: str) -> Path | None:
        s = r["scene"]
        n = self.passes.get(s.sid, 0) + 1
        self.passes[s.sid] = n
        default = f"{s.sid.lower()}-pass{n}"
        name = (self.ask_name(default) or "").strip() or default
        recs = Path(self.win.runs_dir) / "recs"
        folder = recs / f"{slug(name)}_{r['started']}"
        k = 2
        while folder.exists():
            folder = recs / f"{slug(name)}_{r['started']}-{k}"
            k += 1
        try:
            folder.mkdir(parents=True)
            keep = ("time", "test", "config", "seconds", "frames", "fps", "index_gaps", "loss_pct", "status",
                    "t_beep_start", "t_beep_end", "run_dir")
            info = {
                "name": name, "folder": folder.name, "take_id": r["take"], "result": msg, "started": r["started"],
                "scene": s.sid, "scene_title": s.title, "parts": [p.pid for p in s.phases], "place": r["place"],
                "configs": r["configs"], "rounds": r["rounds"], "seconds_per_part": r["lengths"],
                "sync_lead_in_s": r["lead"], "notes": r["notes"], "session": self.win.session_id,
                "rig": {k: v for k, v in self.win._meta().items()} if hasattr(self.win, "meta_edits") else {},
                "recordings": [{**{k: e["row"].get(k) for k in keep}, "run": Path(str(e["row"].get("run_dir") or ""))
                                .name, "part": e["part"], "verdict": e["kind"], "why": e["why"],
                                "lead_in_s": e["lead"], "sync_swipes_s": e["swipes"]} for e in r["rows"]],
            }
            (folder / "take.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            lines = [f"# {name} ({s.title})", "",
                     f"Take `{r['take']}`, started {r['started']}, {msg}. Place: {r['place']}. Notes: {r['notes'] or '-'}",
                     f"The first recording starts with a {r['lead']:.0f} s sync lead-in (hand swipe), not part of "
                     "the scene.", "",
                     "Put here: the **original phone video** (USB, not WhatsApp) and the **Sensor Logger zip**.",
                     "Then: `uv run ravidata pack --folder runs/recs/" + folder.name + "`", "",
                     "| part | config | s | frames | loss % | verdict | run |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {e['part']} | {e['cfg']} | {e['row'].get('seconds', '')} | {e['row'].get('frames', 0)} | "
                      f"{e['row'].get('loss_pct', '')} | {e['kind']} {e['why']} | "
                      f"{Path(str(e['row'].get('run_dir') or '')).name} |" for e in r["rows"]]
            (folder / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as e:  # pragma: no cover
            self.results.append(f"<span style='color:#ff5252'>could not create {folder}: {e}</span>")
            return None
        self.last_folder = folder
        return folder

    # ------------------------------------------------------------------ periodic
    def _close_link(self, r: dict):
        """Stop the stream and remember the old source: the next one opens only once it has really closed."""
        w = self.win
        r["old_src"] = w.eng.source
        if w.eng.session is not None:
            w.stop_stream()
        r["t_free"] = time.monotonic()

    def _fail_or_retry(self, r: dict, why: str):
        item = r["item"]
        self._close_link(r)
        if item is not None and item[1] not in r["retried"]:
            r["retried"].add(item[1])
            r["queue"].insert(0, item)                     # the same config again, once
            self._say(f"<b>{item[1]}</b>: {why} — retrying once …", "#8d6e00")
            r["phase"] = "go"
        else:
            self._add(r["cfg"], {"status": "no_frames", "frames": 0}, None, s_part(r))
            r["done"] += 1
            r["phase"] = "next"

    def tick(self):
        r = self.run
        if r is None:
            return
        w = self.win
        now = time.monotonic()
        s = r["scene"]
        if r["phase"] == "next":
            if not r["queue"]:
                self.finish("done")
                return
            part = r["queue"][0][0]
            if part != r["part"]:                         # a new part: wait for F9 (the first one: count down)
                r["part"] = part
                ph = s.phases[part]
                if part == 0:
                    r["phase"], r["t"] = "countdown", now
                else:
                    r["phase"] = "ready"
                    self._say(f"<b>Part {part + 1}/{len(s.phases)}: {ph.pid} {ph.title}</b><br>Set up: "
                              f"{' '.join(ph.setup)}<br>Then press <b>F9</b> (recording starts {self.countdown_s:.0f} s "
                              "later). The phone keeps filming.", "#8d6e00")
                    self._btn(f"▶ Start part {part + 1}: {ph.title}  (F9)", True)
                    return
            else:
                r["phase"] = "go"
        if r["phase"] == "ready":
            return
        if r["phase"] == "countdown":
            ph = s.phases[r["part"]]
            left = self.countdown_s - (now - r["t"])
            self._btn("…", False)
            if left > 0:
                self._say(f"<b>{ph.pid} {ph.title}</b> starts in {left:.0f} s — get into position.<br>"
                          f"{' '.join(ph.during)}", "#37474f")
                return
            r["phase"] = "go"
        if r["phase"] == "go":
            item = r["queue"][0]
            cur = w.eng.cfg.name if (w.eng.session is not None and w.eng.cfg is not None) else None
            if cur != item[1] and w.eng.session is not None:
                self._close_link(r)
            old = r.get("old_src")
            if old is not None and _thread_alive(old) and now - r.get("t_free", now) < FREE_TIMEOUT_S:
                self._say(f"<b>{item[1]}</b>: waiting for the radar link to close …", "#37474f")
                return                                     # COM port still held by the previous link
            if old is not None and now - r.get("t_free", 0) < 0.5:
                return                                     # give Windows a moment to release the port
            r["old_src"] = None
            r["queue"].pop(0)
            r["item"] = item
            _, r["cfg"], r["secs"] = item
            k = r["done"] + 1
            if cur != r["cfg"] or w.eng.session is None:
                self._say(f"<b>{k}/{r['n']} {r['cfg']}</b>: starting the radar … keep going, the beep comes when "
                          "it records.", "#37474f")
                if not w.start_stream(presets()[r["cfg"]], label=f"{r['take']}_{r['cfg']}"):
                    self._fail_or_retry(r, "could not start")
                    return
            r["phase"], r["t"] = "wait_first", now
        if r["phase"] == "wait_first":
            ses = w.eng.session
            if ses is not None and ses.frames > 0:
                ph = s.phases[r["part"]]
                first = r["recorded"] == 0
                r["lead_now"] = r["lead"] if first else 0.0
                st = Step(sid=s.sid, test=ph.pid, text=ph.text(), seconds=r["secs"], moving=True)
                notes = self.notes.text().strip()
                w.start_record(step=st, rep=1, label=r["take"], env=self.place.currentText(),
                               text=ph.text() + (f" | notes: {notes}" if notes else ""))
                r["phase"], r["t"], r["next"] = "recording", now, False
                r["det"] = SwipeDetector()
                r["seen_seq"] = -1
                self._btn("⏭ Next config  (F9)" if r["secs"] == 0 else "recording …", r["secs"] == 0 or first)
            elif now - r["t"] > START_TIMEOUT_S or w.eng.state == "error":
                self._fail_or_retry(r, "no radar frames (link or config failed)")
            return
        if r["phase"] == "recording":
            self._recording(r, now)

    def _recording(self, r: dict, now: float):
        w = self.win
        ph = r["scene"].phases[r["part"]]
        k = r["done"] + 1
        first, last = r["recorded"] == 0, not r["queue"]
        el = now - r["t"]
        lead = r["lead_now"]
        fr = w.eng.latest_frame
        if fr is not None and fr.seq != r["seen_seq"]:
            r["seen_seq"] = fr.seq
            try:
                cfg = w.eng.cfg or presets()[r["cfg"]]
                prof = profile_lin(fr.cube)
                rr = np.arange(prof.size) * cfg.derive().bin_mm / 1000.0
                if r["det"].update(el, rr, prof):
                    w.eng.note(f"sync swipe seen at +{el:.1f} s", kind="marker")
            except Exception:  # pragma: no cover
                pass
        det = r["det"]
        if first:
            sw = ("<span style='color:#81c784'>sync swipe seen ✓ (" + ", ".join(f"+{t:.0f} s" for t in det.events[:4])
                  + ")</span>") if det.events else ("<span style='color:#ffb74d'>sync swipe not seen yet</span>"
                                                    if el < lead + 10 else
                                                    "<span style='color:#ff5252'>no sync swipe seen</span>")
        else:
            sw = ("<span style='color:#81c784'>sync done in the first recording ✓</span>" if r.get("synced") else
                  "<span style='color:#ffb74d'>no sync swipe in the first recording</span>")
        head = f"<b>● REC {k}/{r['n']} · {ph.pid} {ph.title} · {r['cfg']}</b>"
        if el < lead:
            self._say(f"{head}<br><b>SYNC: {SYNC}</b> ({lead - el:.0f} s; F9 when done)<br>"
                      f"Then: {' '.join(ph.during)}<br>{sw}", "#4a148c")
        else:
            pe = el - lead
            end = f" ({SYNC_END} first, as a check)" if last else ""
            left = (f"{pe:.0f} s — press <b>F9</b> when done{end}" if r["secs"] == 0 else
                    f"{pe:.0f} / {r['secs']:.0f} s" + (f" — {SYNC_END} now (check)" if last and r["secs"] - pe < 5
                                                        else ""))
            self._say(f"{head} &nbsp; {left}<br>Now: {' '.join(ph.during)}<br>{sw}", "#b71c1c")
            if first:
                self._btn("⏭ Next config  (F9)" if r["secs"] == 0 else "recording …", r["secs"] == 0)
        if el >= lead and (r["next"] if r["secs"] == 0 else el - lead >= r["secs"]):
            if w.rec is not None:
                extra = []
                if lead > 0:
                    extra.append(f"sync lead-in: first {lead:.1f} s")
                if det.events:
                    extra.append("sync swipes at " + ", ".join(f"+{t:.1f}s" for t in det.events))
                if extra:
                    w.rec["meta"]["text"] += " | " + " | ".join(extra)
            row = w.stop_record() or {"status": "?"}
            if first:
                r["synced"] = any(t <= lead + 10 for t in det.events)
            self._add(r["cfg"], row, det.events, ph.pid, first, lead)
            r["done"] += 1
            r["recorded"] += 1
            r["phase"] = "next"
            self._btn("…", False)

    def _add(self, cfg: str, row: dict, swipes, part: str = "", first: bool = False, lead: float = 0.0):
        kind, why = verdict(row, swipes if first else None, within_s=lead + 10)   # swipe: first recording only
        self.run["rows"].append({"kind": kind, "why": why, "cfg": cfg, "part": part, "row": row, "first": first,
                                 "swipes": list(swipes or []), "lead": lead})
        line = (f"<span style='color:{COL[kind]}'>{ICON[kind]} {part + ' · ' if part else ''}{cfg}: "
                f"{row.get('frames', 0)} frames, {row.get('fps', '-')} fps, loss {row.get('loss_pct', '-')} %</span>"
                + (f" — {why}" if why else ""))
        self.results.append(line)
        self.res.setText("<br>".join(self.results))


def s_part(r: dict) -> str:
    return r["scene"].phases[r["part"]].pid if r.get("part") is not None else ""
