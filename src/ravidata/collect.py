"""TCC collection: a scripted, text-only recorder (protocol: ``docs/protocols/TCC-001-collection.md``).

``uv run ravidata collect --ctrl COM10 --env closed`` walks through the protocol step by step. For every step it
prints what to set up, waits for Enter, then records each configuration back to back (radar still, scene
still = a fair A/B). No plots and no detectors run while recording (``Engine(process=False)``), so a slow
laptop does not drop frames. Every recorded segment is one normal run directory (``cubes.h5`` + logs); the
index of all of them is ``runs/TCC_TAKES.csv``.

A beep marks the moment the radar delivers its first frame of each segment, and a double beep the end.
The phone video's audio track picks both up: that is the sync between video and radar for the moving takes.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

CONFIGS = ("tcc-1d-40mm", "tcc-2d-40mm")
CSV_NAME = "TCC_TAKES.csv"
CSV_COLS = ["time", "session", "env", "step", "test", "target", "r_m", "theta_deg", "x_m", "y_m", "config", "rep",
            "seconds", "frames", "fps", "index_gaps", "loss_pct", "status", "t_beep_start", "t_beep_end", "run_dir", "text"]
# What the operator measures once per session (stored in runs/TCC_SESSION_<id>.json and in every segment's events).
# Camera offsets: from the centre of the radar chip (antenna package) to the centre of the phone lens, seen from
# behind the rig: dx + = camera to the right, dy + = camera above, dz + = camera lens behind the radar face.
SESSION_FIELDS = [
    ("mount", "mount (stand / selfie-stick-tripod / handheld)", "selfie-stick-tripod"),
    ("radar_height_m", "radar chip height above the floor, m", ""),
    ("cam_dx_cm", "camera dx, cm (+ = right of the radar, seen from behind)", ""),
    ("cam_dy_cm", "camera dy, cm (+ = above the radar)", ""),
    ("cam_dz_cm", "camera dz, cm (+ = lens behind the radar face)", ""),
    ("radar_rotation_deg", "module pin-1 dot seen from the front: 0 = top left, 90 = top right, 180 = bottom right, "
     "270 = bottom left (ravidata/orientation.py)", "180"),        # the TCC rig (owner, 2026-11-13)
    ("phone_orientation", "phone: landscape / portrait", "portrait"),
    ("radar_orientation", "antenna pairs (the recorder fills this in from the rotation)", ""),
    ("place", "place (room / outdoor area, size)", ""),
    ("floor", "floor / ground surface", ""),
    ("phone", "phone + apps + video settings", "camera app (video + sound, portrait) + Sensor Logger (IMU only)"),
    ("notes", "anything else", ""),
]


@dataclass
class Step:
    sid: str                     # unique id, e.g. "T1-can-r2.0-a+30"
    test: str                    # T0 empty, T1 static precision, T2 dynamic, T3 angular separation, T4 bonus
    text: str                    # what the operator sets up before pressing Enter
    target: str = ""
    r_m: float | None = None     # polar position of the target (radar = origin, boresight = +y)
    theta_deg: float | None = None
    seconds: float = 5.0         # per configuration
    moving: bool = False         # the operator moves during the recording (countdown, then go on the beep)
    reps: int = 1                # moving takes: repetitions per configuration
    extra: dict = field(default_factory=dict)

    def xy(self) -> tuple[float, float] | None:
        if self.r_m is None or self.theta_deg is None:
            return None
        a = math.radians(self.theta_deg)
        return round(self.r_m * math.sin(a), 2), round(self.r_m * math.cos(a), 2)


def _pos_text(r: float, th: float) -> str:
    a = math.radians(th)
    return f"r = {r:.1f} m, angle {th:+.0f} deg -> floor mark x = {r * math.sin(a):+.2f} m, y = {r * math.cos(a):.2f} m"


def tcc_plan(env: str, t3: bool | None = None, bonus: bool = False, walk: bool = False) -> list[Step]:
    """The TCC-001 protocol as a list of steps. ``t3`` defaults to True outdoors ("open") only."""
    S: list[Step] = []
    S.append(Step(f"T0-empty", "T0", "Empty scene: nothing placed, stand at the parking spot (behind the radar).",
                  target="none", seconds=30))
    # T1 static precision: polar grid (TCC1 fig. 17a)
    grid = [(r, th) for r in (1.0, 2.0, 3.0) for th in (-30, 0, 30)] + [(1.5, -45), (1.5, 45)]
    for r, th in grid:
        S.append(Step(f"T1-can-r{r:.1f}-a{th:+d}", "T1", f"CAN (metal cylinder) at radar height. {_pos_text(r, th)}",
                      target="can", r_m=r, theta_deg=th))
    for r, th in [(1.0, 0), (2.0, 0), (3.0, 0), (2.0, -30), (2.0, 30)]:
        S.append(Step(f"T1-panel-r{r:.1f}-a{th:+d}", "T1", f"PANEL 50x50 cm facing the radar, centre at radar height. "
                      f"{_pos_text(r, th)}", target="panel", r_m=r, theta_deg=th))
    for r in (1.0, 2.0, 3.0):
        S.append(Step(f"T1-person-r{r:.1f}-a+0", "T1", f"PERSON standing still, facing the radar. {_pos_text(r, 0)} "
                      "(someone else presses Enter, or stand there within the countdown)",
                      target="person", r_m=r, theta_deg=0, moving=True, seconds=6))
    # T2 dynamic (TCC1 fig. 17b): the operator walks; video = truth
    S.append(Step("T2-radial", "T2", "PERSON walks from y = 3.5 m straight to the radar along boresight, stops at y = 0.5 m. "
                  "Start on the beep, steady pace.", target="person", moving=True, seconds=10, reps=3))
    S.append(Step("T2-transverse", "T2", "PERSON walks across at y = 2.0 m, from x = -2 m to x = +2 m. Start on the beep.",
                  target="person", moving=True, seconds=10, reps=3))
    # T3 angular separation (TCC1 fig. 17c): two cans at 2 m, symmetric about boresight
    if t3 if t3 is not None else env == "open":
        r = 2.0
        for dth in (30, 20, 15, 10, 5):
            half = math.radians(dth / 2)
            x, y = r * math.sin(half), r * math.cos(half)
            S.append(Step(f"T3-sep{dth:02d}", "T3", f"TWO CANS at r = 2.0 m, separation {dth} deg: marks at "
                          f"x = -{x:.2f} and x = +{x:.2f} m, y = {y:.2f} m (gap {2 * x:.2f} m).",
                          target="2 cans", r_m=r, theta_deg=0, extra={"sep_deg": dth, "gap_m": round(2 * x, 3)}))
    if walk:    # T5: the rig is carried (selfie stick at hip height, laptop carried); video + beeps = truth
        S.append(Step("T5-rig-empty", "T5", "Hold the stick still at walking height, pointing at open space "
                      "(nothing within 5 m ahead). Measures the rig's own reflections (hand, stick, phone).",
                      target="none", moving=True, seconds=10))
        for tgt, txt in [("wall", "a WALL"), ("panel", "the PANEL on its stand at y = 2.0 m"),
                         ("can", "the CAN on its stand at y = 2.0 m")]:
            S.append(Step(f"T5-walk-{tgt}", "T5", f"Walk with the rig toward {txt}: start 4 m away, steady pace, rig at "
                          "hip height, stop 0.5 m before it. Start on the beep.", target=tgt, moving=True,
                          seconds=12, reps=3))
        if bonus:
            S.append(Step("T5-walk-head", "T5", "Walk toward the HEAD-HEIGHT BOARD (bottom edge 1.7 m) from 4 m, stop "
                          "0.5 m before it. Start on the beep.", target="head-height board", moving=True, seconds=12,
                          reps=3, extra={"bottom_m": 1.7}))
    if bonus:   # product-relevant extras (head-height obstacle at the edge of the elevation FOV)
        for r in (1.0, 1.5):
            S.append(Step(f"T4-head-r{r:.1f}", "T4", f"BOARD hanging with its bottom edge at 1.7 m, on boresight at y = {r:.1f} m.",
                          target="head-height board", r_m=r, theta_deg=0, extra={"bottom_m": 1.7}))
    S.sort(key=lambda s: s.test)   # stable: T0..T4 (radar on the stand) first, T5 (rig carried) last
    return S


def _beep(n: int = 1):
    try:
        import winsound
        for _ in range(n):
            winsound.Beep(1200, 180)
            time.sleep(0.08)
    except Exception:
        sys.stdout.write("\a" * n)
        sys.stdout.flush()


class Collector:
    """Runs the steps. ``start(engine, cfg, label)`` starts one recording (SPI, UART or simulator)."""

    def __init__(self, start: Callable, env: str, runs_dir: Path | None = None, ask: Callable = input,
                 beep: Callable = _beep, out: Callable = print, first_frame_timeout_s: float = 15.0,
                 max_loss_pct: float = 15.0, configs=CONFIGS, countdown_s: int = 10,
                 session_id: str = "", session_meta: dict | None = None):
        from .engine import Engine
        from .session import default_runs_dir
        self.eng = Engine(runs_dir=runs_dir, process=False)
        self.start, self.env, self.ask, self.beep, self.out = start, env, ask, beep, out
        self.runs_dir = Path(runs_dir) if runs_dir else default_runs_dir()
        self.csv = self.runs_dir / CSV_NAME
        self.timeout = first_frame_timeout_s
        self.max_loss = max_loss_pct
        self.configs = configs
        self.countdown = countdown_s
        self.session_id = session_id
        self.session_meta = session_meta or {}

    # ---------------------------------------------------------------- one segment
    def record(self, st: Step, cfg_name: str, rep: int) -> dict:
        from .config import presets
        cfg = presets()[cfg_name]
        label = f"tcc_{self.env}_{st.sid}_{cfg_name}" + (f"_r{rep}" if st.reps > 1 else "")
        if st.moving:
            self.out(f"   {cfg_name}{f' rep {rep}' if st.reps > 1 else ''}: go to the start position, "
                     f"wait for the BEEP ({self.countdown} s + radar start-up)")
            for k in range(self.countdown, 0, -1):
                if k <= 3 or k % 5 == 0:
                    self.out(f"   {k}")
                time.sleep(1.0)
        self.start(self.eng, cfg, label)
        ses = self.eng.session
        xy = st.xy()
        meta = {"tcc_step": st.sid, "test": st.test, "env": self.env, "target": st.target, "r_m": st.r_m,
                "theta_deg": st.theta_deg, "x_m": xy[0] if xy else None, "y_m": xy[1] if xy else None,
                "rep": rep, "config": cfg_name, "session": self.session_id, "session_meta": self.session_meta,
                **st.extra}
        self.eng.note("tcc " + json.dumps(meta), kind="tcc_step")
        t0 = time.monotonic()
        while ses.frames == 0 and time.monotonic() - t0 < self.timeout and self.eng.state != "error":
            time.sleep(0.05)
        status = "ok"
        if ses.frames == 0:
            status = "no_frames"
        else:
            f0 = ses.frames
            gaps0 = (ses.source_stats or {}).get("index_gaps", 0) or 0
            self.eng.note("tcc start beep", kind="tcc_beep")
            tb0 = time.time()
            self.beep(1)
            t1 = time.monotonic()
            while time.monotonic() - t1 < st.seconds and self.eng.state != "error":
                time.sleep(0.05)
            tb1 = time.time()
            self.beep(2)
            self.eng.note("tcc end beep", kind="tcc_beep")
            dur = time.monotonic() - t1
        frames = ses.frames
        gaps = (ses.source_stats or {}).get("index_gaps", 0) or 0
        run_dir = str(ses.dir)
        self.eng.stop(f"tcc step {st.sid} done")
        if status == "ok":
            n, g = frames - f0, max(0, gaps - gaps0)
            loss = 100.0 * g / max(1, n + g)
            fps = n / max(dur, 1e-3)
            if loss > self.max_loss:
                status = "high_loss"
        else:
            n, g, loss, fps = 0, 0, 100.0, 0.0
            tb0 = tb1 = None
        row = {"time": datetime.now().isoformat(timespec="seconds"), "session": self.session_id, "env": self.env,
               "step": st.sid, "test": st.test,
               "target": st.target, "r_m": st.r_m, "theta_deg": st.theta_deg,
               "x_m": xy[0] if xy else "", "y_m": xy[1] if xy else "", "config": cfg_name, "rep": rep,
               "seconds": st.seconds, "frames": n, "fps": round(fps, 1), "index_gaps": g, "loss_pct": round(loss, 1),
               "status": status, "t_beep_start": round(tb0, 3) if tb0 else "", "t_beep_end": round(tb1, 3) if tb1 else "",
               "run_dir": run_dir, "text": st.text}
        self._write(row)
        return row

    def _write(self, row: dict):
        append_take_row(self.csv, row)

    # ---------------------------------------------------------------- the loop
    def run(self, steps: list[Step]) -> list[dict]:
        rows: list[dict] = []
        i = 0
        while i < len(steps):
            st = steps[i]
            self.out("")
            self.out(f"[{i + 1}/{len(steps)}] {self.env.upper()} {st.sid}")
            self.out(f"   {st.text}")
            cmd = self.ask("   Enter = record   s = skip   b = back one step   q = quit  > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                i += 1
                continue
            if cmd == "b":
                i = max(0, i - 1)
                continue
            step_rows = []
            for rep in range(1, st.reps + 1):
                order = self.configs if rep % 2 == 1 else tuple(reversed(self.configs))   # ABBA across reps
                for cfg_name in order:
                    r = self.record(st, cfg_name, rep)
                    step_rows.append(r)
                    self.out(f"   -> {cfg_name:12s} {r['status']:9s} {r['frames']:4d} frames  {r['fps']:5.1f} fps  "
                             f"loss {r['loss_pct']:4.1f} %")
                    if r["status"] == "no_frames":
                        self.out("   !! no frames: unplug/replug the radar USB (resets the radar), then repeat this step")
            rows += step_rows
            if any(r["status"] != "ok" for r in step_rows):
                if self.ask("   some segment failed: r = repeat this step, Enter = keep and go on  > ").strip().lower() == "r":
                    continue
            i += 1
        self.out(f"\nindex of all segments: {self.csv}")
        return rows


def append_take_row(csv_path: Path, row: dict) -> None:
    """Append one segment to the take index (shared by ``collect`` and the recorder GUI)."""
    csv_path = Path(csv_path)
    if csv_path.exists():   # older index without the newer columns: start a new file next to it
        with csv_path.open(encoding="utf-8") as f:
            head = f.readline().strip().split(",")
        if head != CSV_COLS:
            csv_path.rename(csv_path.with_name(f"TCC_TAKES_old_{datetime.now():%Y%m%d_%H%M%S}.csv"))
    new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLS})


def last_session_meta(runs_dir: Path) -> dict:
    """Measurements of the newest ``TCC_SESSION_*.json`` (defaults for the next session)."""
    olds = sorted(Path(runs_dir).glob("TCC_SESSION_*.json"))
    if not olds:
        return {}
    try:
        return json.loads(olds[-1].read_text(encoding="utf-8")).get("meta", {})
    except Exception:
        return {}


# What belongs to the rig (fixed once the rig is built) vs. to the session (place, floor, notes).
RIG_KEYS = ("mount", "radar_height_m", "cam_dx_cm", "cam_dy_cm", "cam_dz_cm", "radar_rotation_deg",
            "phone_orientation", "phone")
RIG_FILE = "RIG.json"


def load_rig(runs_dir: Path) -> dict:
    """The locked rig measurements (``runs/RIG.json``), {} if none."""
    p = Path(runs_dir) / RIG_FILE
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("rig", {}) if p.exists() else {}
    except Exception:
        return {}


def save_rig(runs_dir: Path, meta: dict) -> Path:
    """Write the rig part of the session measurements to ``runs/RIG.json`` (the recorder's Setup tab, locked)."""
    p = Path(runs_dir) / RIG_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    rig = {k: str(meta.get(k, "")) for k in RIG_KEYS}
    p.write_text(json.dumps({"saved": datetime.now().isoformat(timespec="seconds"), "rig": rig}, indent=1),
                 encoding="utf-8")
    return p


def save_session_meta(runs_dir: Path, env: str, meta: dict) -> str:
    """Write ``runs/TCC_SESSION_<id>.json``; returns the session id."""
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    sid = f"{datetime.now():%Y%m%d_%H%M%S}_{env}"
    (runs_dir / f"TCC_SESSION_{sid}.json").write_text(json.dumps({"session": sid, "env": env, "meta": meta}, indent=1),
                                                      encoding="utf-8")
    return sid


def ask_session_meta(runs_dir: Path, env: str, ask: Callable = input, out: Callable = print) -> tuple[str, dict]:
    """Ask the per-session measurements (Enter keeps the value from the last session) and save them."""
    prev = last_session_meta(runs_dir)
    out("Session measurements (Enter = keep the value in [brackets]):")
    meta = {}
    for key, label, default in SESSION_FIELDS:
        d = str(prev.get(key, default))
        v = ask(f"   {label} [{d}]: ").strip()
        meta[key] = v or d
    return save_session_meta(runs_dir, env, meta), meta


def main_collect(a) -> int:
    steps = tcc_plan(a.env, t3=True if a.t3 else None, bonus=a.bonus, walk=a.walk)
    if a.only:
        steps = [s for s in steps if s.test in set(a.only.split(","))]
    if a.start:
        ids = [s.sid for s in steps]
        if a.start not in ids:
            raise SystemExit(f"--start {a.start!r} not in plan; ids: {', '.join(ids)}")
        steps = steps[ids.index(a.start):]
    if a.list:
        for k, s in enumerate(steps, 1):
            print(f"{k:3d}  {s.sid:24s} {s.text}")
        print(f"\n{len(steps)} steps")
        return 0
    if a.sim:
        def start(eng, cfg, label):
            eng.start_sim(cfg, record=True)
    elif a.link == "spi":
        if not a.ctrl:
            raise SystemExit("--ctrl is required (or --sim)")
        from .cli_spi import int_bit_arg

        def start(eng, cfg, label):
            eng.start_spi(cfg, a.ctrl, a.spi_dev, int_bit_arg(a.int_bit), record=True, label=label)
    else:
        if not (a.ctrl and a.data):
            raise SystemExit("--ctrl and --data are required for the UART link")

        def start(eng, cfg, label):
            if cfg.link == "spi":
                print(f"   (skipping {cfg.name}: SPI-only config on the UART link)")
                raise SystemExit(1)
            eng.start_serial(cfg.copy(link="uart"), a.ctrl, a.data, record=True, label=label)
    configs = tuple(a.configs.split(",")) if a.configs else (CONFIGS if a.link == "spi" or a.sim else CONFIGS[:1])
    if a.seconds:
        for s in steps:
            s.seconds = a.seconds
    from .session import default_runs_dir
    sid, meta = ask_session_meta(default_runs_dir(), a.env)
    Collector(start, a.env, configs=configs, countdown_s=a.countdown, session_id=sid, session_meta=meta).run(steps)
    return 0
