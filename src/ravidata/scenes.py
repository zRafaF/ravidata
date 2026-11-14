"""Scenes of the free-course collection (D-033, docs/plans/2026-11-13-moving-radar-pivot.md) and the live check of
the sync hand swipe. Used by the recorder's Simple tab (``gui/guided.py``); no Qt here.

A *take* = one scene = one phone video. A scene has one or more parts (the static scene: empty → static → moving);
in every part each ticked radar config is recorded once per round, back to back (one recording per config, each with
its own beeps and its own row in ``runs/TCC_TAKES.csv``). The recorder waits for F9 between parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SYNC = "swipe your hand twice, 30–50 cm in front of the radar + camera"
SYNC_END = "swipe your hand twice again"

# configs offered for a take (only those that exist in config.presets() are shown); 1D + 2D are the minimum
DEFAULT_CONFIGS = ["tcc-1d-40mm", "tcc-2d-40mm-12fps", "tcc-2d-40mm"]     # the system's configs (2026-11-13)
SYNC_LEAD_S = 8.0                 # extra seconds at the start of a take's first recording for the hand swipe
COMPARE_CONFIGS = ["tcc-1d-40mm", "tcc-2d-40mm-12fps", "tcc-2d-40mm", "tcc-2d-40mm-k8", "fine-1d-15m",
                   "doppler-2d-40mm", "super-2d-30mm"]


@dataclass
class Phase:
    """One part of a take. Every ticked config is recorded once per round in each part."""
    pid: str                          # goes into TCC_TAKES.csv "test" (S0, S1, S2, D1 …)
    title: str
    seconds: float                    # default length per config (the Simple tab's box for this part); 0 = until F9
    setup: list[str] = field(default_factory=list)      # shown before the part starts (F9 to go on)
    during: list[str] = field(default_factory=list)     # what to do in every recording, after the sync swipe

    def text(self) -> str:
        return f"{self.title}. " + " ".join(self.during)


@dataclass
class Scene:
    sid: str                          # short id, goes into the take label (TCC_TAKES.csv "step")
    title: str
    radar: str                        # "still" | "carried"
    phases: list[Phase]
    env: str | None = None            # forced place ("open" outdoors), None = choose
    configs: list[str] | None = None  # default configs (None = DEFAULT_CONFIGS)


_GROUND = "Point the rig down at the 3 ground tags for ~3 s, then tilt up slowly."
_COURSE = ["3 ground tags laid out (an L or a triangle, 0.5–1 m apart, spacing measured).",
           "Course planned; 2–3 obstacles with tags if you have them."]

SCENES: list[Scene] = [
    Scene("S-static", "Radar still: empty → static → moving (one video)", "still", [
        Phase("S0", "Empty scene", 10,
              setup=["Radar still, pointing where the scene will be. Nothing in front of it; you stand behind the "
                     "rig, out of its view."],
              during=["Stay still behind the radar until the double beep."]),
        Phase("S1", "Static scene", 20,
              setup=["Place the objects: tags facing the camera, inside the picture.",
                     "Then step out of the radar's view (behind the rig)."],
              during=["Stay out of the radar's view and still."]),
        Phase("S2", "Things moving", 0,
              setup=["Broom / people ready; broom tags facing the camera."],
              during=["Walk around in front of the radar at several distances.",
                      "Sweep the broom across the field of view: near / far, left / right, low / high, slowly.",
                      "Keep the tags in the picture."]),
    ]),
    Scene("D1-carried-indoor", "Radar carried, indoors", "carried", [
        Phase("D1", "Course indoors", 0, setup=_COURSE,
              during=[_GROUND, "Walk the course at normal pace.", "At the end point down at the ground tags again."]),
    ], env="closed"),
    Scene("D2-carried-outdoor", "Radar carried, outdoors", "carried", [
        Phase("D2", "Course outdoors", 0,
              setup=_COURSE + ["Laptop shielded from the wind, volume at maximum (beeps)."],
              during=[_GROUND, "Walk the course at normal pace.", "At the end point down at the ground tags again."]),
    ], env="open"),
    Scene("S2-moving-only", "Radar still, things moving (part S2 alone)", "still", [
        Phase("S2", "Things moving", 0, setup=["Broom / people ready; broom tags facing the camera."],
              during=["Walk around in front of the radar; sweep the broom across the field of view."]),
    ]),
    Scene("M0-2d-loss-test", "2D frame-loss test", "still", [
        Phase("M0", "2D loss test", 10, setup=["Radar still; you walk in front of it during each recording."],
              during=["Walk back and forth in front of the radar."]),
    ], configs=["tcc-2d-40mm", "tcc-2d-40mm-12fps", "tcc-2d-40mm-k8"]),
    Scene("free", "Free recording", "still", [
        Phase("free", "Free", 0, setup=["Whatever you need; describe it in the notes."], during=["Do the scene."]),
    ]),
]


def scene(sid: str) -> Scene:
    return next(s for s in SCENES if s.sid == sid)


def take_queue(sc: Scene, configs: list[str], rounds: int, lengths: list[float] | None = None
               ) -> list[tuple[int, str, float]]:
    """(part index, config, seconds) in recording order: part by part, configs ABBA within each part.
    ``lengths``: seconds per config for each part (0 = until F9); default = the parts' own defaults."""
    out = []
    for i, ph in enumerate(sc.phases):
        secs = float(lengths[i]) if lengths is not None else ph.seconds
        out += [(i, c, secs) for c in config_order(configs, rounds)]
    return out


def config_order(configs: list[str], rounds: int) -> list[str]:
    """ABBA across rounds: [A, B], 2 rounds -> A B B A; 3 configs, 2 rounds -> A B C C B A."""
    out: list[str] = []
    for k in range(max(1, rounds)):
        out += configs if k % 2 == 0 else configs[::-1]
    return out


# ------------------------------------------------------------------------------------------------ sync swipe check
class SwipeDetector:
    """Live check that a hand moved close in front of the radar (the sync event).

    Feed one range profile (linear magnitude per bin) per update. The near-range energy (``r_min``..``r_max``,
    TX leakage excluded) is compared with its own quiet level (20th percentile so far); a rise of ``thresh_db``
    is an event. Events closer than ``refractory_s`` merge. Works for 1D and 2D (profile = |cube| averaged)."""

    def __init__(self, r_min: float = 0.25, r_max: float = 0.9, thresh_db: float = 10.0, refractory_s: float = 1.0):
        self.r_min, self.r_max, self.thresh_db, self.refractory_s = r_min, r_max, thresh_db, refractory_s
        self.e: list[float] = []
        self.events: list[float] = []       # seconds since the recording started
        self._in = False

    def update(self, t: float, r: np.ndarray, prof_lin: np.ndarray) -> bool:
        m = (r >= self.r_min) & (r <= self.r_max)
        if not m.any():
            return False
        e = float(np.sum(np.asarray(prof_lin, float)[m] ** 2)) + 1e-30
        self.e.append(e)
        if len(self.e) < 4:
            return False
        quiet = float(np.percentile(self.e, 20))
        hot = 10 * np.log10(e / quiet) >= self.thresh_db
        new = hot and not self._in and (not self.events or t - self.events[-1] >= self.refractory_s)
        if new:
            self.events.append(round(t, 2))
        self._in = hot
        return new

    def seen_start(self, within_s: float = 15.0) -> bool:
        return any(t <= within_s for t in self.events)


def profile_lin(cube: np.ndarray) -> np.ndarray:
    """(intervals/Doppler, RX, R) complex -> (R,) mean magnitude."""
    return np.abs(cube).mean(axis=tuple(range(cube.ndim - 1)))


def verdict(row: dict, swipes: list[float] | None, loss_warn: float = 12.5, within_s: float = SYNC_LEAD_S + 10
            ) -> tuple[str, str]:
    """('ok' | 'warn' | 'bad', reason) for one recording of a take."""
    if row.get("status") == "no_frames" or not row.get("frames"):
        return "bad", "no radar frames: unplug/replug the radar USB, then repeat"
    if row.get("status") != "ok":
        return "bad", f"{row.get('loss_pct')} % frames lost: repeat with a lighter config / close other programs"
    notes = []
    if float(row.get("loss_pct") or 0) > loss_warn:
        notes.append(f"{row.get('loss_pct')} % frames lost")
    if swipes is not None and not any(t <= within_s for t in swipes):
        notes.append("no sync swipe seen by the radar (swipe closer / repeat if the video matters)")
    return ("warn", "; ".join(notes)) if notes else ("ok", "")
