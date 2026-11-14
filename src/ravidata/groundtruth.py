"""Ground-truth takes (GT-001 protocol) and real-data scoring of all detectors.

A *take* is one repetition of one scene, delimited by two markers written from the GUI:

    take start {"scene": "S3", "rep": 1, "mode": "walk", "expect": "alarm", "obj": "...", "h_m": 1.7, "lat_m": 0.0}
    take end

Inside a take, the distance markers (Ctrl+1..8, text ``d=1.5 m``) mean:

* ``mode == "walk"``: the radar *passes* a floor mark at that distance from the obstacle. The truth range
  between two passes is interpolated linearly in time; before the first and after the last pass it is
  unknown (frames not scored for range, but still counted for false alarms in ``no_alarm`` takes).
* ``mode == "still"``: the radar *holds* at that distance until the next marker (or the take end).

Scores per take and detector (tolerance = max(tol_m, 2 range bins); walk takes use a wider default because
the key press lags the pass by ~0.1-0.3 s):

* ``alarm`` takes: per scored frame *hit* (|y - truth| <= tol), *clutter* (y nearer than truth - tol),
  *far* (y farther than truth + tol), *miss* (nothing). **Warning distance** = truth range at the first
  frame that starts ``k_confirm`` consecutive hits (0 if never). A take counts as *detected* when the
  warning distance >= ``warn_min_m``.
* ``no_alarm`` takes: *false-alarm %* = frames with an output nearer than ``alarm_max_m``, and false-alarm
  episodes per minute.

The owner's live labels (F4 "MISSED obstacle", F5 "FALSE alarm") are counted per take for reference.
Nothing here depends on synthetic injection: these numbers come from real obstacles only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TAKE_START = "take start"
TAKE_END = "take end"

# Scene catalogue for GT-001 (docs/protocols/GT-001-ground-truth-session.md). Kept here so the GUI,
# the scorer and the protocol use the same ids. Heights are obstacle BOTTOM edge above the floor.
SCENES = [
    {"id": "S0", "title": "empty: walk toward open space (nothing within 3.5 m ahead)", "expect": "no_alarm",
     "mode": "walk"},
    {"id": "S1", "title": "wall: walk straight at a flat wall", "expect": "alarm", "mode": "walk"},
    {"id": "S2", "title": "pole: thin vertical pole (broom handle) standing on the floor, straight ahead",
     "expect": "alarm", "mode": "walk"},
    {"id": "S3", "title": "head height: board ~30x40 cm hanging, bottom edge ~1.6-1.8 m, nothing below",
     "expect": "alarm", "mode": "walk", "h_m": 1.7},
    {"id": "S4", "title": "head height at the edge: S3 moved ~0.4 m sideways from the walking line",
     "expect": "alarm", "mode": "walk", "h_m": 1.7, "lat_m": 0.4},
    {"id": "S5", "title": "head height, low RCS: horizontal thin rod / branch / rope at ~1.7 m",
     "expect": "alarm", "mode": "walk", "h_m": 1.7},
    {"id": "S6", "title": "overhead, must NOT alarm: object with bottom edge >= 2.1 m (door frame top, high shelf)",
     "expect": "no_alarm", "mode": "walk", "h_m": 2.1},
    {"id": "S7", "title": "side, must NOT alarm: walk past a wall/object >= 0.8 m to the side",
     "expect": "no_alarm", "mode": "walk", "lat_m": 0.8},
    {"id": "S8", "title": "standing still 1.0 m in front of the S3 obstacle (10 s)", "expect": "alarm",
     "mode": "still", "h_m": 1.7},
]
SCENE_BY_ID = {s["id"]: s for s in SCENES}


def take_start_text(scene_id: str, rep: int = 1, **meta) -> str:
    base = {k: v for k, v in SCENE_BY_ID.get(scene_id, {}).items() if k not in ("title",)}
    base.update({"scene": scene_id, "rep": rep})
    base.pop("id", None)
    base.update({k: v for k, v in meta.items() if v not in (None, "")})
    return TAKE_START + " " + json.dumps(base, sort_keys=True)


@dataclass
class Take:
    run: str
    scene: str
    meta: dict
    i0: int
    i1: int                                   # exclusive
    dist_markers: list[tuple[int, float]] = field(default_factory=list)
    owner_missed: int = 0
    owner_false: int = 0

    @property
    def mode(self) -> str:
        return self.meta.get("mode", SCENE_BY_ID.get(self.scene, {}).get("mode", "walk"))

    @property
    def expect(self) -> str:
        return self.meta.get("expect", SCENE_BY_ID.get(self.scene, {}).get("expect", "alarm"))


def parse_takes(run: str, markers: list[dict], n_frames: int) -> list[Take]:
    """``markers``: evaluate.load_markers() output (sorted by frame here)."""
    mk = sorted([m for m in markers if m.get("frame") is not None], key=lambda m: m["frame"])
    takes: list[Take] = []
    cur: Take | None = None
    for m in mk:
        text = str(m.get("text", "")).strip()
        f = int(m["frame"])
        if text.lower().startswith(TAKE_START):
            if cur is not None:                       # unterminated take: close it here
                cur.i1 = f
                takes.append(cur)
            try:
                meta = json.loads(text[len(TAKE_START):].strip() or "{}")
            except Exception:
                meta = {"parse_error": text}
            cur = Take(run, str(meta.get("scene", "?")), meta, f, n_frames)
        elif text.lower().startswith(TAKE_END):
            if cur is not None:
                cur.i1 = f
                takes.append(cur)
                cur = None
        elif cur is not None:
            if m.get("dist") is not None and m.get("kind") == "marker":
                cur.dist_markers.append((f, float(m["dist"])))
            elif "missed" in text.lower():
                cur.owner_missed += 1
            elif "false" in text.lower():
                cur.owner_false += 1
    if cur is not None:
        cur.i1 = n_frames
        takes.append(cur)
    return [t for t in takes if t.i1 > t.i0]


def truth_range(take: Take, t: np.ndarray) -> np.ndarray:
    """Truth range per frame of the recording (nan outside the take or where unknown)."""
    out = np.full(len(t), np.nan)
    dm = [(i, d) for i, d in take.dist_markers if take.i0 <= i < min(take.i1, len(t))]
    if not dm:
        return out
    if take.mode == "still":
        for j, (i, d) in enumerate(dm):
            b = dm[j + 1][0] if j + 1 < len(dm) else take.i1
            out[i:min(b, len(t))] = d
        return out
    if len(dm) < 2:
        return out
    ti = np.array([t[i] for i, _ in dm])
    di = np.array([d for _, d in dm])
    a, b = dm[0][0], dm[-1][0]
    seg = np.arange(a, b + 1)
    out[seg] = np.interp(t[seg], ti, di)
    return out


@dataclass
class ScoreCfg:
    tol_walk_m: float = 0.25
    tol_still_m: float = 0.12
    k_confirm: int = 3
    warn_min_m: float = 1.0
    alarm_max_m: float = 2.5


def score_take(y: np.ndarray, take: Take, t: np.ndarray, bin_m: float, sc: ScoreCfg) -> dict:
    sl = slice(take.i0, min(take.i1, len(y)))
    yy, tt = y[sl], t[sl]
    dur = max(1e-6, float(tt[-1] - tt[0])) if len(tt) > 1 else 1e-6
    fps = (len(tt) - 1) / dur if len(tt) > 1 else float("nan")
    row = {"run": take.run, "scene": take.scene, "rep": take.meta.get("rep"), "expect": take.expect,
           "mode": take.mode, "frames": int(len(yy)), "dur_s": round(dur, 1), "fps": round(fps, 1),
           "owner_missed": take.owner_missed, "owner_false": take.owner_false}
    if take.expect == "no_alarm":
        fa = ~np.isnan(yy) & (yy <= sc.alarm_max_m)
        episodes = int(np.count_nonzero(np.diff(np.r_[0, fa.astype(int)]) == 1))
        row.update({"fa %": 100 * fa.mean() if len(fa) else float("nan"), "fa/min": 60 * episodes / dur})
        return row
    truth = truth_range(take, t)[sl]
    known = ~np.isnan(truth)
    tol = max(sc.tol_still_m if take.mode == "still" else sc.tol_walk_m, 2 * bin_m)
    if not known.any():
        row.update({"scored": 0, "note": "no usable distance markers"})
        return row
    yk, tk = yy[known], truth[known]
    has = ~np.isnan(yk)
    hit = has & (np.abs(yk - tk) <= tol)
    clutter = has & (yk < tk - tol)
    far = has & (yk > tk + tol)
    # warning distance: first frame that starts k consecutive hits (along the take, known frames only)
    warn = 0.0
    k = sc.k_confirm
    for i in range(0, len(hit) - k + 1):
        if hit[i:i + k].all():
            warn = float(tk[i])
            break
    row.update({"scored": int(known.sum()), "tol_m": round(tol, 2), "hit %": 100 * hit.mean(),
                "clutter %": 100 * clutter.mean(), "far %": 100 * far.mean(), "miss %": 100 * (~has).mean(),
                "warn_m": round(warn, 2), "detected": bool(warn >= sc.warn_min_m if take.mode == "walk"
                                                          else hit.mean() >= 0.8)})
    return row


def score_runs(run_dirs, detectors: dict[str, dict] | None = None, sc: ScoreCfg | None = None,
               progress=None) -> tuple[list[dict], list[Take]]:
    """Replay every run with takes through all (or the given) detectors; one row per (take, detector)."""
    from .evaluate import run_offline
    sc = sc or ScoreCfg()
    rows, all_takes = [], []
    for rd in run_dirs:
        rr = run_offline(rd, detectors, max_r=3.0, progress=progress)
        takes = parse_takes(rr.run, rr.markers, len(rr.t))
        all_takes += takes
        bin_m = float(rr.r_m[1] - rr.r_m[0]) if len(rr.r_m) > 1 else 0.1
        for tk in takes:
            for name, y in rr.outputs.items():
                r = score_take(y, tk, rr.t, bin_m, sc)
                r.update({"detector": name, "preset": rr.cfg.name})
                rows.append(r)
    return rows, all_takes


def _nanmean(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def leaderboard(rows: list[dict]) -> list[dict]:
    """One line per (preset, detector): real-obstacle detection and real false alarms."""
    out = []
    keys = sorted({(r["preset"], r["detector"]) for r in rows})
    for p, d in keys:
        rs = [r for r in rows if r["preset"] == p and r["detector"] == d]
        al = [r for r in rs if r["expect"] == "alarm" and r.get("scored")]
        na = [r for r in rs if r["expect"] == "no_alarm"]
        walk = [r for r in al if r["mode"] == "walk"]
        out.append({"preset": p, "detector": d,
                    "alarm takes": len(al), "detected": sum(1 for r in al if r.get("detected")),
                    "median warn m": float(np.median([r["warn_m"] for r in walk])) if walk else float("nan"),
                    "hit %": _nanmean([r.get("hit %") for r in al]),
                    "clutter %": _nanmean([r.get("clutter %") for r in al]),
                    "miss %": _nanmean([r.get("miss %") for r in al]),
                    "no-alarm takes": len(na), "fa %": _nanmean([r.get("fa %") for r in na]),
                    "fa/min": _nanmean([r.get("fa/min") for r in na])})
    out.sort(key=lambda r: (r["preset"], -(r["detected"]), -np.nan_to_num(r["median warn m"]),
                            np.nan_to_num(r["fa/min"], nan=1e9)))
    return out


def _fmt(v):
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"
    return str(v)


def report_md(rows: list[dict], takes: list[Take], sc: ScoreCfg) -> str:
    L = ["# Ground-truth score (real obstacles, GT-001)", "",
         f"Rules: tolerance max({sc.tol_walk_m} m walk / {sc.tol_still_m} m still, 2 bins); warning distance = truth "
         f"range at the first of {sc.k_confirm} consecutive hits; walk take 'detected' if warning >= "
         f"{sc.warn_min_m} m; false alarm = any output nearer than {sc.alarm_max_m} m in a must-not-alarm take. "
         "Defaults of every detector, no tuning on these data.", "",
         f"Takes: {len(takes)} ({sum(1 for t in takes if t.expect == 'alarm')} alarm, "
         f"{sum(1 for t in takes if t.expect == 'no_alarm')} must-not-alarm).", ""]
    lb = leaderboard(rows)
    cols = ["preset", "detector", "alarm takes", "detected", "median warn m", "hit %", "clutter %", "miss %",
            "no-alarm takes", "fa %", "fa/min"]
    L += ["## Leaderboard (per preset, sorted by detected takes, then warning distance, then false alarms)", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in lb:
        L.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    L += ["", "## Per scene (detected / takes, median warning m, false-alarm %)", ""]
    scenes = sorted({r["scene"] for r in rows})
    for p in sorted({r["preset"] for r in rows}):
        L += [f"### {p}", "", "| detector | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
        for d in sorted({r["detector"] for r in rows if r["preset"] == p}):
            cells = []
            for s in scenes:
                rs = [r for r in rows if r["preset"] == p and r["detector"] == d and r["scene"] == s]
                if not rs:
                    cells.append("")
                elif rs[0]["expect"] == "no_alarm":
                    cells.append(f"fa {_fmt(_nanmean([r.get('fa %') for r in rs]))} %")
                else:
                    sc_ = [r for r in rs if r.get("scored")]
                    cells.append(f"{sum(1 for r in sc_ if r.get('detected'))}/{len(sc_)}, "
                                 f"{_fmt(float(np.median([r['warn_m'] for r in sc_])) if sc_ else float('nan'))} m")
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Takes", "", "| run | scene | rep | expect | mode | frames | fps | dist markers | owner missed/false |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in takes:
        n = t.i1 - t.i0
        L.append(f"| {t.run} | {t.scene} | {t.meta.get('rep')} | {t.expect} | {t.mode} | {n} | – | "
                 f"{len(t.dist_markers)} | {t.owner_missed}/{t.owner_false} |")
    L += ["", "<details><summary>All rows (JSON)</summary>", "", "```json", json.dumps(rows, indent=0, default=str),
          "```", "</details>"]
    return "\n".join(L)
