"""Offline evaluation: replay recordings through detectors and compute comparison metrics.

Used by ``ravidata compare`` (table for agents/owner) and ``ravidata ab`` (blind A/B review GUI).

Without ground truth the metrics are **proxies** and must be read as such:

* ``det %``      frames with an output (an obstacle reported).
* ``=strong %``  output within max(0.1 m, 2 bins) of the strongest return in [min_r, max_r] (in wall
                 runs this is usually the wall; elsewhere it may be clutter).
* ``nearer %``   output nearer than the strongest return by more than that tolerance (clutter proxy).
* ``jitter cm``  median |change| of the reported range between consecutive frames with output.
* ``flips/min``  on/off changes of the output per minute (flicker).
* ``truth``      when the run has markers like ``d=1.5`` / ``1.5 m`` (EXP-004), frames within the
                 marker's hold are scored against that distance: ``hit %`` within the tolerance,
                 ``near-miss``, ``clutter`` (nearer), ``miss`` (nothing).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import RadarConfig
from .cube import CubeShape
from .frames import Frame
from .processing.pipeline import Processor
from .sources import open_recording

_DIST = re.compile(r"(?:^|[^\w.])(?:d\s*=\s*)?(\d+(?:[.,]\d+)?)\s*(?:m\b|$)", re.I)


@dataclass
class RunResult:
    run: str
    cfg: RadarConfig
    t: np.ndarray                               # (N,) s
    r_m: np.ndarray                             # (R,)
    power_db: np.ndarray                        # (N, R) combined raw power (for display)
    dyn_db: np.ndarray                          # (N, R) dynamic channel (for display)
    strongest: np.ndarray                       # (N,) range of the strongest return in the window
    outputs: dict[str, np.ndarray]              # detector -> (N,) reported range (nan = nothing)
    markers: list[dict] = field(default_factory=list)   # {"t","frame","text","dist"}
    params: dict = field(default_factory=dict)


def load_markers(run_dir: Path) -> list[dict]:
    ev = Path(run_dir) / "events.jsonl"
    out = []
    if not ev.exists():
        return out
    for line in ev.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") in ("marker", "note") and "text" in e:
            m = _DIST.search(str(e["text"]))
            out.append({"t": e.get("t"), "frame": e.get("frame"), "text": e["text"],
                        "dist": float(m.group(1).replace(",", ".")) if m else None, "kind": e["type"]})
    return out


def run_offline(path: str | Path, detectors: dict[str, dict] | None = None, preprocess: dict | None = None,
                min_r: float = 0.25, max_r: float = 3.0, progress=None) -> RunResult:
    """Replay a recording through one Processor. ``detectors``: {output_name: {"kind": registered name,
    **params}}; the same registered detector can appear several times with different parameters."""
    rec = open_recording(path)
    proc = Processor(rec.cfg)
    for k, v in (preprocess or {}).items():
        proc.pre.set(k, v)
    if detectors:
        from .processing.detectors import DETECTORS
        dets = {}
        for name, spec in detectors.items():
            spec = dict(spec)
            kind = spec.pop("kind", name)
            dets[name] = DETECTORS[kind](**spec)
        proc.detectors = dets
        proc.primary = next(iter(dets))
    names = list(proc.detectors)
    N = rec.n
    outs = {n: np.full(N, np.nan) for n in names}
    pdb, ddb, strong, ts = [], [], np.full(N, np.nan), np.zeros(N)
    r_m = None
    for i in range(N):
        c = rec.get_cube(i)
        fr = Frame(i, int(rec.frame_index[i]), 0.0, float(rec.times[i]), c, CubeShape(*c.shape),
                   extra={"t_rec": float(rec.times[i]), "replay_pos": i})
        res = proc.process(fr)
        f = res.features
        r_m = f.r_m
        ts[i] = rec.times[i]
        pdb.append(f.p_db.astype(np.float32))
        ddb.append((10 * np.log10(np.maximum(f.p_dyn, 1e-12))).astype(np.float32))
        w = (f.r_m >= min_r) & (f.r_m <= max_r)
        if w.any():
            k = np.flatnonzero(w)
            strong[i] = f.r_m[k[np.argmax(f.p_lin[k])]]
        for n, o in res.outputs.items():
            if o.nearest is not None:
                outs[n][i] = o.nearest.range_m
        if progress and i % 200 == 0:
            progress(i, N)
    run_dir = Path(rec.path).parent
    return RunResult(run_dir.name, rec.cfg, ts, r_m, np.array(pdb), np.array(ddb), strong, outs,
                     load_markers(run_dir), proc.params_snapshot())


def metrics(rr: RunResult, name: str) -> dict:
    y = rr.outputs[name]
    has = ~np.isnan(y)
    N = len(y)
    dur = max(1e-6, rr.t[-1] - rr.t[0]) if N > 1 else 1.0
    bin_m = float(rr.r_m[1] - rr.r_m[0]) if len(rr.r_m) > 1 else 0.1
    tol = max(0.1, 2 * bin_m)
    both = has & ~np.isnan(rr.strongest)
    eq = both & (np.abs(y - rr.strongest) <= tol)
    nearer = both & (y < rr.strongest - tol)
    cons = has[1:] & has[:-1]
    jit = np.abs(np.diff(y))[cons]
    m = {"det %": 100 * has.mean() if N else 0, "=strong %": 100 * eq.mean() if N else 0,
         "nearer %": 100 * nearer.mean() if N else 0,
         "jitter cm": 100 * float(np.median(jit)) if jit.size else float("nan"),
         "flips/min": 60 * float(np.count_nonzero(np.diff(has.astype(int)))) / dur}
    tr = truth_windows(rr)
    if tr is not None:
        idx, dist = tr
        yy = y[idx]
        hit = ~np.isnan(yy) & (np.abs(yy - dist) <= tol)
        clutter = ~np.isnan(yy) & (yy < dist - tol)
        m.update({"truth frames": int(len(idx)), "hit %": 100 * hit.mean(), "clutter %": 100 * clutter.mean(),
                  "miss %": 100 * np.isnan(yy).mean()})
    return m


def truth_windows(rr: RunResult, hold_s: float = 4.0):
    """Frames from each distance marker to the next marker (max hold_s) -> (frame indices, truth range).
    Markers carry the session frame counter, which equals the recording index when recording from the start."""
    mk = sorted([m for m in rr.markers if m.get("frame") is not None], key=lambda m: m["frame"])
    # distance markers inside GT-001 *walk* takes are passes, not holds: scored by `ravidata score` instead
    from .groundtruth import parse_takes
    walk = [(tk.i0, tk.i1) for tk in parse_takes(rr.run, rr.markers, len(rr.t)) if tk.mode == "walk"]
    mk = [m for m in mk if not any(a <= int(m["frame"]) < b for a, b in walk)
          and not str(m.get("text", "")).lower().startswith("take ")]
    if not any(m.get("dist") is not None for m in mk):
        return None
    N = len(rr.t)
    idx, dist = [], []
    for j, m in enumerate(mk):
        if m.get("dist") is None:
            continue
        a = int(m["frame"])
        b = int(mk[j + 1]["frame"]) if j + 1 < len(mk) else N
        if a >= N:
            continue
        t_end = rr.t[a] + hold_s
        sel = [i for i in range(a, min(b, N)) if rr.t[i] < t_end]
        idx += sel
        dist += [m["dist"]] * len(sel)
    return (np.array(idx, int), np.array(dist)) if idx else None


def table(results: list[RunResult], names: list[str] | None = None) -> str:
    lines = []
    for rr in results:
        ns = names or list(rr.outputs)
        ms = {n: metrics(rr, n) for n in ns}
        cols = list(next(iter(ms.values())).keys()) if ms else []
        lines.append(f"\n### {rr.run}  ({len(rr.t)} frames, {rr.t[-1] - rr.t[0]:.0f} s, bin {1000 * (rr.r_m[1] - rr.r_m[0]):.0f} mm, "
                     f"{len([m for m in rr.markers if m.get('dist') is not None])} distance markers)\n")
        lines.append("| detector | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---|" * len(cols))
        for n, m in ms.items():
            lines.append(f"| {n} | " + " | ".join(f"{m[c]:.0f}" if isinstance(m[c], float) and c != "jitter cm"
                                                   else (f"{m[c]:.1f}" if isinstance(m[c], float) else str(m[c]))
                                                   for c in cols) + " |")
    return "\n".join(lines)
