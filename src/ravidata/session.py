"""Run session: one directory per connect/replay with logs, metadata, events, optional raw
recording and an agent-readable ``SESSION_REPORT.md``.

Layout::

    runs/<YYYYmmdd_HHMMSS>_<source>_<preset>/
        summary.log        strict, collapsed log (read this first)
        verbose.log        everything (DEBUG), still collapsed
        meta.json          config, derived values, validation, chip command log, params, stats
        events.jsonl       notes, markers, state changes, alert on/off transitions, param changes
        cubes.h5           raw DWORDs + timestamps + per-frame primary detection (only if recording)
        SESSION_REPORT.md  human/agent digest, rewritten periodically and at close
    runs/LATEST.txt        path of the most recent run
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__
from .config import RadarConfig
from .frames import Frame
from .logsetup import KEEP, attach_run_logs, get_logger
from .processing.pipeline import ProcessResult

log = get_logger("session")


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path.cwd()


def default_runs_dir() -> Path:
    return repo_root() / "runs"


def _git_info(root: Path) -> dict:
    try:
        # --no-optional-locks: never write .git/index. A plain `git status` that hits the timeout is killed while
        # holding .git/index.lock, which then breaks every later git command (seen 2026-11-08 and 2026-11-13).
        c = subprocess.run(["git", "--no-optional-locks", "rev-parse", "--short", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=3)
        s = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain"], cwd=root,
                           capture_output=True, text=True, timeout=3)
        return {"commit": c.stdout.strip() or None, "dirty_files": len([x for x in s.stdout.splitlines() if x.strip()])}
    except Exception:
        return {"commit": None}


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


class Recorder:
    """Appends raw DWORDs (lossless, decode-convention independent) to HDF5."""

    def __init__(self, path: Path, cfg: RadarConfig, meta: dict, max_words: int):
        import h5py
        self.path = path
        self.f = h5py.File(path, "w")
        self.f.attrs["config_json"] = json.dumps(cfg.to_dict())
        self.f.attrs["meta_json"] = json.dumps(meta, default=_jsonable)
        self.f.attrs["format"] = "ravidata-cubes-v1"
        self.max_words = max_words
        mk = lambda name, shape, dt: self.f.create_dataset(  # noqa: E731
            name, shape=(0, *shape), maxshape=(None, *shape), dtype=dt, chunks=(64, *shape),
            compression="gzip" if name == "words" else None, compression_opts=1 if name == "words" else None)
        self.ds = {
            "words": mk("words", (max_words,), "<u4"),
            "shape": mk("shape", (3,), "<i4"),
            "frame_index": mk("frame_index", (), "<i8"),
            "t_host": mk("t_host", (), "<f8"),
            "t_mono": mk("t_mono", (), "<f8"),
            "det_range_m": mk("det_range_m", (), "<f4"),
            "det_height_m": mk("det_height_m", (), "<f4"),
        }
        self.n = 0
        self.truncated = 0

    def append(self, fr: Frame, res: ProcessResult | None):
        if fr.words is None:
            return
        i = self.n
        for ds in self.ds.values():
            ds.resize(i + 1, axis=0)
        w = fr.words
        if len(w) > self.max_words:
            self.truncated += 1
            w = w[: self.max_words]
        row = np.zeros(self.max_words, dtype=np.uint32)
        row[: len(w)] = w
        self.ds["words"][i] = row
        self.ds["shape"][i] = (fr.shape.n_dop, fr.shape.n_rx, fr.shape.n_range)
        self.ds["frame_index"][i] = fr.frame_index
        self.ds["t_host"][i] = fr.t_host
        self.ds["t_mono"][i] = fr.t_mono
        nr = res.primary_out.nearest if res else None
        self.ds["det_range_m"][i] = nr.range_m if nr else np.nan
        self.ds["det_height_m"][i] = nr.height_m if nr else np.nan
        self.n += 1
        if self.n % 100 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:  # pragma: no cover
            pass


class RunSession:
    HEALTH_EVERY_S = 10.0
    REPORT_EVERY_S = 30.0

    def __init__(self, cfg: RadarConfig, source_kind: str, source_info: dict | None = None,
                 runs_dir: Path | None = None, label: str = ""):
        self.cfg = cfg
        self.source_kind = source_kind
        self.runs_dir = Path(runs_dir or default_runs_dir())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in (label or cfg.name))[:40]
        self.run_id = f"{stamp}_{source_kind}_{safe}"
        self.dir = self.runs_dir / self.run_id
        k = 1
        while self.dir.exists():
            k += 1
            self.dir = self.runs_dir / f"{self.run_id}_{k}"
        self.dir.mkdir(parents=True)
        self.logs = attach_run_logs(self.dir)
        self._lock = threading.RLock()
        self._events = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        self.t0 = time.time()
        self.t0_mono = time.monotonic()
        self.recorder: Recorder | None = None
        self.recordings: list[dict] = []
        self.state_history: list[tuple[float, str, dict]] = []
        self.config_report: dict | None = None
        self.source_stats: dict = {}
        self.parser_stats: dict = {}
        self.frames = 0
        self.proc_ms: list[float] = []
        self.det_stats: dict[str, dict] = {}
        self._alert_state: dict[str, bool] = {}
        self._last_health = time.monotonic()
        self._last_report = time.monotonic()
        self.notes: list[dict] = []
        self.params_start: dict | None = None
        self.params_last: dict | None = None
        self.truth_stats = Counter()
        self._last_gaps = 0
        self._last_incomplete = 0
        self.closed = False
        root = repo_root()
        d = cfg.derive()
        issues = cfg.validate()
        self.meta = {
            "run_id": self.run_id, "started": datetime.now().isoformat(timespec="seconds"),
            "tool_version": __version__,
            "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
            "git": _git_info(root),
            "source": {"kind": source_kind, **(source_info or {})},
            "config": cfg.to_dict(), "derived": d.to_dict(),
            "validation": [i.__dict__ for i in issues],
        }
        self._write_meta()
        (self.runs_dir / "LATEST.txt").write_text(str(self.dir) + "\n", encoding="utf-8")
        log.info("=== run %s started (source=%s, preset=%s) -> %s", self.run_id, source_kind, cfg.name, self.dir)
        for line in d.summary_lines():
            log.info("derived: %s", line, extra=KEEP)
        if not issues:
            log.info("config validation: no issues", extra=KEEP)
        for i in issues:
            (log.error if i.level == "error" else log.warning if i.level == "warning" else log.info)(
                "config validation: %s", i, extra=KEEP)
        self.event("run_start", preset=cfg.name, source=source_kind, issues=[str(i) for i in issues])

    # ------------------------------------------------------------------ events
    def event(self, etype: str, **data):
        data.pop("type", None)
        rec = {"t": round(time.time() - self.t0, 3), "wall": datetime.now().strftime("%H:%M:%S"), "type": etype, **data}
        with self._lock:
            if self._events.closed:
                return
            self._events.write(json.dumps(rec, default=_jsonable) + "\n")
            self._events.flush()

    def note(self, text: str, kind: str = "note"):
        t = round(time.time() - self.t0, 2)
        self.notes.append({"t": t, "kind": kind, "text": text, "frame": self.frames})
        log.info("USER %s @%.1fs (frame %d): %s", kind.upper(), t, self.frames, text, extra=KEEP)
        self.event(kind, text=text, frame=self.frames)

    def param_changed(self, owner: str, name: str, value):
        log.info("param %s.%s = %r", owner, name, value)
        self.event("param", owner=owner, name=name, value=value)

    # ------------------------------------------------------------------ source hooks
    def handle_state(self, state: str, info: dict):
        self.state_history.append((time.time() - self.t0, state, {k: v for k, v in info.items() if k != "config_report"}))
        if "config_report" in info:
            self.config_report = info["config_report"]
            self.meta["config_report"] = self.config_report
            self._write_meta()
        self.event("state", state=state, **{k: v for k, v in info.items() if k != "config_report"})

    def handle_frame(self, fr: Frame, res: ProcessResult | None, source_stats: dict | None = None,
                     parser_stats: dict | None = None):
        with self._lock:
            self.frames += 1
            if source_stats is not None:
                self.source_stats = source_stats
            if parser_stats is not None:
                self.parser_stats = parser_stats
            if res is not None:
                self.proc_ms.append(res.proc_ms)
                if len(self.proc_ms) > 5000:
                    del self.proc_ms[:1000]
                for name, out in res.outputs.items():
                    st = self.det_stats.setdefault(name, {"frames": 0, "alert_frames": 0, "ranges": [], "first_alert_t": None,
                                                           "transitions": 0})
                    st["frames"] += 1
                    if out.alert:
                        st["alert_frames"] += 1
                        st["ranges"].append(out.nearest.range_m)
                        if len(st["ranges"]) > 20000:
                            del st["ranges"][:5000]
                        if st["first_alert_t"] is None:
                            st["first_alert_t"] = round(time.time() - self.t0, 2)
                    prev = self._alert_state.get(name)
                    if prev is None or prev != out.alert:
                        self._alert_state[name] = out.alert
                        if prev is not None:
                            st["transitions"] += 1
                            nr = out.nearest
                            self.event("alert_on" if out.alert else "alert_off", detector=name, frame=fr.seq,
                                       range_m=round(nr.range_m, 3) if nr else None,
                                       height_m=round(nr.height_m, 2) if nr else None)
                            log.debug("%s alert %s at frame %d (%s)", name, "ON" if out.alert else "off", fr.seq,
                                      f"{nr.range_m:.2f} m" if nr else "-")
                if res.truth:
                    self._score_truth(res)
                nr = res.primary_out.nearest
                log.debug("frame %d idx %d: %s nearest=%s proc %.2f ms", fr.seq, fr.frame_index, res.primary,
                          f"{nr.range_m:.2f}m h={nr.height_m:.2f}" if nr else "none", res.proc_ms)
            if self.recorder is not None:
                try:
                    self.recorder.append(fr, res)
                except Exception as e:
                    log.error("recording failed, stopping recorder: %s", e)
                    self.stop_recording()
        now = time.monotonic()
        if now - self._last_health >= self.HEALTH_EVERY_S:
            self._last_health = now
            self.health_line()
            self.logs.flush_suppressed()
        if now - self._last_report >= self.REPORT_EVERY_S:
            self._last_report = now
            self.write_report()

    def _score_truth(self, res: ProcessResult):
        vis = [v for v in res.truth.get("visible", []) if v["label"] != "ground"]
        maxr = res.extra.get("max_range_m", 2.5)
        in_range = [v for v in vis if v["r"] <= maxr]
        alert = res.primary_out.alert
        if in_range and alert:
            self.truth_stats["hit"] += 1
            err = abs(res.primary_out.nearest.range_m - min(v["r"] for v in in_range))
            self.truth_stats["range_err_cm_sum"] += int(err * 100)
        elif in_range:
            self.truth_stats["miss"] += 1
        elif alert:
            self.truth_stats["false_alert"] += 1
        else:
            self.truth_stats["correct_quiet"] += 1

    def health_line(self):
        ss = self.source_stats
        ps = self.parser_stats
        d = self.cfg.derive()
        pm = np.median(self.proc_ms[-500:]) if self.proc_ms else float("nan")
        parts = [f"{ss.get('fps_recent', 0):.1f} fps (exp {d.expected_fps:.1f})", f"frames {self.frames}",
                 f"idx gaps {ss.get('index_gaps', 0)}", f"parse err {ps.get('errors', {})}",
                 f"incomplete {ps.get('frames_incomplete', 0)}", f"proc {pm:.1f} ms"]
        if ss.get("shape_mismatch_frames"):
            parts.append(f"SHAPE MISMATCH {ss['shape_mismatch_frames']} ({ss.get('shape_used')})")
        for name, st in self.det_stats.items():
            r = st["ranges"][-200:]
            parts.append(f"{name}: alert {100*st['alert_frames']/max(1,st['frames']):.0f}%"
                         + (f" med {np.median(r):.2f} m" if r else ""))
        if self.truth_stats:
            parts.append("sim truth " + ", ".join(f"{k}={v}" for k, v in self.truth_stats.items() if k != "range_err_cm_sum"))
        log.info("health: %s", " | ".join(parts))
        # problems worth a WARNING (collapsed if they persist)
        fps = ss.get("fps_recent", 0) or 0
        if self.source_kind != "replay" and self.frames > 50 and fps < 0.8 * d.expected_fps:
            log.warning("frame rate %.1f fps is below 80%% of the configured %.1f fps (link %.0f%% busy by estimate; "
                        "chip timing may differ from the config)", fps, d.expected_fps, 100 * d.link_utilisation)
        gaps = ss.get("index_gaps", 0)
        if gaps > self._last_gaps:
            log.warning("radar frame index jumped: %d frames missing so far (+%d since last health line)", gaps,
                        gaps - self._last_gaps)
        self._last_gaps = gaps
        inc = ps.get("frames_incomplete", 0)
        if inc > self._last_incomplete:
            log.warning("%d incomplete frames so far (fragments lost on the data link)", inc)
        self._last_incomplete = inc

    # ------------------------------------------------------------------ recording
    def start_recording(self, max_words: int | None = None):
        # The HDF5 file is created OUTSIDE the lock: creating it can take most of a second on Windows (antivirus),
        # and while the lock is held the source thread cannot hand over frames, the radar link is not drained and
        # the chip drops frames (2026-11-13: the first 21 frames of a 1D take were lost that way).
        with self._lock:
            if self.recorder is not None:
                return self.recorder.path
            k = len(self.recordings) + 1
            p = self.dir / ("cubes.h5" if k == 1 else f"cubes_{k:02d}.h5")
            mw = max_words or max(self.cfg.derive().cube_dwords, 1)
            meta = {"run_id": self.run_id, "config_report": self.config_report}
        rec = Recorder(p, self.cfg, meta, mw)
        with self._lock:
            self.recorder = rec
            self.recordings.append({"file": p.name, "start_frame": self.frames, "start_t": round(time.time() - self.t0, 2)})
            log.info("recording raw cubes -> %s (%d dwords/frame)", p.name, mw)
            self.event("record_start", file=p.name)
            return p

    def stop_recording(self):
        with self._lock:
            if self.recorder is None:
                return
            r = self.recorder
            self.recorder = None
            self.recordings[-1].update({"frames": r.n, "end_t": round(time.time() - self.t0, 2), "truncated": r.truncated})
        r.close()                                     # flush/close outside the lock (see start_recording)
        log.info("recording stopped: %s, %d frames", r.path.name, r.n)
        self.event("record_stop", file=r.path.name, frames=r.n)

    # ------------------------------------------------------------------ output
    def _write_meta(self):
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, default=_jsonable), encoding="utf-8")

    def set_params(self, snapshot: dict):
        if self.params_start is None:
            self.params_start = snapshot
            self.meta["params_start"] = snapshot
        self.params_last = snapshot

    def write_report(self):
        from .report import build_report
        try:
            txt = build_report(self)
            (self.dir / "SESSION_REPORT.md").write_text(txt, encoding="utf-8")
        except Exception as e:  # never let reporting kill a run
            log.exception("could not write session report: %s", e)

    def _append_ledger(self, reason: str):
        """One line per run in runs/config_ledger.jsonl: which configurations worked, failed, or misbehaved."""
        try:
            d = self.cfg.derive()
            rep = self.config_report or {}
            row = {
                "run_id": self.run_id, "source": self.source_kind, "preset": self.cfg.name,
                "config": {k: v for k, v in self.cfg.to_dict().items() if k not in ("notes",)},
                "validation": [f"{v['level']}:{v['code']}" for v in self.meta.get("validation", [])],
                "boot_ok": rep.get("boot_ok"), "rejected_cmds": rep.get("failed", []),
                "readback_mismatches": rep.get("check_mismatches", []),
                "link_notes": self.source_stats.get("notes", {}),
                "frames": self.frames, "duration_s": round(time.time() - self.t0, 1),
                "fps_measured": self.source_stats.get("fps_recent"), "fps_expected": round(d.expected_fps, 2),
                "shape_used": self.source_stats.get("shape_used"), "index_gaps": self.source_stats.get("index_gaps"),
                "parse_errors": self.parser_stats.get("errors"), "end_reason": reason,
                "notes": [n["text"] for n in self.notes][:20],
            }
            with open(self.runs_dir / "config_ledger.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=_jsonable) + "\n")
        except Exception as e:  # pragma: no cover
            log.warning("could not append to config ledger: %s", e)

    def close(self, reason: str = "closed"):
        if self.closed:
            return
        self.closed = True
        self.stop_recording()
        self.health_line()
        self.meta.update({"ended": datetime.now().isoformat(timespec="seconds"), "end_reason": reason,
                          "duration_s": round(time.time() - self.t0, 1), "frames": self.frames,
                          "source_stats": self.source_stats, "parser_stats": self.parser_stats,
                          "params_end": self.params_last, "recordings": self.recordings,
                          "detector_stats": {k: {kk: vv for kk, vv in v.items() if kk != "ranges"}
                                             for k, v in self.det_stats.items()},
                          "sim_truth": dict(self.truth_stats), "notes": self.notes})
        self._write_meta()
        self.event("run_end", reason=reason, frames=self.frames)
        log.info("=== run %s ended (%s): %d frames in %.1f s", self.run_id, reason, self.frames, time.time() - self.t0)
        self.logs.flush_suppressed()
        self.write_report()
        self._append_ledger(reason)
        with self._lock:
            self._events.close()
        self.logs.close()

