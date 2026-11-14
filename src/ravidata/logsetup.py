"""Logging for humans *and* agents.

Two log files are written for every run (plus the console / GUI):

* ``verbose.log`` — DEBUG and up. Everything, including per-frame debug lines, but
  **spam is collapsed**: after ``burst`` identical-template messages inside ``window_s``,
  further ones are counted and replaced by one summary line.
* ``summary.log`` — INFO and up with much stricter collapsing. It is meant to be read as
  context by an agent: lifecycle, configuration, chip responses, warnings, errors and
  periodic health lines.

"Identical" means the same logger, level and *message template* (``record.msg`` before
``%`` formatting). So ``log.warning("CRC error in packet %d", n)`` collapses even though ``n``
changes.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-18s %(message)s"
DATE_FORMAT = "%H:%M:%S"
ROOT_NAME = "ravidata"
#: pass as ``extra=KEEP`` for lines that must never be collapsed (config, chip answers, lifecycle)
KEEP = {"no_collapse": True}


@dataclass
class _KeyState:
    first_t: float
    count: int = 0            # emitted within the window
    suppressed: int = 0       # swallowed within the window
    last_record: logging.LogRecord | None = None


class CollapsingHandler(logging.Handler):
    """Wraps a handler and collapses repeated messages.

    For each key (logger, level, template): the first ``burst`` records inside ``window_s``
    pass through; the rest are counted. When the window ends (checked on every emit and on
    :meth:`flush_suppressed`), one line ``[collapsed] N more x '<template>' in T s; last: ...``
    is emitted.
    """

    def __init__(self, target: logging.Handler, burst: int = 3, window_s: float = 30.0):
        super().__init__(level=target.level)
        self.target = target
        self.burst = burst
        self.window_s = window_s
        self._states: dict[tuple, _KeyState] = {}
        self._lock_c = threading.RLock()

    def setFormatter(self, fmt):  # noqa: N802 (logging API)
        super().setFormatter(fmt)
        self.target.setFormatter(fmt)

    def _key(self, record: logging.LogRecord) -> tuple:
        return (record.name, record.levelno, str(record.msg))

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < self.level:
            return
        now = record.created
        with self._lock_c:
            self._expire(now)
            if getattr(record, "no_collapse", False):
                self.target.handle(record)
                return
            key = self._key(record)
            st = self._states.get(key)
            if st is None:
                st = self._states[key] = _KeyState(first_t=now)
            if st.count < self.burst:
                st.count += 1
                self.target.handle(record)
            else:
                st.suppressed += 1
                st.last_record = record

    def _expire(self, now: float) -> None:
        dead = []
        for key, st in self._states.items():
            if now - st.first_t >= self.window_s:
                if st.suppressed:
                    self._emit_summary(key, st, now)
                dead.append(key)
        for key in dead:
            del self._states[key]

    def _emit_summary(self, key, st: _KeyState, now: float) -> None:
        last = st.last_record
        try:
            last_txt = last.getMessage() if last else ""
        except Exception:  # pragma: no cover
            last_txt = str(last.msg) if last else ""
        rec = logging.LogRecord(
            name=key[0], level=key[1], pathname=__file__, lineno=0,
            msg="[collapsed] %d more x '%s' in %.1fs; last: %s",
            args=(st.suppressed, key[2][:120], now - st.first_t, last_txt[:300]),
            exc_info=None,
        )
        rec.created = now
        rec.msecs = (now - int(now)) * 1000
        self.target.handle(rec)

    def flush_suppressed(self, force: bool = False) -> None:
        now = time.time()
        with self._lock_c:
            if force:
                for key, st in list(self._states.items()):
                    if st.suppressed:
                        self._emit_summary(key, st, now)
                self._states.clear()
            else:
                self._expire(now)
        self.target.flush()

    def flush(self):
        self.target.flush()

    def close(self):
        self.flush_suppressed(force=True)
        self.target.close()
        super().close()


class RingBufferHandler(logging.Handler):
    """Keeps the last N formatted lines (used by the GUI log panel and the session report)."""

    def __init__(self, capacity: int = 2000, level=logging.INFO):
        super().__init__(level=level)
        self.lines: deque[tuple[int, str]] = deque(maxlen=capacity)
        self.counter = 0  # monotonically increasing, lets consumers fetch "new since"

    def emit(self, record):
        try:
            self.counter += 1
            self.lines.append((self.counter, self.format(record)))
        except Exception:  # pragma: no cover
            self.handleError(record)

    def since(self, counter: int) -> list[tuple[int, str]]:
        return [x for x in list(self.lines) if x[0] > counter]


class IssueCollector(logging.Handler):
    """Collects WARNING+ records (deduplicated by template) for the session report."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.items: dict[tuple, dict] = {}

    def emit(self, record):
        try:
            norm = re.sub(r"\d+(\.\d+)?", "#", record.getMessage())[:160]
        except Exception:  # pragma: no cover
            norm = str(record.msg)
        key = (record.name, record.levelname, norm)
        it = self.items.get(key)
        if it is None:
            self.items[key] = it = {
                "logger": record.name, "level": record.levelname, "template": str(record.msg),
                "count": 0, "first": record.created, "example": "",
            }
        it["count"] += 1
        it["last"] = record.created
        if not it["example"]:
            try:
                it["example"] = record.getMessage()[:500]
            except Exception:  # pragma: no cover
                it["example"] = str(record.msg)[:500]


@dataclass
class RunLogging:
    """Handles attached for one run directory; detach with :meth:`close`."""

    run_dir: Path
    handlers: list[logging.Handler] = field(default_factory=list)
    collapsers: list[CollapsingHandler] = field(default_factory=list)
    issues: IssueCollector = field(default_factory=IssueCollector)

    def flush_suppressed(self):
        for c in self.collapsers:
            c.flush_suppressed()

    def close(self):
        root = logging.getLogger(ROOT_NAME)
        for h in self.handlers:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:  # pragma: no cover
                pass
        self.handlers.clear()


_ring: RingBufferHandler | None = None
_console_installed = False


def get_ring() -> RingBufferHandler:
    global _ring
    if _ring is None:
        _ring = RingBufferHandler()
        _ring.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logging.getLogger(ROOT_NAME).addHandler(_ring)
    return _ring


def setup_base(console_level: int = logging.INFO, app_log_dir: Path | None = None) -> None:
    """Console + ring buffer + (optionally) app-lifetime logs. Safe to call more than once."""
    global _console_installed
    root = logging.getLogger(ROOT_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    get_ring()
    if not _console_installed:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(console_level)
        cc = CollapsingHandler(ch, burst=3, window_s=30)
        cc.setLevel(console_level)
        cc.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(cc)
        _console_installed = True
        _install_excepthooks()
        if app_log_dir is not None:
            app_log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(app_log_dir / "app.log", encoding="utf-8")
            fh.setLevel(logging.INFO)
            ac = CollapsingHandler(fh, burst=5, window_s=30)
            ac.setLevel(logging.INFO)
            ac.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)-18s %(message)s", "%Y-%m-%d %H:%M:%S"))
            root.addHandler(ac)


def attach_run_logs(run_dir: Path) -> RunLogging:
    """Attach ``verbose.log`` and ``summary.log`` for a run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger(ROOT_NAME)
    fmt = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    rl = RunLogging(run_dir=run_dir)

    vf = logging.FileHandler(run_dir / "verbose.log", encoding="utf-8")
    vf.setLevel(logging.DEBUG)
    vc = CollapsingHandler(vf, burst=20, window_s=5.0)
    vc.setLevel(logging.DEBUG)
    vc.setFormatter(fmt)

    sf = logging.FileHandler(run_dir / "summary.log", encoding="utf-8")
    sf.setLevel(logging.INFO)
    sc = CollapsingHandler(sf, burst=3, window_s=60.0)
    sc.setLevel(logging.INFO)
    sc.setFormatter(fmt)

    for h in (vc, sc, rl.issues):
        root.addHandler(h)
        rl.handlers.append(h)
    rl.collapsers = [vc, sc]
    return rl


def _install_excepthooks():
    log = logging.getLogger(ROOT_NAME + ".crash")

    def hook(exc_type, exc, tb):
        log.critical("Unhandled exception: %s\n%s", exc, "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    def thook(args):
        log.critical("Unhandled exception in thread %s: %s\n%s", args.thread.name if args.thread else "?",
                     args.exc_value, "".join(traceback.format_exception(args.exc_type, args.exc_value,
                                                                        args.exc_traceback)))

    sys.excepthook = hook
    threading.excepthook = thook


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_NAME}.{name}")
