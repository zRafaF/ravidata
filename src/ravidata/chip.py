"""Control channel (`mmwc` text commands) and device configuration for the RS6130.

Every command, its raw response, classification and timing is kept in a
:class:`CommandResult`. Failures do not abort the configuration: all commands are tried so
one run shows *every* rejected setting. The caller decides whether to start streaming.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field

from .config import RadarConfig
from .logsetup import KEEP, get_logger

log = get_logger("chip")

PROMPT = b"psic:/$"
_OK_PATTERNS = ("Command Success", "<Success")
_WARN_PATTERNS = ("<Fail 1>",)
_FAIL_PATTERNS = ("Command Failed", "<Fail", "ERR", "Error", "error", "Invalid", "invalid", "not support",
                  "Unknown", "unknown command")


@dataclass
class CommandResult:
    cmd: str
    status: str            # ok | warn | fail | no_response | unknown | skipped
    response: str
    elapsed_ms: float
    phase: str = "config"  # config | query | start | stop

    def to_dict(self) -> dict:
        return asdict(self)


_ERR_TEXT = re.compile(r"fail|error|invalid|not support|unknown command|init fail", re.I)


def classify(text: str) -> str:
    """ok | warn | fail | no_response | unknown.

    A response can contain an error line *and* "Command Success" (seen on the RS6130:
    ``report cube`` -> "host interface init fail / Command Success"). That is a failure: the
    success line only means the shell parsed the command.
    """
    if not text.strip():
        return "no_response"
    if any(p in text for p in _WARN_PATTERNS):
        return "warn"
    body = text.replace("Command Success", "")
    if _ERR_TEXT.search(body):
        return "fail"
    if any(p in text for p in _OK_PATTERNS):
        return "ok"
    return "unknown"


def clean_response(raw: bytes, cmd: str) -> str:
    txt = raw.decode("utf-8", errors="replace").replace("\r", "")
    txt = txt.replace(PROMPT.decode(), "").strip()
    # drop the echoed command line
    lines = [ln for ln in txt.split("\n") if ln.strip() and ln.strip() != f"mmwc {cmd}".strip()]
    return "\n".join(lines).strip()


class ControlChannel:
    """Talks to the radar's command UART. ``port`` is anything with read/write/in_waiting."""

    def __init__(self, port, name: str = "ctrl"):
        self.port = port
        self.name = name
        self.history: list[CommandResult] = []

    def _drain(self) -> bytes:
        junk = b""
        t0 = time.monotonic()
        while time.monotonic() - t0 < 0.3:
            n = getattr(self.port, "in_waiting", 0)
            if not n:
                break
            junk += self.port.read(n)
        if junk:
            log.debug("drained %d unsolicited bytes from %s: %r", len(junk), self.name, junk[:200])
        return junk

    def send(self, cmd: str, timeout_s: float = 1.5, phase: str = "config") -> CommandResult:
        self._drain()
        line = (f"mmwc {cmd}".strip() + "\n").encode()
        t0 = time.monotonic()
        self.port.write(line)
        raw = b""
        last_rx = t0
        while True:
            now = time.monotonic()
            n = getattr(self.port, "in_waiting", 0)
            if n:
                raw += self.port.read(n)
                last_rx = now
                if raw.rstrip().endswith(PROMPT) and len(raw.rstrip()) > len(PROMPT) + 2:
                    # allow a short tail for multi-line answers
                    tail_t = time.monotonic()
                    while time.monotonic() - tail_t < 0.05:
                        m = getattr(self.port, "in_waiting", 0)
                        if m:
                            raw += self.port.read(m)
                        else:
                            time.sleep(0.005)
                    break
            else:
                if now - t0 > timeout_s:
                    break
                if raw and now - last_rx > 0.4:   # silence after some output
                    break
                time.sleep(0.005)
        elapsed = (time.monotonic() - t0) * 1000
        text = clean_response(raw, cmd)
        status = classify(text)
        res = CommandResult(cmd=cmd, status=status, response=text, elapsed_ms=elapsed, phase=phase)
        self.history.append(res)
        short = re.sub(r"\s+", " ", text)[:160]
        if status == "ok" or (phase in ("query", "boot") and status == "unknown"):
            log.debug("mmwc %-28s -> %s (%.0f ms) %s", cmd, status, elapsed, short, extra=KEEP)
        elif phase == "boot" and status == "no_response":
            log.debug("mmwc (boot probe) -> no response yet")
        else:
            lvl = log.error if status in ("fail", "no_response") else log.warning
            lvl("mmwc %-28s -> %s (%.0f ms): %s", cmd, status.upper(), elapsed, short or "<no bytes>", extra=KEEP)
        return res


@dataclass
class ConfigReport:
    results: list[CommandResult] = field(default_factory=list)
    queries: dict[str, str] = field(default_factory=dict)
    readback: dict[str, dict] = field(default_factory=dict)
    checks: list = field(default_factory=list)
    boot_ok: bool = False

    @property
    def failed(self) -> list[CommandResult]:
        return [r for r in self.results if r.status in ("fail", "no_response") and r.phase == "config"]

    @property
    def failed_queries(self) -> list[CommandResult]:
        return [r for r in self.results if r.status in ("fail", "no_response") and r.phase == "query"]

    @property
    def ok(self) -> bool:
        return self.boot_ok and not self.failed

    def to_dict(self) -> dict:
        return {"boot_ok": self.boot_ok, "results": [r.to_dict() for r in self.results],
                "queries": self.queries, "failed": [r.cmd for r in self.failed],
                "failed_queries": [r.cmd for r in self.failed_queries], "readback": self.readback,
                "checks": [c.to_dict() for c in self.checks],
                "check_mismatches": [c.item for c in self.checks if not c.ok]}


def wait_for_boot(ctrl: ControlChannel, timeout_s: float = 8.0) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        r = ctrl.send("", timeout_s=1.0, phase="boot")
        if r.status != "no_response":
            log.info("radar answered on control port after %.1f s", time.monotonic() - t0)
            return True
    log.error("radar did not answer `mmwc` within %.0f s (wrong port? baud %s? powered?)", timeout_s,
              getattr(ctrl.port, "baudrate", "?"))
    return False


def configure(ctrl: ControlChannel, cfg: RadarConfig, query: bool = True) -> ConfigReport:
    rep = ConfigReport()
    rep.boot_ok = wait_for_boot(ctrl)
    if not rep.boot_ok:
        return rep
    cmds = cfg.to_commands()
    log.info("sending %d config commands for preset '%s'", len(cmds), cfg.name)
    for c in cmds:
        rep.results.append(ctrl.send(c))
    if query:
        for q in cfg.QUERY_COMMANDS:
            r = ctrl.send(q, phase="query")
            rep.results.append(r)
            rep.queries[q] = r.response
    if rep.failed:
        log.error("%d/%d config commands FAILED: %s", len(rep.failed), len(cmds), ", ".join(r.cmd for r in rep.failed),
                  extra=KEEP)
    else:
        log.info("all %d config commands accepted", len(cmds))
    for q, a in rep.queries.items():
        log.info("chip read-back `%s`: %s", q, re.sub(r"\s+", " ", a)[:200] or "<empty>", extra=KEEP)
    if rep.queries:
        from .readback import compare, log_checks, parse_all
        try:
            rep.readback = parse_all(rep.queries)
            rep.checks = compare(cfg, rep.readback)
            log_checks(rep.checks)
        except Exception as e:  # never block streaming on a parser problem
            log.warning("could not parse/compare read-back values: %s", e)
    return rep
