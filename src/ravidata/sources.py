"""Frame sources: live radar (serial), simulator (fake serial, same code path) and replay.

All sources run a background thread and call ``on_frame(frame)`` for every frame and
``on_state(state, info)`` for lifecycle changes. They keep :class:`SourceStats`, which the
run session turns into health log lines and the session report.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import chip
from .config import RadarConfig
from .cube import CubeShape, decode, expected_shape, infer_shape
from .frames import Frame
from .hif import HifParser
from .logsetup import KEEP, get_logger

log = get_logger("source")


@dataclass
class SourceStats:
    started_mono: float = 0.0
    frames: int = 0
    last_frame_index: int | None = None
    index_gaps: int = 0              # frames missing according to the radar frame index
    index_resets: int = 0
    shape_mismatch: int = 0
    shape_used: str = ""
    decode_errors: int = 0
    frame_times: deque = field(default_factory=lambda: deque(maxlen=200))
    bytes_times: deque = field(default_factory=lambda: deque(maxlen=200))
    notes: Counter = field(default_factory=Counter)

    def on_frame(self, fr: Frame):
        self.frames += 1
        self.frame_times.append(fr.t_mono)
        if self.last_frame_index is not None:
            gap = fr.frame_index - self.last_frame_index - 1
            if gap > 0:
                self.index_gaps += gap
            elif gap < -1:
                self.index_resets += 1
        self.last_frame_index = fr.frame_index

    def fps(self) -> float:
        t = self.frame_times
        if len(t) < 2 or t[-1] == t[0]:
            return 0.0
        return (len(t) - 1) / (t[-1] - t[0])

    def to_dict(self) -> dict:
        return {"frames": self.frames, "fps_recent": round(self.fps(), 2), "index_gaps": self.index_gaps,
                "index_resets": self.index_resets, "shape_mismatch_frames": self.shape_mismatch,
                "shape_used": self.shape_used, "decode_errors": self.decode_errors, "notes": dict(self.notes)}


OnFrame = Callable[[Frame], None]
OnState = Callable[[str, dict], None]


class BaseSource:
    kind = "base"

    def __init__(self, cfg: RadarConfig, on_frame: OnFrame, on_state: OnState | None = None):
        self.cfg = cfg
        self.on_frame = on_frame
        self.on_state = on_state or (lambda s, i: None)
        self.stats = SourceStats()
        self.state = "idle"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.info: dict = {}

    def _set_state(self, st: str, **info):
        self.state = st
        self.info.update(info)
        log.info("source %s state -> %s %s", self.kind, st, {k: v for k, v in info.items() if k != "config_report"},
                 extra=KEEP)
        try:
            self.on_state(st, info)
        except Exception:
            log.exception("on_state callback failed")

    def start(self):
        self._stop.clear()
        self.stats = SourceStats(started_mono=time.monotonic())
        self._thread = threading.Thread(target=self._run_guarded, name=f"src-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout)

    def _run_guarded(self):
        try:
            self._run()
        except Exception as e:
            log.exception("source %s crashed: %s", self.kind, e)
            self._set_state("error", error=repr(e))
        else:
            if self.state not in ("error",):
                self._set_state("stopped")

    def _run(self):  # pragma: no cover
        raise NotImplementedError


class _StopRequested(Exception):
    """Raised inside long diagnostics when the user presses Stop, so the ports are released at once."""


class SerialSource(BaseSource):
    """Live source. Works with real pyserial ports or the simulator's fake ports."""

    kind = "serial"

    def __init__(self, cfg: RadarConfig, on_frame: OnFrame, on_state: OnState | None = None,
                 ctrl_port: str | None = None, data_port: str | None = None, fake=None,
                 start_on_config_errors: bool = True, diagnose: bool = True):
        super().__init__(cfg, on_frame, on_state)
        self.diagnose = diagnose
        self.ctrl_port_name = ctrl_port
        self.data_port_name = data_port
        self.fake = fake                        # sim.SimSerial or None
        self.start_on_config_errors = start_on_config_errors
        self.parser = HifParser()
        self.config_report: chip.ConfigReport | None = None
        if fake is not None:
            self.kind = "sim"
        self._ctrl = None
        self._data = None
        self.raw_head = bytearray()          # first bytes seen on the data port, saved to the run dir
        self.RAW_HEAD_MAX = 256 * 1024

    def _open(self):
        if self.fake is not None:
            return self.fake.ctrl, self.fake.data
        import serial  # pyserial
        log.info("opening control port %s @ %d and data port %s @ %d", self.ctrl_port_name, self.cfg.ctrl_baudrate,
                 self.data_port_name, self.cfg.baudrate)
        # Retry briefly: Windows keeps a COM port locked for a moment after the previous run closed it
        # (2026-09-25 22:28: a restart right after Stop failed with PermissionError 13 on COM9).
        def _open_retry(name, baud, timeout):
            t0 = time.monotonic()
            while True:
                try:
                    return serial.Serial(name, baud, timeout=timeout)
                except serial.SerialException as e:
                    if time.monotonic() - t0 > 5.0 or self._stop.is_set():
                        raise
                    log.warning("port %s busy (%s); retrying", name, e)
                    time.sleep(0.5)
        ctrl = _open_retry(self.ctrl_port_name, self.cfg.ctrl_baudrate, 0.05)
        try:
            data = _open_retry(self.data_port_name, self.cfg.baudrate, 0.02)
        except Exception:
            ctrl.close()
            raise
        data.reset_input_buffer()
        return ctrl, data

    def _run(self):
        self._set_state("connecting", ctrl=self.ctrl_port_name or "sim",
                        data=self.data_port_name or ("spi" if self.kind.startswith("spi") else "sim"))
        self._ctrl, self._data = self._open()
        cc = chip.ControlChannel(self._ctrl)
        self._cc = cc
        self._set_state("configuring")
        rep = chip.configure(cc, self.cfg)
        self.config_report = rep
        self._set_state("configured", config_report=rep.to_dict(), config_ok=rep.ok)
        if not rep.boot_ok:
            self._set_state("error", error="radar did not answer on control port")
            self._close()
            return
        if rep.failed and not self.start_on_config_errors:
            self._set_state("error", error=f"{len(rep.failed)} config commands failed")
            self._close()
            return
        exp = expected_shape(self.cfg)
        log.info("expecting cube %s dop x %s rx x %s range = %d dwords", exp.n_dop, exp.n_rx, exp.n_range, exp.dwords)
        start_res = cc.send("start", phase="start")
        self._set_state("streaming", start_status=start_res.status)
        t_first = None
        t_start = time.monotonic()
        diagnosed = False
        last_ctrl_check = t_start
        self.ctrl_parser = HifParser()
        shape_cache: dict[int, CubeShape] = {}
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_ctrl_check > 1.0:
                last_ctrl_check = now
                self._watch_ctrl_port()
            if self.diagnose and not diagnosed and now - t_start > 2.5 and self.parser.stats.frames_ok == 0:
                diagnosed = True
                try:
                    self._diagnose_link(cc, got_bytes=self.parser.stats.bytes_in)
                except _StopRequested:
                    log.info("link diagnostics interrupted by Stop")
                continue
            try:
                chunk = self._read_chunk()
            except Exception as e:   # e.g. SerialException / PermissionError when the USB device goes away
                log.error("data link %s read failed: %s (USB unplugged, board reset or port closed?)",
                          self.data_port_name or self.kind, e)
                self._set_state("error", error=f"data port read failed: {e}")
                break
            if not chunk:
                continue
            if len(self.raw_head) < self.RAW_HEAD_MAX:
                self.raw_head.extend(chunk[: self.RAW_HEAD_MAX - len(self.raw_head)])
            if t_first is None:
                t_first = time.monotonic()
                log.info("first data bytes after %.2f s: %s", t_first - t_start, chunk[:24].hex(" "))
            for af in self.parser.feed(chunk):
                nw = len(af.words)
                shp = shape_cache.get(nw)
                if shp is None:
                    shp = infer_shape(nw, self.cfg, exp)
                    shape_cache[nw] = shp
                    if shp.inferred:
                        log.error("cube size mismatch: chip sends %d dwords, config predicts %d (%dx%dx%d). Using %s. "
                                  "Check `fft_len`/`chirp_info` read-back.", nw, exp.dwords, exp.n_dop, exp.n_rx,
                                  exp.n_range, shp.note)
                    self.stats.shape_used = f"{shp.n_dop}x{shp.n_rx}x{shp.n_range}" + (" (inferred)" if shp.inferred else "")
                if shp.inferred:
                    self.stats.shape_mismatch += 1
                try:
                    cube = decode(af.words, shp, self.cfg)
                except Exception as e:
                    self.stats.decode_errors += 1
                    log.error("decode failed for frame %d: %s", af.frame_index, e)
                    continue
                now_m, now_h = time.monotonic(), time.time()
                extra = {}
                if self.fake is not None:
                    extra["truth"] = self.fake.truth_log.get(af.frame_index)
                fr = Frame(seq=self.stats.frames, frame_index=af.frame_index, t_host=now_h, t_mono=now_m,
                           cube=cube, shape=shp, words=af.words, source=self.kind, extra=extra)
                self.stats.on_frame(fr)
                self.on_frame(fr)
        try:
            cc.send("stop", phase="stop", timeout_s=1.0)
        except Exception as e:  # pragma: no cover
            log.warning("stop command failed (port gone?): %s", e)
        self._close()

    def _read_chunk(self) -> bytes:
        n = getattr(self._data, "in_waiting", 0)
        return self._data.read(n if n else 1)

    # ------------------------------------------------------------------ link diagnostics
    def _watch_ctrl_port(self):
        """Unsolicited bytes on the control port while streaming: log them; detect HIF data there."""
        try:
            n = getattr(self._ctrl, "in_waiting", 0)
            if not n:
                return
            b = self._ctrl.read(n)
        except Exception as e:  # pragma: no cover
            log.debug("ctrl watch failed: %s", e)
            return
        before = self.ctrl_parser.stats.packets_ok
        self.ctrl_parser.feed(b)
        if self.ctrl_parser.stats.packets_ok > before:
            self.stats.notes["hif_on_ctrl_port"] += 1
            log.error("HIF data packets are arriving on the CONTROL port %s (%d so far). The chip reports data on "
                      "the UART used for commands: swap the port roles or check the DIP switches.",
                      self.ctrl_port_name or "sim", self.ctrl_parser.stats.packets_ok)
        else:
            txt = b.decode("utf-8", errors="replace").strip()
            log.info("unsolicited text on control port: %r", txt[:200])

    def _probe(self, port, seconds: float) -> tuple[int, int, bytes]:
        """Read ``port`` for ``seconds``; return (bytes, valid HIF packets, first bytes)."""
        p = HifParser()
        t0 = time.monotonic()
        total, first = 0, b""
        while time.monotonic() - t0 < seconds:
            if self._stop.is_set():
                raise _StopRequested()
            n = getattr(port, "in_waiting", 0)
            b = port.read(n if n else 1)
            if b:
                total += len(b)
                if len(first) < 32:
                    first += b[: 32 - len(first)]
                p.feed(b)
        return total, p.stats.packets_ok, first

    def _diagnose_link(self, cc: chip.ControlChannel, got_bytes: int):
        """No frames 2.5 s after `start`: find out why and, if possible, fix it (auto baud)."""
        ps = self.parser.stats
        if got_bytes:
            log.error("data port %s: %d bytes but 0 complete frames after 2.5 s (packets ok %d, errors %s). "
                      "Probing baud rates.", self.data_port_name, got_bytes, ps.packets_ok, dict(ps.errors))
        else:
            log.error("no bytes on data port %s @ %d baud 2.5 s after `start`. Running link diagnostics.",
                      self.data_port_name or "sim", self.cfg.baudrate)
        # 1) chip state
        st = cc.send("state", phase="diag")
        log.info("diag: chip state after start: %s", st.response.replace("\n", " ")[:160])
        # 2) data on the control port?
        tot, pk, first = self._probe(self._ctrl, 0.8)
        log.info("diag: control port %s: %d bytes, %d HIF packets in 0.8 s %s", self.ctrl_port_name or "sim", tot, pk,
                 first.hex(" ") if tot else "")
        if pk:
            log.error("diag: RESULT - cube data comes out on the CONTROL port (%s). Swap roles or DIP switches.",
                      self.ctrl_port_name or "sim")
            self.stats.notes["data_on_ctrl_port"] += 1
            return
        if self.fake is not None or not hasattr(self._data, "baudrate"):
            return
        # 3) baud scan on the data port
        results = []
        found = None
        for baud in (self.cfg.baudrate, 2_000_000, 1_000_000, 921_600, 460_800, 230_400, 115_200):
            if baud in [b for b, *_ in results]:
                continue
            try:
                self._data.baudrate = baud
                self._data.reset_input_buffer()
            except Exception as e:
                results.append((baud, -1, -1, f"cannot set: {e}"))
                continue
            tot, pk, first = self._probe(self._data, 0.7)
            results.append((baud, tot, pk, first.hex(" ")))
            if pk and found is None:
                found = baud
        if found is None:
            # handshake lines: UART1 on the RS6130 supports auto flow control; try RTS/DTR states
            for rts, dtr in ((False, True), (True, False), (False, False)):
                try:
                    self._data.baudrate = self.cfg.baudrate
                    self._data.rts, self._data.dtr = rts, dtr
                    self._data.reset_input_buffer()
                except Exception as e:
                    log.info("diag: cannot set RTS/DTR: %s", e)
                    break
                tot, pk, first = self._probe(self._data, 0.7)
                log.info("diag: data port @ %d with RTS=%s DTR=%s: %d bytes, %d HIF packets %s", self.cfg.baudrate,
                         rts, dtr, tot, pk, first.hex(" ")[:60])
                if pk:
                    found = self.cfg.baudrate
                    log.error("diag: RESULT - data flows only with RTS=%s DTR=%s (flow-control lines matter)", rts, dtr)
                    break
            if found is None:
                try:
                    self._data.rts, self._data.dtr = True, True
                except Exception:
                    pass
                # re-arm reporting while running (HIF init may depend on state)
                for c in ("report cube 4294967295", "uart on"):
                    r = cc.send(c, phase="diag")
                    log.info("diag: re-sent `%s` while running -> %s %s", c, r.status, r.response.replace("\n", " ")[:120])
                tot, pk, first = self._probe(self._data, 1.0)
                log.info("diag: after re-arm @ %d: %d bytes, %d HIF packets %s", self.cfg.baudrate, tot, pk,
                         first.hex(" ")[:60])
                if pk:
                    found = self.cfg.baudrate
                    log.error("diag: RESULT - data started after re-sending `report cube`/`uart on` while running: "
                              "the command ORDER matters")
        for baud, tot, pk, first in results:
            log.info("diag: data port %s @ %7d baud: %6d bytes, %4d HIF packets %s", self.data_port_name, baud, tot, pk,
                     first[:60])
        if found is not None:
            self._data.baudrate = found
            self._data.reset_input_buffer()
            self.parser = HifParser()
            self.stats.notes[f"data_baud_used_{found}"] += 1
            if found != self.cfg.baudrate:
                log.error("diag: RESULT - data arrives at %d baud, not the configured %d (`mmwc baudrate` not applied "
                          "to the data UART?). Continuing at %d.", found, self.cfg.baudrate, found)
            else:
                log.warning("diag: RESULT - data now arrives at the configured %d baud (late start?)", found)
            return
        self._data.baudrate = self.cfg.baudrate
        any_bytes = any(t > 0 for _, t, _, _ in results)
        log.error("diag: RESULT - %s. Checklist: (1) dev-board DIP 'UART0/UART1/SPI/IIC SEL' switches 1,2 AND 3,4 ON so "
                  "both UARTs reach USB; (2) MODE SEL = OFF (dual serial); (3) data port = LOWER COM number (UART1); "
                  "(4) `uart on` and `report cube` accepted (see command log); (5) try another preset / 1D vs 2D.",
                  "bytes but no valid HIF packets at any baud" if any_bytes else "no bytes at any baud rate on the data port")
        self.stats.notes["no_data_any_baud"] += 1

    def _close(self):
        for p in (self._ctrl, self._data):
            try:
                if p is not None:
                    p.close()
            except Exception:  # pragma: no cover
                pass
        if self.fake is not None:
            self.fake.close()


# ------------------------------------------------------------------------ replay
@dataclass
class Recording:
    path: Path
    cfg: RadarConfig
    times: np.ndarray                 # seconds, relative
    frame_index: np.ndarray
    n: int
    get_cube: Callable[[int], np.ndarray]
    get_words: Callable[[int], np.ndarray | None]
    meta: dict
    kind: str                         # "h5" | "legacy_npz"


def open_recording(path: str | Path) -> Recording:
    """Open one of our runs (``cubes.h5`` or its run dir) or a legacy perception2 session."""
    p = Path(path)
    if p.is_dir():
        if (p / "cubes.h5").exists():
            p = p / "cubes.h5"
        elif (p / "frames.npz").exists():
            p = p / "frames.npz"
        else:
            raise FileNotFoundError(f"no cubes.h5 or frames.npz in {p}")
    if p.suffix == ".h5":
        import h5py
        f = h5py.File(p, "r")
        cfg = RadarConfig.from_dict(json.loads(f.attrs["config_json"]))
        words = f["words"]
        shapes = f["shape"][:]
        t = f["t_mono"][:]
        fi = f["frame_index"][:]
        n = len(t)

        def gw(i):
            return words[i, : int(np.prod(shapes[i]))]

        def gc(i):
            s = shapes[i]
            return decode(gw(i), CubeShape(int(s[0]), int(s[1]), int(s[2])), cfg)

        meta = json.loads(f.attrs.get("meta_json", "{}"))
        return Recording(p, cfg, t - (t[0] if n else 0), fi, n, gc, gw, meta, "h5")
    if p.suffix == ".npz":
        z = np.load(p)
        cubes = z["cubes"]
        meta_p = p.parent / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        from .config import presets
        P = presets()
        if "legacy-perception2" not in P:
            raise ValueError("legacy perception2 sessions need the private presets")
        cfg = P["legacy-perception2"]
        mc = meta.get("config", {})
        if mc:
            cfg.notes += f" legacy detector config: sensor_height_mm={mc.get('sensor_height_mm')}"
        t = z["times"].astype(float)
        return Recording(p, cfg, t - t[0], np.arange(len(cubes)), len(cubes),
                         lambda i: cubes[i].astype(np.complex64), lambda i: None, meta, "legacy_npz")
    raise ValueError(f"unsupported recording {p}")


class ReplaySource(BaseSource):
    kind = "replay"

    def __init__(self, rec: Recording, on_frame: OnFrame, on_state: OnState | None = None, speed: float = 1.0,
                 loop: bool = False):
        super().__init__(rec.cfg, on_frame, on_state)
        self.rec = rec
        self.speed = speed
        self.loop = loop
        self.paused = False
        self.pos = 0
        self._seek: int | None = None
        self._step = 0

    def seek(self, i: int):
        self._seek = max(0, min(self.rec.n - 1, int(i)))

    def step(self, n: int = 1):
        self._step += n

    def _run(self):
        rec = self.rec
        self._set_state("streaming", path=str(rec.path), frames=rec.n, format=rec.kind)
        t_wall0 = time.monotonic()
        t_rec0 = rec.times[0] if rec.n else 0.0
        while not self._stop.is_set():
            if self._seek is not None:
                self.pos = self._seek
                self._seek = None
                t_wall0, t_rec0 = time.monotonic(), rec.times[self.pos]
            if self.pos >= rec.n:
                if self.loop:
                    self.pos = 0
                    t_wall0, t_rec0 = time.monotonic(), rec.times[0]
                    continue
                self._set_state("finished")
                self._stop.wait(0.2)
                while not self._stop.is_set() and self._seek is None:
                    self._stop.wait(0.1)
                continue
            if self.paused and self._step <= 0:
                self._stop.wait(0.05)
                t_wall0, t_rec0 = time.monotonic(), rec.times[self.pos]
                continue
            if not self.paused:
                due = t_wall0 + (rec.times[self.pos] - t_rec0) / max(self.speed, 1e-3)
                wait = due - time.monotonic()
                if wait > 0:
                    self._stop.wait(min(wait, 0.1))
                    if wait > 0.1:
                        continue
            else:
                self._step -= 1
            i = self.pos
            cube = rec.get_cube(i)
            shp = CubeShape(*cube.shape)
            fr = Frame(seq=i, frame_index=int(rec.frame_index[i]), t_host=time.time(), t_mono=time.monotonic(),
                       cube=cube, shape=shp, words=rec.get_words(i), source="replay", extra={"replay_pos": i, "t_rec": float(rec.times[i])})
            self.stats.on_frame(fr)
            self.on_frame(fr)
            self.pos += 1
