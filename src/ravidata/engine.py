"""Glue between a source, the processor and the run session. Used by the GUI and the CLI."""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path

import numpy as np

from .config import RadarConfig
from .frames import Frame
from .logsetup import get_logger
from .processing.pipeline import Processor, ProcessResult
from .session import RunSession
from .sim import SimSerial, Simulator, scene_presets
from .sources import BaseSource, ReplaySource, SerialSource, open_recording

log = get_logger("engine")


class Engine:
    HISTORY = 300

    def __init__(self, runs_dir: Path | None = None, process: bool = True):
        self.runs_dir = runs_dir
        # process=False: record-only mode (no detectors on the source thread; for slow laptops, `collect`)
        self.process = process
        self.source: BaseSource | None = None
        self.session: RunSession | None = None
        # an idle processor exists so parameters can be edited before the first run
        self.processor: Processor | None = Processor(RadarConfig())
        self.cfg: RadarConfig | None = None
        self._lock = threading.RLock()
        self.latest: ProcessResult | None = None
        self.latest_frame: Frame | None = None
        self.history: deque[np.ndarray] = deque(maxlen=self.HISTORY)
        self.state = "idle"
        self.state_info: dict = {}
        self._saved_params: dict | None = None
        self.frame_counter = 0

    # ------------------------------------------------------------------ lifecycle
    def _begin(self, cfg: RadarConfig, kind: str, info: dict, label: str = ""):
        self.stop()
        if self.processor is not None:
            self._saved_params = self.processor.params_snapshot()
        self.cfg = cfg
        self.processor = Processor(cfg)
        if self._saved_params:
            self._apply_params(self._saved_params)
        self.history.clear()
        self.latest = None
        self.frame_counter = 0
        self.session = RunSession(cfg, kind, info, runs_dir=self.runs_dir, label=label)
        self.session.set_params(self.processor.params_snapshot())

    def start_serial(self, cfg: RadarConfig, ctrl_port: str, data_port: str, record: bool = False,
                     start_on_config_errors: bool = True, diagnose: bool = True, label: str = ""):
        self._begin(cfg, "serial", {"ctrl_port": ctrl_port, "data_port": data_port}, label=label)
        self.source = SerialSource(cfg, self._on_frame, self._on_state, ctrl_port=ctrl_port, data_port=data_port,
                                   start_on_config_errors=start_on_config_errors, diagnose=diagnose)
        if record:
            self.session.start_recording()
        self.source.start()

    def start_spi(self, cfg: RadarConfig, ctrl_port: str, spi_dev: int = 0, int_bit="auto", record: bool = False,
                  ch347_dll: str | None = None, burst: int = 64, label: str = ""):
        """Control shell on ``ctrl_port`` (single UART, MODE SEL ON), datacube over the CH347 SPI bridge."""
        from .spi.ch347 import Ch347Bus
        from .spi.source import SpiSource
        cfg = cfg.copy(link="spi")
        self._begin(cfg, "spi", {"ctrl_port": ctrl_port, "spi_dev": spi_dev, "int_bit": int_bit,
                                 "spi_clock_hz": cfg.spi_clock_hz}, label=label)
        self.source = SpiSource(cfg, self._on_frame, self._on_state, ctrl_port=ctrl_port, int_bit=int_bit,
                                burst=burst, bus_factory=lambda: Ch347Bus(spi_dev, cfg.spi_clock_hz, ch347_dll))
        if record:
            self.session.start_recording()
        self.source.start()

    def start_sim(self, cfg: RadarConfig, scene: str = "branch+pole", record: bool = False,
                  fail_commands: set[str] | None = None, chip_overrides: dict | None = None, route: str = "data"):
        """``chip_overrides`` makes the simulated chip run a *different* config than requested
        (e.g. ``{"range_res_mm": 100}``), to exercise the mismatch detection paths."""
        sc = scene_presets()[scene]
        self._begin(cfg, "sim", {"scene": scene, "fail_commands": sorted(fail_commands or []),
                                 "chip_overrides": chip_overrides or {}, "route": route}, label=f"{cfg.name}_{scene}")
        chip_cfg = cfg.copy(**(chip_overrides or {}))
        fake = SimSerial(Simulator(chip_cfg, sc), fail_commands=fail_commands, route=route)
        if cfg.link == "spi":
            from .spi.simbus import SimSpiBus
            from .spi.source import SpiSource
            bus = SimSpiBus(fake)
            self.source = SpiSource(cfg, self._on_frame, self._on_state, fake=fake, bus_factory=lambda: bus)
        else:
            self.source = SerialSource(cfg, self._on_frame, self._on_state, fake=fake)
        if record:
            self.session.start_recording()
        self.source.start()

    def start_replay(self, path: str | Path, speed: float = 1.0, loop: bool = True):
        rec = open_recording(path)
        self._begin(rec.cfg, "replay", {"path": str(rec.path), "frames": rec.n, "format": rec.kind},
                    label=Path(rec.path).parent.name[:30])
        self.source = ReplaySource(rec, self._on_frame, self._on_state, speed=speed, loop=loop)
        self.source.start()

    def stop(self, reason: str = "stopped by user"):
        src, ses = self.source, self.session
        self.source = None
        if src is not None:
            src.stop()
        if ses is not None:
            head = getattr(src, "raw_head", None)
            if head:
                try:
                    (ses.dir / "data_port_head.bin").write_bytes(bytes(head))
                    log.info("saved first %d raw data-port bytes to data_port_head.bin (for protocol debugging)", len(head))
                except Exception as e:  # pragma: no cover
                    log.warning("could not save raw data-port bytes: %s", e)
            dump = getattr(src, "fail_dump", b"")
            if dump:
                try:
                    (ses.dir / "spi_fail_dump.bin").write_bytes(bytes(dump))
                    log.info("saved %d B of failing long SPI messages to spi_fail_dump.bin (EXP-002)", len(dump))
                except Exception as e:  # pragma: no cover
                    log.warning("could not save spi_fail_dump.bin: %s", e)
            if self.processor:
                ses.set_params(self.processor.params_snapshot())
                self._saved_params = self.processor.params_snapshot()
            ses.close(reason)
        self.session = None
        self.state = "idle"

    # ------------------------------------------------------------------ callbacks (source thread)
    def _on_state(self, state: str, info: dict):
        self.state = state
        self.state_info = {k: v for k, v in info.items() if k != "config_report"}
        if self.session:
            self.session.handle_state(state, info)

    def _on_frame(self, fr: Frame):
        with self._lock:
            try:
                res = self.processor.process(fr) if self.process else None
            except Exception as e:
                log.exception("processing failed on frame %d: %s", fr.seq, e)
                res = None
            self.latest_frame = fr
            if res is not None:
                self.latest = res
                self.history.append(res.features.p_db.astype(np.float32))
            self.frame_counter += 1
        src = self.source
        ps = getattr(src, "parser", None)
        if self.session:
            sst = src.stats.to_dict() if src else None
            if sst is not None and hasattr(src, "spi_stats") and self.frame_counter % 20 == 1:
                self._spi_cache = src.spi_stats()
            if sst is not None and getattr(self, "_spi_cache", None) is not None:
                sst["spi"] = self._spi_cache
            self.session.handle_frame(fr, res, sst,
                                      ps.stats.to_dict() if ps is not None else None)

    # ------------------------------------------------------------------ parameters
    def set_param(self, owner: str, name: str, value):
        with self._lock:
            if self.processor is None:
                return
            target = self.processor.pre if owner == "preprocess" else self.processor.detectors[owner]
            target.set(name, value)
            snap = self.processor.params_snapshot()
            self._saved_params = snap
        if self.session:
            self.session.param_changed(owner, name, value)
            self.session.set_params(snap)

    def set_primary(self, name: str):
        with self._lock:
            if self.processor and name in self.processor.detectors:
                self.processor.primary = name
                self.processor.primary_user = True
        if self.session:
            self.session.param_changed("processor", "primary", name)

    def _apply_params(self, snap: dict):
        for k, v in snap.get("preprocess", {}).items():
            try:
                self.processor.pre.set(k, v)
            except Exception:
                pass
        for det, ps in snap.get("detectors", {}).items():
            if det in self.processor.detectors:
                for k, v in ps.items():
                    try:
                        self.processor.detectors[det].set(k, v)
                    except Exception:
                        pass
        # only a primary the user picked follows into the next run; otherwise the mode default applies
        # (the detector add-on's choice per mode, else the CFAR baseline)
        if snap.get("primary_user") and snap.get("primary") in self.processor.detectors:
            self.processor.primary = snap["primary"]
            self.processor.primary_user = True

    def capture_background(self, n: int = 40):
        with self._lock:
            if self.processor:
                self.processor.pre.start_background_capture(n)
        if self.session:
            self.session.note(f"background capture requested ({n} frames)", kind="calibration")

    def calibrate_phase(self):
        with self._lock:
            if self.processor:
                self.processor.request_phase_calibration()
        if self.session:
            self.session.note("phase calibration requested (strongest target assumed at 0 deg elevation)", kind="calibration")

    def note(self, text: str, kind: str = "note"):
        if self.session:
            self.session.note(text, kind)

    def set_recording(self, on: bool):
        if not self.session:
            return
        if on:
            self.session.start_recording()
        else:
            self.session.stop_recording()

    def snapshot(self):
        with self._lock:
            return self.latest, self.latest_frame, list(self.history), self.frame_counter
