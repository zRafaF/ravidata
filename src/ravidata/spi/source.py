"""Live source with the control shell on a UART and the datacube over SPI (EXP-002)."""

from __future__ import annotations

import time

from .. import chip
from ..config import RadarConfig
from ..logsetup import KEEP, get_logger
from ..sources import OnFrame, OnState, SerialSource
from .link import HifSpiLink, pick_int_bit, watch_gpio

log = get_logger("spi")


class SpiSource(SerialSource):
    """``bus_factory()`` returns an opened SpiBus (CH347 or simulator). ``int_bit``: GPIO bit of the
    radar INT line, "auto" (detect after `start`), or None (blind polling)."""

    kind = "spi"

    def __init__(self, cfg: RadarConfig, on_frame: OnFrame, on_state: OnState | None = None,
                 ctrl_port: str | None = None, bus_factory=None, int_bit: int | str | None = "auto",
                 burst: int = 64, fake=None, start_on_config_errors: bool = True):
        if cfg.link != "spi":
            cfg = cfg.copy(link="spi")
        super().__init__(cfg, on_frame, on_state, ctrl_port=ctrl_port, data_port=None, fake=fake,
                         start_on_config_errors=start_on_config_errors, diagnose=False)
        self.kind = "spi-sim" if fake is not None else "spi"
        self.bus_factory = bus_factory
        self.int_bit_req = int_bit
        self.burst = burst
        self.bus = None
        self.link: HifSpiLink | None = None
        self.gpio_watch: dict | None = None
        self._t_stream = None
        self._spi_diag_done = False
        self._last_new = None
        self.hif_resets = 0
        self.stall_s = 2.0              # no new message for this long -> re-initialise the chip's SPI HIF
        self.max_resets = 3

    def _open(self):
        if self.fake is not None:
            ctrl = self.fake.ctrl
        else:
            import serial
            log.info("opening control port %s @ %d (SPI data link)", self.ctrl_port_name, self.cfg.ctrl_baudrate)
            ctrl = serial.Serial(self.ctrl_port_name, self.cfg.ctrl_baudrate, timeout=0.05)
        self.bus = self.bus_factory()
        log.info("SPI bus: %s", getattr(self.bus, "name", "?"), extra=KEEP)
        self.info["spi_bus"] = getattr(self.bus, "name", "?")
        return ctrl, None

    def _setup_link(self):
        ib = self.int_bit_req
        if ib == "auto":
            self.gpio_watch = watch_gpio(self.bus, 1.0)
            ib = pick_int_bit(self.gpio_watch)
            log.info("SPI INT auto-detect: GPIO toggles %s duty %s -> INT bit %s%s", self.gpio_watch.get("toggles"),
                     self.gpio_watch.get("duty"), ib, "" if ib is not None else " (none found: blind polling)",
                     extra=KEEP)
        self.link = HifSpiLink(self.bus, int_bit=ib, burst=self.burst)
        self.link.sync()
        self.info["int_bit"] = ib

    def _read_chunk(self) -> bytes:
        if self.link is None:
            self._t_stream = time.monotonic()
            self._setup_link()
        pks = self.link.read_burst()
        now = time.monotonic()
        if pks:
            self._last_new = now
        elif (now - (self._last_new or self._t_stream) > self.stall_s and self.hif_resets < self.max_resets
              and getattr(self, "_cc", None) is not None):
            self._hif_reset("no new message for %.1f s" % (now - (self._last_new or self._t_stream)))
            self._last_new = time.monotonic()
            return b""
        if not pks:
            if not self._spi_diag_done and time.monotonic() - self._t_stream > 3.0 and self.parser.stats.frames_ok == 0:
                self._spi_diag_done = True
                self._spi_diagnose()
            time.sleep(0.0003 if self.link.int_bit is not None else 0.002)
            return b""
        return b"".join(pks)

    def _hif_reset(self, why: str):
        """Force the chip to tear down and re-create its SPI host interface (UART re-init in between),
        then drain 4 KB. Bench 2026-09-25: after the link desynchronised the chip only sent stale bytes."""
        self.hif_resets += 1
        log.error("SPI: %s -> re-initialising the chip's SPI host interface (reset %d/%d); link stats %s",
                  why, self.hif_resets, self.max_resets, self.link.stats.to_dict(), extra=KEEP)
        cc = self._cc
        for c in ("stop", "comtype 0", f"baudrate {self.cfg.baudrate}", "report cube 4294967295", "comtype 1",
                  "report cube 4294967295", "start"):
            r = cc.send(c, phase="recover")
            if r.status not in ("ok", "unknown"):
                log.error("SPI recovery: `%s` -> %s %s", c, r.status, r.response[:80], extra=KEEP)
        self.link.sync()
        self.info["hif_resets"] = self.hif_resets

    def _spi_diagnose(self):
        st = self.link.stats.to_dict()
        w = watch_gpio(self.bus, 0.5)
        log.error("SPI: no frames after 3 s. link stats %s; GPIO watch %s; first bytes clocked in: %s", st, w,
                  bytes(self.link.first_bytes[:48]).hex(" ") or "<none>", extra=KEEP)
        log.error("SPI checklist: (1) MODE SEL ON, DIP 1,2 + 5,6,7 + 9 ON, rest OFF; (2) module is P1812 (P1806 "
                  "has no SPI); (3) `comtype 1` accepted before `report cube`; (4) MISO all 0xFF = bridge not driving "
                  "/ wrong CS; all 0x00 = device idle (no INT/poll); (5) try --int-bit none (blind) or a lower "
                  "--spi-clock", extra=KEEP)

    def spi_stats(self) -> dict:
        d = self.link.stats.to_dict() if self.link else {}
        d["int_bit"] = self.link.int_bit if self.link else None
        d["hif_resets"] = self.hif_resets
        d["gpio_watch"] = self.gpio_watch
        if self.link is not None:
            d["long_ok"], d["long_fail"] = dict(self.link.long_ok), dict(self.link.long_fail)
        d["long_mode"] = getattr(getattr(self, "bus", None), "long_mode", None)
        d["long_mode_xfers"] = dict(getattr(getattr(self, "bus", None), "mode_stats", {}) or {})
        return d

    @property
    def fail_dump(self) -> bytes:
        return bytes(self.link.fail_dump) if self.link is not None else b""

    def _close(self):
        if self.link is not None:
            log.info("SPI link stats at close: %s", self.spi_stats(), extra=KEEP)
            self.info["spi_stats"] = self.spi_stats()
        super()._close()
        if self.bus is not None and self.fake is None:
            self.bus.close()
