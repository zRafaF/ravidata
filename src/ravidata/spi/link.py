"""HIF over SPI, host (master) side: passive reporting with a notify/INT line (EXP-002).

Written from the vendor sources, read as text only (see
``docs/vendor/possumic/update-2026-09-25/REVIEW.md`` and ``docs/experiments/EXP-002-spi-link.md``):

* The device raises INT (active high, radar PA6) when report data is queued. It waits
  ~100 ms for a POLL, then **discards** the data.
* POLL = HIF message 0x0C (HOST_READ), type 1, REQ|CHECK: ``A5 ck 15 0C 04 00 | poll_type=1,
  err_num=0, burst u16 | check32``. ``burst`` is the maximum number of HIF messages the device
  sends for this POLL (a count, not a time).
* After the POLL's CS rising edge the device drops INT, chains the queued messages into its TX
  DMA, then raises INT again: the first byte clocked out is then the 0xA5 of the first message.
* The host reads each message as **two transfers**: 6 header bytes, then ``len + 4`` (payload +
  check32). Every message but the last has MORE=1. If the queue empties first, the burst ends
  with a Null frame ``A5 ck 03 0C 00 (seq<<4)`` (type 3, no check32).
* Some silicon revisions add a pad byte after the header and after the check32. This is detected
  automatically: if check32 fails, retry with a 1-byte shift and keep that mode once it passes.
* Dummy bytes sent while reading must be 0x00 (``55 FF`` is the wake-up sequence and flushes
  the device queue).

The bus is abstract (:class:`SpiBus`): the CH347 USB bridge (:mod:`.ch347`) or the simulator
(:mod:`.simbus`). Nothing here is Windows-specific.
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Protocol

from ..hif import MAGIC, check8, check32
from ..logsetup import get_logger

log = get_logger("spi")

MSG_HOST_READ = 0x0C
POLL_TYPE_ACK = 1
IDLE_BYTES = (0x00, 0xFF)


class SpiBus(Protocol):
    name: str

    def xfer(self, tx: bytes) -> bytes:
        """One chip-select assertion, full duplex; returns len(tx) bytes."""

    def gpio(self) -> int | None:
        """Bit mask of the bridge's GPIO input levels (None if the bus has no GPIO)."""

    def close(self) -> None: ...


def build_poll(burst: int = 64, seq: int = 0) -> bytes:
    payload = struct.pack("<BBH", POLL_TYPE_ACK, 0, burst & 0xFFFF)
    flags = 0x01 | (1 << 2) | (1 << 4)            # type=1 (host->device), REQ, CHECK
    hdr = flags | (MSG_HOST_READ << 8) | (len(payload) << 16) | ((seq & 7) << 28)
    hb = struct.pack("<I", hdr)
    return bytes([MAGIC, check8(hb)]) + hb + payload + struct.pack("<I", check32(hb + payload))


def parse_header(h6: bytes) -> dict | None:
    if len(h6) < 6 or h6[0] != MAGIC or h6[1] != check8(h6[2:6]):
        return None
    raw = struct.unpack("<I", h6[2:6])[0]
    f = raw & 0xFF
    return {"type": f & 3, "check": (f >> 4) & 1, "more": (f >> 5) & 1, "ext": (f >> 6) & 1,
            "msg_id": (raw >> 8) & 0xFF, "length": (raw >> 16) & 0xFFF, "seq": (raw >> 28) & 7}


@dataclass
class SpiLinkStats:
    polls: int = 0
    polls_empty: int = 0             # POLL cycles that produced no new message
    int_timeouts: int = 0            # INT did not come back after a POLL
    messages: int = 0                # new (non-duplicate) data messages
    replays: int = 0                 # valid messages seen again (slave FIFO replaying old bytes) - dropped
    null_frames: int = 0             # Null frames accepted as end of burst
    null_ignored: int = 0            # Null frames ignored (inside a POLL response or stale replay)
    burst_timeouts: int = 0          # read loop ran out of time without an end-of-burst marker
    check32_fail: int = 0
    garbage_bytes: int = 0           # non-idle bytes skipped while looking for 0xA5
    syncs: int = 0                   # 4 KB drain reads (vendor state-sync rule)
    xfers: int = 0
    bytes_clocked: int = 0
    payload_bytes: int = 0
    xfer_s: float = 0.0              # time spent inside bus.xfer
    pad_mode: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["xfer_s"] = round(self.xfer_s, 4)
        d["mean_xfer_us"] = round(1e6 * self.xfer_s / self.xfers, 1) if self.xfers else None
        return d


class HifSpiLink:
    """Drives one POLL/read burst at a time.

    Every byte clocked in (including the MISO bytes returned while the POLL is sent) goes into one
    stream buffer: the device's TX DMA continues across CS toggles, so data can arrive at any time.
    Valid messages are recognised by check8 + check32 and **de-duplicated**: when the slave's TX FIFO
    runs dry it replays old bytes (seen on the bench 2026-09-25), which would otherwise look like new
    messages or a new Null frame.

    ``int_bit`` = GPIO bit carrying the radar INT, or None for blind polling (POLL, wait
    ``blind_wait_s``, then read until an end-of-burst marker or ``burst_timeout_s``)."""

    def __init__(self, bus: SpiBus, int_bit: int | None = None, burst: int = 64,
                 int_timeout_s: float = 0.02, blind_wait_s: float = 0.003, burst_timeout_s: float = 0.03,
                 max_xfer: int = 4096, pad_mode: bool | None = None):
        self.bus = bus
        self.int_bit = int_bit
        self.burst = burst
        self.int_timeout_s = int_timeout_s
        self.blind_wait_s = blind_wait_s
        self.burst_timeout_s = burst_timeout_s
        self.max_xfer = max_xfer
        self.pad_mode = bool(pad_mode)
        self._pad_known = pad_mode is not None
        self.stats = SpiLinkStats(pad_mode=self.pad_mode)
        self.first_bytes = bytearray()          # first raw bytes clocked in (for diagnostics)
        self.last_rx = b""                      # last raw bytes clocked in (for diagnostics)
        self.stream = bytearray()
        self._seen: deque[bytes] = deque(maxlen=256)
        self._seen_set: set[bytes] = set()
        self._last_null: bytes | None = None
        # D-017: diagnostics for the large-message failure (EXP-002)
        self.fail_dump = bytearray()            # first failing long messages, raw (saved by the engine)
        self._long_fail_run = 0
        self.long_ok: dict[str, int] = {}
        self.long_fail: dict[str, int] = {}

    # ---------------------------------------------------------------- bus helpers
    def _xfer(self, tx: bytes) -> bytes:
        t0 = time.perf_counter()
        rx = self.bus.xfer(tx)
        self.stats.xfer_s += time.perf_counter() - t0
        self.stats.xfers += 1
        self.stats.bytes_clocked += len(tx)
        if len(self.first_bytes) < 512:
            self.first_bytes.extend(rx[: 512 - len(self.first_bytes)])
        self.stream += rx
        self.last_rx = (self.last_rx + rx)[-128:]
        return rx

    def _read(self, n: int) -> bytes:
        out = bytearray()
        while n > 0:
            k = min(n, self.max_xfer)
            out += self._xfer(b"\x00" * k)
            n -= k
        return bytes(out)

    def sync(self, n: int = 4096):
        """Vendor HIF Ref §3.2.4.1: clock out ~4 KB to resynchronise with the device; drop the buffer."""
        self._read(n)
        self.stream.clear()
        self.stats.syncs += 1

    def int_high(self) -> bool | None:
        if self.int_bit is None:
            return None
        m = self.bus.gpio()
        return None if m is None else bool((m >> self.int_bit) & 1)

    def _wait_int(self, level: bool, timeout_s: float) -> bool:
        t_end = time.perf_counter() + timeout_s
        while True:
            if self.int_high() == level:
                return True
            if time.perf_counter() > t_end:
                return False
            time.sleep(0.0002)

    # ---------------------------------------------------------------- stream parsing
    @staticmethod
    def _c32_ok(hb: bytes, body: bytes) -> bool:
        if len(body) < 4:
            return False
        (rx,) = struct.unpack("<I", body[-4:])
        return rx == check32(hb[2:6] + body[:-4])

    def _extract(self) -> tuple[list[tuple[dict, bytes]], int]:
        """Pull complete messages off the front of ``self.stream``. Returns (messages, bytes still
        needed to complete the message at the front, 0 if none started)."""
        buf = self.stream
        out: list[tuple[dict, bytes]] = []
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                self.stats.garbage_bytes += sum(1 for b in buf if b not in IDLE_BYTES)
                buf.clear()
                return out, 0
            if i:
                self.stats.garbage_bytes += sum(1 for b in buf[:i] if b not in IDLE_BYTES)
                del buf[:i]
            if len(buf) < 6:
                return out, 6 - len(buf)
            h = parse_header(bytes(buf[:6]))
            if h is None:
                self.stats.garbage_bytes += 1
                del buf[:1]
                continue
            if h["type"] == 3 and h["msg_id"] == MSG_HOST_READ and h["length"] == 0:
                out.append((h, bytes(buf[:6])))
                del buf[:6]
                continue
            n = h["length"] + (4 if h["check"] else 0) + (4 if h["ext"] else 0)
            pad = 1 if self.pad_mode else 0
            total = 6 + pad + n + (pad if h["check"] else 0)
            if len(buf) < total:
                return out, total - len(buf)
            hb = bytes(buf[:6])
            body = bytes(buf[6 + pad: 6 + pad + n])
            if not h["check"] or self._c32_ok(hb, body):
                if not self._pad_known and h["check"]:
                    self._pad_known = True
                if n > 2000:
                    m = getattr(self.bus, "long_mode", "?")
                    self.long_ok[m] = self.long_ok.get(m, 0) + 1
                    self._long_fail_run = 0
                out.append((h, hb + body))
                del buf[:total]
                continue
            if not self.pad_mode and not self._pad_known:
                if len(buf) < 6 + 1 + n + 1:
                    return out, 6 + 1 + n + 1 - len(buf)
                alt = bytes(buf[7: 7 + n])
                if self._c32_ok(hb, alt):
                    self.pad_mode = self._pad_known = self.stats.pad_mode = True
                    log.warning("SPI: device inserts a pad byte after header and check32 (DMA pad) -> pad_mode on")
                    out.append((h, hb + alt))
                    del buf[: 6 + 1 + n + 1]
                    continue
            self.stats.check32_fail += 1
            if n > 2000:
                self._on_long_fail(bytes(buf[:total]))
            del buf[:1]

    def _on_long_fail(self, raw: bytes):
        m = getattr(self.bus, "long_mode", "?")
        self.long_fail[m] = self.long_fail.get(m, 0) + 1
        if len(self.fail_dump) < 64 * 1024:
            self.fail_dump += b"FAIL" + struct.pack("<I", len(raw)) + m.encode()[:12].ljust(12, b" ") + raw
        self._long_fail_run += 1
        if self._long_fail_run >= 3 and hasattr(self.bus, "rotate_long_mode"):
            self._long_fail_run = 0
            new = self.bus.rotate_long_mode()
            log.warning("SPI: 3 long messages failed check32 in mode %s -> trying %s (ok %s, fail %s)", m, new,
                        self.long_ok, self.long_fail)

    def _accept(self, msgs, out: list[bytes], in_poll_response: bool) -> bool:
        """Sort extracted messages; returns True when an end-of-burst marker was seen."""
        ended = False
        for h, pk in msgs:
            if h["type"] == 3:                               # Null frame
                if in_poll_response or (pk == self._last_null and not out):
                    self.stats.null_ignored += 1
                    continue
                self._last_null = pk
                self.stats.null_frames += 1
                ended = True
                continue
            sig = pk[:6] + pk[-4:]
            if sig in self._seen_set:
                self.stats.replays += 1
                continue
            if len(self._seen) == self._seen.maxlen:
                self._seen_set.discard(self._seen[0])
            self._seen.append(sig)
            self._seen_set.add(sig)
            out.append(pk)
            self.stats.messages += 1
            self.stats.payload_bytes += h["length"]
            if not h["more"]:
                ended = True
        return ended

    def read_burst(self) -> list[bytes]:
        """One POLL + read cycle. Returns new, clean HIF messages (no pads, no duplicates, no Null frames).
        With an INT line, returns [] immediately when the device has nothing to report."""
        if self.int_bit is not None and self.int_high() is False:
            return []
        out: list[bytes] = []
        self._xfer(build_poll(self.burst))
        self.stats.polls += 1
        msgs, _ = self._extract()
        self._accept(msgs, out, in_poll_response=True)      # leftovers of an earlier burst: keep data only
        if self.int_bit is not None:
            # INT drops at the POLL's CS edge and rises once the TX DMA is armed; a USB bridge may
            # miss the low phase, so only wait for "high" after a short delay (ESP32 host: 500 us).
            time.sleep(0.0005)
            if not self._wait_int(True, self.int_timeout_s):
                self.stats.int_timeouts += 1
                return out
        else:
            time.sleep(self.blind_wait_s)
        t_end = time.perf_counter() + self.burst_timeout_s
        idle = 0
        while True:
            msgs, need = self._extract()
            if self._accept(msgs, out, in_poll_response=False):
                break
            if len(out) > self.burst or time.perf_counter() > t_end:
                self.stats.burst_timeouts += 1
                break
            rx = self._read(need or 6)
            if all(b in IDLE_BYTES for b in rx):
                idle += 1
                if idle >= 4 and (out or self.int_bit is not None):
                    break
            else:
                idle = 0
        if not out:
            self.stats.polls_empty += 1
        return out


def watch_gpio(bus: SpiBus, seconds: float = 1.0, period_s: float = 0.0005) -> dict:
    """Sample the bridge GPIO; count level changes per bit. Used to find the radar INT line."""
    toggles = [0] * 8
    high = [0] * 8
    n = 0
    last = None
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        m = bus.gpio()
        if m is None:
            return {"available": False}
        n += 1
        for b in range(8):
            high[b] += (m >> b) & 1
        if last is not None:
            x = m ^ last
            for b in range(8):
                toggles[b] += (x >> b) & 1
        last = m
        time.sleep(period_s)
    dirs = bus.gpio_dir() if hasattr(bus, "gpio_dir") else None
    return {"available": True, "samples": n, "toggles": toggles, "dir_out_mask": dirs,
            "duty": [round(h / n, 3) if n else 0 for h in high], "last_mask": last}


# CH347T GPIO0..2 are SCK / MISO / SCS0 in mode 1 (go-ch347 pin map): never the radar INT.
SPI_GPIO_BITS = (0, 1, 2)


def pick_int_bit(watch: dict, min_toggles: int = 4, exclude=SPI_GPIO_BITS) -> int | None:
    if not watch.get("available"):
        return None
    t = watch["toggles"]
    b = max((i for i in range(8) if i not in exclude), key=lambda i: t[i])
    return b if t[b] >= min_toggles else None
