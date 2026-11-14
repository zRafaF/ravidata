"""Simulated RS6130 HIF-over-SPI device, for tests and the GUI simulator (EXP-002).

Wraps :class:`ravidata.sim.SimSerial`: its control port answers ``mmwc`` commands as usual and
the cube packets it would have written to the data UART are queued here instead and served
over the emulated SPI passive-reporting protocol (see :mod:`.link`). Device behaviours that
are modelled (from the vendor source): INT high while data is queued; POLL drops INT, arms
the burst and raises INT; MORE flags; Null frame at the end; data not polled within
``notify_timeout_s`` is discarded; optional pad bytes; per-transfer latency (USB bridge).
"""

from __future__ import annotations

import struct
import threading
import time
from collections import deque

from ..hif import MAGIC, check8, check32
from .link import MSG_HOST_READ, parse_header

INT_BIT_DEFAULT = 4


def _split_packets(buf: bytearray) -> list[bytes]:
    out = []
    while True:
        i = buf.find(MAGIC)
        if i < 0:
            buf.clear()
            return out
        del buf[:i]
        if len(buf) < 6:
            return out
        h = parse_header(bytes(buf[:6]))
        if h is None:
            del buf[:1]
            continue
        n = 6 + h["length"] + (4 if h["check"] else 0) + (4 if h["ext"] else 0)
        if len(buf) < n:
            return out
        out.append(bytes(buf[:n]))
        del buf[:n]


def _with_more(pk: bytes, more: bool) -> bytes:
    raw = struct.unpack("<I", pk[2:6])[0]
    raw = (raw | (1 << 5)) if more else (raw & ~(1 << 5))
    hb = struct.pack("<I", raw)
    body = pk[6:]
    if (raw >> 4) & 1:
        body = body[:-4] + struct.pack("<I", check32(hb + body[:-4]))
    return bytes([MAGIC, check8(hb)]) + hb + body


def null_frame(seq: int = 0) -> bytes:
    hb = bytes([0x03, MSG_HOST_READ, 0x00, (seq & 7) << 4])
    return bytes([MAGIC, check8(hb)]) + hb


class SimSpiBus:
    name = "SIM-SPI"

    def __init__(self, sim_serial, int_bit: int = INT_BIT_DEFAULT, pad: bool = False,
                 xfer_latency_s: float = 0.0, notify_timeout_s: float = 0.1, max_queue: int = 64,
                 stale_replay: bool = False, arm_delay_s: float = 0.0, has_int: bool = True):
        """``stale_replay``: when the TX DMA is empty the slave clocks out its last bytes again (seen on
        the bench). ``arm_delay_s``: time between POLL and armed data. ``has_int=False``: INT not wired."""
        self.stale_replay = stale_replay
        self.arm_delay_s = arm_delay_s
        self.has_int = has_int
        self._tail = bytearray(b"\x00")
        self._tail_i = 0
        self._armed_at = 0.0
        self.fake = sim_serial                 # SimSerial(route="data"); we drain its data port
        self.int_bit = int_bit
        self.pad = pad
        self.xfer_latency_s = xfer_latency_s
        self.notify_timeout_s = notify_timeout_s
        self.max_queue = max_queue
        self._rx = bytearray()
        self._queue: deque[tuple[float, bytes]] = deque()
        self._tx = bytearray()                 # armed burst bytes
        self._int = False
        self._int_since: float | None = None
        self._rearm_after = 0.0
        self._lock = threading.Lock()
        self.discarded = 0                     # packets dropped by the device model (not polled in time)
        self.overflow = 0
        self.polls = 0

    # ------------------------------------------------------------------ device model
    def _pump(self):
        n = self.fake.data.in_waiting
        if n:
            self._rx += self.fake.data.read(n)
            now = time.monotonic()
            for pk in _split_packets(self._rx):
                if len(self._queue) >= self.max_queue:
                    self._queue.popleft()
                    self.overflow += 1
                self._queue.append((now, pk))
        if not self._tx:
            now = time.monotonic()
            # vendor hif.c: INT is raised when data is queued; if no POLL arrives within sendFrameTo
            # (100 ms) INT drops and the WHOLE queue is flushed
            if self._int and self._int_since is not None and now - self._int_since > self.notify_timeout_s:
                self.discarded += len(self._queue)
                self._queue.clear()
                self._int = False
                self._int_since = None
                self._rearm_after = now + 0.001
            elif self._queue and not self._int and now >= self._rearm_after:
                self._int = True
                self._int_since = now

    def _arm(self, burst: int):
        pks = []
        while self._queue and len(pks) < burst:
            pks.append(self._queue.popleft()[1])
        data = bytearray()
        for k, pk in enumerate(pks):
            last = (k == len(pks) - 1)
            # vendor: MORE=1 on every data frame, cleared only on the frame that uses the last burst
            # slot and empties the queue; if the queue empties first, a Null frame (MORE=0) follows
            more = True if (not last or len(pks) < burst) else bool(self._queue)
            pk = _with_more(pk, more)
            data += self._padded(pk)
        if len(pks) < burst:
            data += self._padded(null_frame())
        self._tx = data
        self._int = True
        self._int_since = None

    def _padded(self, pk: bytes) -> bytes:
        if not self.pad:
            return pk
        h = parse_header(pk[:6])
        body = pk[6:]
        return pk[:6] + b"\x00" + body + (b"\x00" if h and h["check"] else b"")

    # ------------------------------------------------------------------ bus API
    def xfer(self, tx: bytes) -> bytes:
        if self.xfer_latency_s:
            time.sleep(self.xfer_latency_s)
        with self._lock:
            self._pump()
            if len(tx) >= 6 and tx[0] == MAGIC:
                h = parse_header(tx[:6])
                if h and h["type"] == 1 and h["msg_id"] == MSG_HOST_READ:
                    out = self._clock(len(tx))          # MISO keeps streaming whatever is pending
                    if not self._tx:                    # device idle: accept the POLL
                        burst = struct.unpack_from("<H", tx, 8)[0] if h["length"] >= 4 else 1
                        self.polls += 1
                        self._int = False
                        self._arm(max(1, burst))
                        self._armed_at = time.monotonic() + self.arm_delay_s
                    return out
            return self._clock(len(tx))

    def _clock(self, n: int) -> bytes:
        ready = time.monotonic() >= self._armed_at
        out = bytes(self._tx[:n]) if ready else b""
        if ready:
            del self._tx[:n]
            if out:
                self._tail = bytearray(out[-16:])
                self._tail_i = 0
            if not self._tx and self._int and out:
                self._int = False        # burst done; _pump re-raises INT if more data is queued
                self._int_since = None
        k = n - len(out)
        if k and self.stale_replay:
            fill = bytearray()
            for _ in range(k):
                fill.append(self._tail[self._tail_i % len(self._tail)])
                self._tail_i += 1
            return out + bytes(fill)
        return out + b"\x00" * k

    def gpio(self) -> int | None:
        with self._lock:
            self._pump()
            if not self.has_int:
                return 0b0110_1110            # bench 2026-09-25: constant levels, nothing toggles
            ready = time.monotonic() >= self._armed_at
            return (1 << self.int_bit) if (self._int and ready) else 0

    def close(self):
        self.fake.close()
