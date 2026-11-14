import struct
import time

import numpy as np
import pytest

from ravidata import hif
from ravidata.config import presets
from ravidata.hif import HifParser
from ravidata.spi.link import HifSpiLink, build_poll, parse_header, pick_int_bit, watch_gpio
from ravidata.spi.simbus import SimSpiBus, null_frame


def test_poll_frame_matches_vendor_bytes():
    # burst=256, seq 0: A5 35 15 0C 04 00 01 00 00 01 E9 F3 FB FE (vendor host driver, see EXP-002)
    assert build_poll(256).hex(" ") == "a5 35 15 0c 04 00 01 00 00 01 e9 f3 fb fe"
    h = parse_header(null_frame())
    assert h["type"] == 3 and h["msg_id"] == 0x0C and h["length"] == 0


class _QueueFake:
    """Minimal stand-in for SimSerial: a data port we push packets into."""

    class _Port:
        def __init__(self):
            self.buf = bytearray()

        @property
        def in_waiting(self):
            return len(self.buf)

        def read(self, n):
            out = bytes(self.buf[:n])
            del self.buf[:n]
            return out

    def __init__(self):
        self.data = self._Port()

    def close(self):
        pass


def _frames(n_frames, dwords):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n_frames):
        w = rng.integers(0, 2**32, dwords, dtype=np.uint64).astype(np.uint32)
        out.append((i, w, hif.build_cube_packets(i, w, max_payload=12 + hif.device_piece_bytes(dwords * 4))))
    return out


@pytest.mark.parametrize("pad,int_bit", [(False, 4), (True, 4), (False, None)])
def test_spi_link_bursts_reassemble_frames(pad, int_bit):
    fake = _QueueFake()
    bus = SimSpiBus(fake, pad=pad)
    link = HifSpiLink(bus, int_bit=int_bit, burst=8, max_xfer=1000)
    parser = HifParser()
    frames = _frames(5, 3 * 64 * 16)          # 12 KB cubes -> 3 KB pieces, several bursts
    got = []
    for i, w, pk in frames:
        fake.data.buf += pk
        for _ in range(20):
            for pkt in link.read_burst():
                got += parser.feed(pkt)
    assert [f.frame_index for f in got] == [0, 1, 2, 3, 4]
    for (i, w, _), f in zip(frames, got):
        assert np.array_equal(f.words, w)
    assert parser.stats.errors == {}
    assert link.stats.pad_mode == pad
    assert link.stats.check32_fail == 0


def test_spi_device_discards_unpolled_data():
    fake = _QueueFake()
    bus = SimSpiBus(fake, notify_timeout_s=0.01)
    fake.data.buf += _frames(1, 96)[0][2]
    assert bus.gpio() == 1 << 4                # INT high: data queued
    time.sleep(0.03)
    assert bus.gpio() == 0 and bus.discarded == 1     # flushed after the notify timeout


def test_int_bit_autodetect_and_spi_sim_end_to_end(tmp_path):
    from ravidata.engine import Engine
    eng = Engine(runs_dir=tmp_path)
    cfg = presets()["coarse-1d-20cm"].copy(link="spi", frame_period_ms=20)
    cmds = cfg.to_commands()
    i = cmds.index("comtype 0")
    assert cmds[i:i + 5] == ["comtype 0", "baudrate 1000000", "report cube 4294967295", "comtype 1",
                             "report cube 4294967295"]
    assert cmds.index("frame 20 0") < i
    eng.start_sim(cfg, "static-target-1m")
    t0 = time.time()
    while eng.frame_counter < 30 and time.time() - t0 < 20:
        time.sleep(0.05)
    src = eng.source
    d = eng.session.dir
    eng.stop()
    assert eng.frame_counter >= 30
    st = src.spi_stats()
    assert st["int_bit"] == 4 and st["check32_fail"] == 0 and st["messages"] >= 30
    assert "SPI link" in (d / "SESSION_REPORT.md").read_text()


def test_watch_gpio_picks_toggling_bit():
    class B:
        t0 = time.perf_counter()

        def gpio(self):
            return 0b0000_0001 | ((int((time.perf_counter() - self.t0) * 200) & 1) << 6)
    w = watch_gpio(B(), 0.1)
    assert pick_int_bit(w) == 6


def test_leakage_ema_removes_static_near_bins_only():
    from ravidata.cube import CubeShape
    from ravidata.frames import Frame
    from ravidata.processing.pipeline import Preprocess
    cfg = presets()["coarse-1d-20cm"]
    pre = Preprocess(cfg)
    rng = np.random.default_rng(1)
    leak = np.zeros((1, 3, 32), np.complex64)
    leak[..., 0], leak[..., 1], leak[..., 31] = 0.4, 0.3 * np.exp(1j), 0.2
    shp = CubeShape(1, 3, 32)
    for i in range(60):
        c = leak + 1e-4 * (rng.standard_normal(leak.shape) + 1j * rng.standard_normal(leak.shape))
        c[..., 10] += 0.01 * np.exp(1j * 0.7 * i)          # moving target at 2 m (outside the leak region)
        f = pre.process(Frame(seq=i, frame_index=i, t_host=0, t_mono=0, cube=c.astype(np.complex64), shape=shp,
                              words=None, source="t", extra={}))
    assert f.p_db[1] < 10 * np.log10(0.3 ** 2) - 30          # leakage bin suppressed by > 30 dB
    assert abs(f.p_db[10] - 10 * np.log10(1e-4)) < 1.0        # target untouched


@pytest.mark.parametrize("int_bit", [4, None])
def test_spi_link_survives_fifo_replay_and_slow_arming(int_bit):
    """Bench 2026-09-25: no INT visible, the slave replays stale bytes when its DMA is empty."""
    fake = _QueueFake()
    bus = SimSpiBus(fake, stale_replay=True, arm_delay_s=0.004, has_int=int_bit is not None)
    link = HifSpiLink(bus, int_bit=int_bit, burst=8, max_xfer=1000)
    parser = HifParser()
    frames = _frames(8, 96)
    got = []
    for i, w, pk in frames:
        fake.data.buf += pk
        for _ in range(6):
            for pkt in link.read_burst():
                got += parser.feed(pkt)
    assert [f.frame_index for f in got] == list(range(8))
    assert parser.stats.errors == {}


def test_ch347_long_transfer_keeps_one_cs_assertion():
    """Bench 2026-09-25: 4096 B WriteRead fails and a CS toggle inside a message corrupts it."""
    import ctypes
    from ravidata.spi.ch347 import Ch347Bus

    class FakeDll:
        def __init__(self):
            self.src = bytes(range(256)) * 20
            self.pos = 0
            self.cs_low = False
            self.cs_cycles = 0

        def CH347OpenDevice(self, i):
            return 1

        def CH347SetTimeout(self, *a):
            return True

        def CH347SPI_Init(self, *a):
            return True

        def CH347SPI_ChangeCS(self, i, st):
            if st and not self.cs_low:
                self.cs_cycles += 1
            self.cs_low = bool(st)
            return True

        def CH347SPI_WriteRead(self, i, cs, n, buf):
            if n >= 4096:
                return False
            if cs & 0x80:
                self.cs_cycles += 1
            data = self.src[self.pos:self.pos + n]
            self.pos += n
            ctypes.memmove(buf, data, n)
            return True

    d = FakeDll()
    bus = Ch347Bus(dll=d)
    rx = bus.xfer(b"\x00" * 3088)
    assert rx == d.src[:3088] and d.cs_cycles == 1 and not d.cs_low
    rx = bus.xfer(b"\x00" * 100)
    assert rx == d.src[3088:3188] and d.cs_cycles == 2


def test_link_rotates_long_transfer_mode_after_check32_failures():
    """D-017: if long messages keep failing check32 in one CH347 mode, the link tries the next mode."""
    fake = _QueueFake()

    class ModalBus(SimSpiBus):
        long_modes = ["single", "read", "manual_cs"]

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.long_mode = "single"

        def rotate_long_mode(self):
            i = self.long_modes.index(self.long_mode)
            self.long_mode = self.long_modes[(i + 1) % 3]
            return self.long_mode

        def xfer(self, tx):
            rx = super().xfer(tx)
            if len(tx) > 2048 and self.long_mode == "single":     # corrupt the tail, like the bench
                rx = rx[:2048] + bytes(len(rx) - 2048)
            return rx

    bus = ModalBus(fake)
    link = HifSpiLink(bus, int_bit=4, burst=8, max_xfer=4096)
    parser = HifParser()
    got = []
    for i, w, pk in _frames(12, 3 * 64 * 16):
        fake.data.buf += pk
        for _ in range(20):
            for pkt in link.read_burst():
                got += parser.feed(pkt)
    assert bus.long_mode == "read"
    assert link.long_fail.get("single", 0) >= 3 and link.long_ok.get("read", 0) > 0
    assert len(got) >= 5 and link.fail_dump.startswith(b"FAIL")


def test_ch347_read_mode_uses_dll_long_read():
    import ctypes
    from ravidata.spi.ch347 import Ch347Bus

    class FakeDll:
        src = bytes(range(256)) * 20

        def CH347OpenDevice(self, i): return 1
        def CH347SetTimeout(self, *a): return True
        def CH347SPI_Init(self, *a): return True
        def CH347SPI_ChangeCS(self, i, st): return True

        def CH347SPI_WriteRead(self, i, cs, n, buf):
            ctypes.memmove(buf, self.src[:n], n)
            return True

        def CH347SPI_Read(self, i, cs, olen, ilen, buf):
            n = ilen._obj.value
            ctypes.memmove(buf, self.src[:n], n)
            self.read_calls = getattr(self, "read_calls", 0) + 1
            return True

    d = FakeDll()
    bus = Ch347Bus(dll=d)
    assert bus.long_modes == ["single", "read", "manual_cs"]
    assert bus.rotate_long_mode() == "read"
    assert bus.xfer(b"\x00" * 3000) == d.src[:3000] and d.read_calls == 1
