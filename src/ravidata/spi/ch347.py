"""WCH CH347 USB bridge as an :class:`~ravidata.spi.link.SpiBus` (Windows, ctypes).

Our own thin wrapper around WCH's ``CH347DLLA64.DLL`` (the USB-chip vendor's driver DLL,
installed with WCH's CH341PAR/CH347PAR driver package; it is not Possumic software and it is
not bundled here). Prototypes are from the WCH "CH347 Application Development Manual" V1.4
and cross-checked with pengwon/ch347-py. Nothing is installed or downloaded by this module.

Dev board setup (MRS6130-AIP-DEV-V1 user guide, Table 2-1): MODE SEL = ON (CH347T mode 1:
single UART + SPI + I2C); communication DIP 1,2 ON (UART0 = mmwc shell on the single COM port),
5,6,7 ON (SPI), 9 ON / 10 OFF (PA6 = INT to the USB chip); everything else OFF.

Which CH347 GPIO bit carries the radar INT is not documented for this board: ``ravidata
spi-probe`` finds it by watching which bit toggles while the radar streams.
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import POINTER, byref, c_ubyte, c_ulong, c_ushort, c_void_p

from ..logsetup import get_logger

log = get_logger("ch347")

# CH347T iClock index -> Hz (manual: 0=60 MHz ... 7=468.75 kHz)
CLOCKS_HZ = [60_000_000, 30_000_000, 15_000_000, 7_500_000, 3_750_000, 1_875_000, 937_500, 468_750]
DLL_NAMES = ("CH347DLLA64.DLL", "CH347DLL.DLL")


class SpiCfg(ctypes.Structure):
    # natural alignment, like the WCH header (mSpiCfgS)
    _fields_ = [("iMode", c_ubyte), ("iClock", c_ubyte), ("iByteOrder", c_ubyte),
                ("iSpiWriteReadInterval", c_ushort), ("iSpiOutDefaultData", c_ubyte),
                ("iChipSelect", c_ulong), ("CS1Polarity", c_ubyte), ("CS2Polarity", c_ubyte),
                ("iIsAutoDeativeCS", c_ushort), ("iActiveDelay", c_ushort), ("iDelayDeactive", c_ulong)]


def clock_index(hz: float) -> int:
    """Fastest CH347 clock <= hz."""
    for i, c in enumerate(CLOCKS_HZ):
        if c <= hz:
            return i
    return len(CLOCKS_HZ) - 1


def load_dll(path: str | None = None):
    if sys.platform != "win32":
        raise OSError("the CH347 SPI bridge is only supported on Windows (WCH CH347DLL)")
    names = [path] if path else list(DLL_NAMES)
    errs = []
    for n in names:
        try:
            dll = ctypes.WinDLL(n)
            break
        except OSError as e:
            errs.append(f"{n}: {e}")
    else:
        raise OSError("could not load the WCH CH347 DLL (" + "; ".join(errs) + "). Install WCH's CH347PAR/CH341PAR "
                      "driver (the SPI/I2C interface driver for the CH347; the installer is in this repo: "
                      "drivers/wch/CH341PAR.EXE), or pass --ch347-dll <path>.")
    dll.CH347OpenDevice.restype = c_void_p
    dll.CH347OpenDevice.argtypes = [c_ulong]
    dll.CH347CloseDevice.argtypes = [c_ulong]
    dll.CH347SetTimeout.argtypes = [c_ulong, c_ulong, c_ulong]
    dll.CH347SPI_Init.argtypes = [c_ulong, POINTER(SpiCfg)]
    dll.CH347SPI_GetCfg.argtypes = [c_ulong, POINTER(SpiCfg)]
    dll.CH347SPI_WriteRead.argtypes = [c_ulong, c_ulong, c_ulong, c_void_p]
    dll.CH347GPIO_Get.argtypes = [c_ulong, POINTER(c_ubyte), POINTER(c_ubyte)]
    dll.CH347GPIO_Set.argtypes = [c_ulong, c_ubyte, c_ubyte, c_ubyte]
    dll.CH347SPI_ChangeCS.argtypes = [c_ulong, c_ubyte]
    if hasattr(dll, "CH347SPI_Read"):   # BOOL CH347SPI_Read(iIndex, iChipSelect, oLength, PULONG iLength, ioBuffer)
        dll.CH347SPI_Read.argtypes = [c_ulong, c_ulong, c_ulong, POINTER(c_ulong), c_void_p]
        dll.CH347SPI_Read.restype = ctypes.c_bool
    dll.CH347SetIntRoutine.argtypes = [c_ulong, c_ubyte, c_ubyte, c_ubyte, c_ubyte, c_void_p]
    for f in ("CH347CloseDevice", "CH347SetTimeout", "CH347SPI_Init", "CH347SPI_GetCfg", "CH347SPI_WriteRead",
              "CH347GPIO_Get", "CH347GPIO_Set", "CH347SetIntRoutine", "CH347SPI_ChangeCS"):
        getattr(dll, f).restype = ctypes.c_bool
    return dll


class Ch347Bus:
    """One CH347 device index. ``xfer`` = one CS assertion (CS1, active low, SPI mode 0, MSB first)."""

    INVALID = c_void_p(-1).value

    def __init__(self, index: int = 0, clock_hz: float = 30e6, dll_path: str | None = None,
                 max_xfer: int = 4096, timeout_ms: int = 200, dll=None):
        self.index = index
        self.dll = dll if dll is not None else load_dll(dll_path)
        h = self.dll.CH347OpenDevice(index)
        if h in (None, self.INVALID):
            raise OSError(f"CH347OpenDevice({index}) failed: no CH347 in SPI mode (MODE SEL ON? CH347PAR driver?)")
        self.dll.CH347SetTimeout(index, timeout_ms, timeout_ms)
        cfg = SpiCfg(iMode=0, iClock=clock_index(clock_hz), iByteOrder=1, iSpiWriteReadInterval=0,
                     iSpiOutDefaultData=0x00, iChipSelect=0x80, CS1Polarity=0, CS2Polarity=0,
                     iIsAutoDeativeCS=1, iActiveDelay=0, iDelayDeactive=0)
        if not self.dll.CH347SPI_Init(index, byref(cfg)):
            raise OSError("CH347SPI_Init failed")
        self.clock_hz = CLOCKS_HZ[cfg.iClock]
        self.max_xfer = max_xfer
        self.chunk = 2048
        self.long_xfers = 0
        # How transfers longer than `chunk` are done (EXP-002 open issue: messages > ~2 kB always failed
        # check32 with "manual_cs"). Tried in this order; HifSpiLink rotates to the next mode after
        # repeated check32 failures on long messages (see rotate_long_mode). D-017.
        #   single    : one CH347SPI_WriteRead with CS control, up to single_max bytes
        #   read      : CH347SPI_Read(oLength=0, iLength=n) - the DLL's own long read under one CS
        #   manual_cs : ChangeCS(1) + several WriteRead(no CS) + ChangeCS(0)  (previous behaviour)
        self.long_modes = ["single", "read", "manual_cs"]
        if not hasattr(self.dll, "CH347SPI_Read"):
            self.long_modes.remove("read")
        self.long_mode = self.long_modes[0]
        self.single_max = 4000
        self.mode_stats: dict[str, int] = {}
        self.name = f"CH347#{index}@{self.clock_hz/1e6:g}MHz"
        log.info("CH347 device %d opened, SPI mode 0, %.3f MHz (requested %.3f)", index, self.clock_hz / 1e6,
                 clock_hz / 1e6)

    def xfer(self, tx: bytes) -> bytes:
        """ONE chip-select assertion for the whole transfer, whatever its length.

        Bench 2026-09-25: a single CH347SPI_WriteRead of 4096 B fails, and splitting a message over two
        CS assertions corrupts it (every 2060 B cube piece failed check32 once it was cut at 2048 B).
        Long transfers therefore hold CS low manually (CH347SPI_ChangeCS 1/0) around several
        WriteRead calls without CS control (iChipSelect bit 7 = 0). The chunk size adapts down on failure."""
        tx = bytes(tx)
        if len(tx) > self.chunk and self.long_mode != "manual_cs":
            rx = self._xfer_long(tx)
            if rx is not None:
                self.long_xfers += 1
                self.mode_stats[self.long_mode] = self.mode_stats.get(self.long_mode, 0) + 1
                return rx
        if len(tx) <= self.chunk:
            buf = ctypes.create_string_buffer(tx, len(tx))
            if self.dll.CH347SPI_WriteRead(self.index, 0x80, len(tx), buf):
                return buf.raw[: len(tx)]
            if len(tx) <= 64:
                raise OSError(f"CH347SPI_WriteRead({len(tx)} B) failed")
            self.chunk = max(64, len(tx) // 2)
            log.warning("CH347SPI_WriteRead(%d B) failed -> chunk %d B with manual CS", len(tx), self.chunk)
        out = bytearray()
        if not self.dll.CH347SPI_ChangeCS(self.index, 1):
            raise OSError("CH347SPI_ChangeCS(assert) failed")
        try:
            i = 0
            while i < len(tx):
                part = tx[i: i + self.chunk]
                buf = ctypes.create_string_buffer(part, len(part))
                if not self.dll.CH347SPI_WriteRead(self.index, 0x00, len(part), buf):
                    if self.chunk > 64 and len(part) > 64:
                        self.chunk //= 2
                        log.warning("CH347SPI_WriteRead(%d B) failed -> chunk reduced to %d B", len(part),
                                    self.chunk)
                        continue
                    raise OSError(f"CH347SPI_WriteRead({len(part)} B, CS held) failed")
                out += buf.raw[: len(part)]
                i += len(part)
        finally:
            self.dll.CH347SPI_ChangeCS(self.index, 0)
        self.long_xfers += 1
        return bytes(out)

    def _xfer_long(self, tx: bytes) -> bytes | None:
        """Long transfer with the current non-manual mode; None = mode not usable, fall through."""
        n = len(tx)
        if self.long_mode == "single":
            if n > self.single_max:
                return None
            buf = ctypes.create_string_buffer(tx, n)
            if self.dll.CH347SPI_WriteRead(self.index, 0x80, n, buf):
                return buf.raw[:n]
            self.single_max = max(self.chunk, n - 1)
            log.warning("CH347SPI_WriteRead(%d B, one CS) failed -> single_max %d B", n, self.single_max)
            return None
        if self.long_mode == "read" and not any(tx):
            ln = c_ulong(n)
            buf = ctypes.create_string_buffer(n)
            if self.dll.CH347SPI_Read(self.index, 0x80, 0, byref(ln), buf):
                got = int(ln.value)
                return buf.raw[:n] if got >= n else buf.raw[:got] + b"\x00" * (n - got)
            log.warning("CH347SPI_Read(%d B) failed", n)
            return None
        return None

    def rotate_long_mode(self) -> str:
        """Switch to the next long-transfer mode (called by the link after repeated check32 failures)."""
        i = self.long_modes.index(self.long_mode) if self.long_mode in self.long_modes else -1
        self.long_mode = self.long_modes[(i + 1) % len(self.long_modes)]
        log.warning("CH347: long-transfer mode -> %s", self.long_mode)
        return self.long_mode

    def gpio(self) -> int | None:
        d, v = c_ubyte(0), c_ubyte(0)
        if not self.dll.CH347GPIO_Get(self.index, byref(d), byref(v)):
            return None
        return int(v.value)

    def gpio_dir(self) -> int | None:
        """Bit mask of GPIOs configured as OUTPUT (1 = output). An output pin reads back its latch, not the
        radar INT, so the INT pin must be an input."""
        d, v = c_ubyte(0), c_ubyte(0)
        if not self.dll.CH347GPIO_Get(self.index, byref(d), byref(v)):
            return None
        return int(d.value)

    def gpio_make_input(self, mask: int) -> bool:
        """Set the GPIOs in ``mask`` to input (CH347GPIO_Set iEnable=mask, iSetDirOut=0)."""
        ok = bool(self.dll.CH347GPIO_Set(self.index, mask & 0xFF, 0, 0))
        log.info("CH347 GPIO_Set: bits 0x%02X -> input: %s", mask & 0xFF, ok)
        return ok

    def int_scan(self, pins=(3, 4, 5, 6, 7), seconds: float = 0.6, mode: int = 2) -> dict:
        """Count interrupt callbacks per GPIO pin using the CH347 interrupt engine (CH347SetIntRoutine,
        TripMode 0 falling / 1 rising / 2 both). Independent from level polling."""
        counts: dict[int, int] = {}
        proto = ctypes.WINFUNCTYPE(None, POINTER(c_ubyte))
        for pin in pins:
            n = [0]

            def cb(_status, n=n):
                n[0] += 1
            fn = proto(cb)
            ok = self.dll.CH347SetIntRoutine(self.index, pin, mode, 8, 0, ctypes.cast(fn, c_void_p))
            time.sleep(seconds)
            self.dll.CH347SetIntRoutine(self.index, 8, 0, 8, 0, None)
            counts[pin] = n[0] if ok else -1
        log.info("CH347 interrupt scan (%.1f s per pin, mode %d): %s", seconds, mode, counts)
        return counts

    def close(self):
        try:
            self.dll.CH347CloseDevice(self.index)
        except Exception:  # pragma: no cover
            pass
