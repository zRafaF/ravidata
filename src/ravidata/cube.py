"""Decode raw DataCube DWORDs into a complex array ``(n_dop, n_rx, n_range)``.

The raw DWORDs are always kept (and recorded), so the decoding convention can be changed
later without re-measuring. Two conventions matter (README D1, D3):

* ``iq_order``: ``"imag_low"`` (HIF manual §4.6.1: imag = low 16 bits, real = high 16 bits)
  or ``"real_low"`` (legacy V3 firmware).
* ``cube_order``: ``"tx_rx_dop_range"`` (legacy perception2 parser) or ``"dop_tx_rx_range"``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Derived, RadarConfig
from .logsetup import get_logger

log = get_logger("cube")
SCALE = 1.0 / 32768.0


@dataclass
class CubeShape:
    n_dop: int
    n_rx: int
    n_range: int
    inferred: bool = False
    note: str = ""

    @property
    def dwords(self) -> int:
        return self.n_dop * self.n_rx * self.n_range


def expected_shape(cfg: RadarConfig, d: Derived | None = None) -> CubeShape:
    d = d or cfg.derive()
    return CubeShape(n_dop=d.doppler_len, n_rx=cfg.n_rx * cfg.n_tx, n_range=d.range_fft_len)


def infer_shape(n_words: int, cfg: RadarConfig, expected: CubeShape) -> CubeShape:
    """Best guess when the chip sends a different cube size than the config predicts."""
    if n_words == expected.dwords:
        return expected
    nrx = cfg.n_rx * cfg.n_tx

    def keep_range():
        if n_words % (nrx * expected.n_range) == 0:
            k = n_words // (nrx * expected.n_range)
            return CubeShape(k, nrx, expected.n_range, True, f"kept range FFT {expected.n_range}, Doppler axis -> {k}")

    def keep_dop():
        if n_words % (nrx * expected.n_dop) == 0:
            r = n_words // (nrx * expected.n_dop)
            return CubeShape(expected.n_dop, nrx, r, True, f"kept Doppler axis {expected.n_dop}, range FFT -> {r}")

    # 1D configs most likely got a different range FFT length; 2D ones a different interval count
    for f in ((keep_dop, keep_range) if expected.n_dop == 1 else (keep_range, keep_dop)):
        s = f()
        if s is not None:
            return s
    if n_words % nrx == 0:
        return CubeShape(1, nrx, n_words // nrx, True, "treated as 1D")
    return CubeShape(1, 1, n_words, True, "not divisible by RX count; single channel view")


def words_to_iq(words: np.ndarray, iq_order: str = "imag_low") -> np.ndarray:
    """uint32 DWORDs -> complex64 normalised to +-1."""
    w = np.asarray(words, dtype=np.uint32)
    lo = (w & 0xFFFF).astype(np.uint16).view(np.int16).astype(np.float32)
    hi = (w >> 16).astype(np.uint16).view(np.int16).astype(np.float32)
    if iq_order == "imag_low":
        re, im = hi, lo
    else:
        re, im = lo, hi
    return (re + 1j * im).astype(np.complex64) * SCALE


def decode(words: np.ndarray, shape: CubeShape, cfg: RadarConfig) -> np.ndarray:
    iq = words_to_iq(words[: shape.dwords], cfg.iq_order)
    if cfg.cube_order == "dop_tx_rx_range":
        return iq.reshape(shape.n_dop, shape.n_rx, shape.n_range)
    # tx_rx_dop_range (tx folded into rx)
    return iq.reshape(shape.n_rx, shape.n_dop, shape.n_range).transpose(1, 0, 2).copy()


def iq_to_words(cube: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    """Inverse of :func:`decode` (used by the simulator and tests). ``cube`` is (n_dop, n_rx, n_range)."""
    if cfg.cube_order == "dop_tx_rx_range":
        flat = cube.reshape(-1)
    else:
        flat = cube.transpose(1, 0, 2).reshape(-1)
    re = np.clip(np.round(flat.real / SCALE), -32768, 32767).astype(np.int16).view(np.uint16).astype(np.uint32)
    im = np.clip(np.round(flat.imag / SCALE), -32768, 32767).astype(np.int16).view(np.uint16).astype(np.uint32)
    if cfg.iq_order == "imag_low":
        return (re << 16) | im
    return (im << 16) | re
