"""Chip-configuration emulation from a finer recording (EXP-013).

The chip delivers Hann-windowed FFTs (range: measured on the 1D runs, EXP-013 §1; Doppler: Hann-like, EXP-008).
An inverse FFT therefore gives back the windowed beat samples (range) or slow-time samples (Doppler). Keeping a
CENTRAL subset of them, removing the old window and applying a new Hann window of the subset's length is what
the chip would have produced with

* fewer ADC samples at the SAME slope and sample rate (``range_res`` coarser, ``range_max`` and IF high-pass
  corner unchanged): bandwidth B' = B * k / N;
* fewer Doppler intervals at the SAME interval spacing (``veloc_max`` unchanged, ``veloc_res`` coarser).

Frame-rate reduction is frame decimation. Not emulated (needs a recording): finer resolution, more intervals,
a different ``range_max`` (slope, HPF corner, aliasing), ``chirp_num``, TX/RX gain.

Targets must be injected into the ORIGINAL cube before emulation (``inject.run_injected(emulation=...)``) so the
target and the real scene degrade the same way.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .config import RadarConfig


@dataclass
class Emulation:
    range_res_mm: int | None = None     # coarser range resolution to emulate (None = keep)
    n_intervals: int | None = None      # fewer Doppler intervals (2D chip-Doppler recordings only; None = keep)
    fps_div: int = 1                    # keep every fps_div-th frame
    label: str = ""

    def cfg_for(self, cfg: RadarConfig) -> RadarConfig:
        c = dataclasses.replace(cfg, name=f"{cfg.name}~{self.name()}")
        if self.range_res_mm and self.range_res_mm > cfg.range_res_mm:
            c = dataclasses.replace(c, range_res_mm=int(self.range_res_mm))
        if self.n_intervals and cfg.fft_mode == 2 and cfg.veloc_max_mm_s:
            K0 = cfg.derive().n_intervals
            if self.n_intervals < K0:
                c = dataclasses.replace(c, veloc_res_mm_s=int(round(2 * cfg.veloc_max_mm_s / self.n_intervals)))
        if self.fps_div > 1:
            c = dataclasses.replace(c, frame_period_ms=cfg.frame_period_ms * self.fps_div)
        return c

    def name(self) -> str:
        if self.label:
            return self.label
        p = []
        if self.range_res_mm:
            p.append(f"{self.range_res_mm}mm")
        if self.n_intervals:
            p.append(f"K{self.n_intervals}")
        if self.fps_div > 1:
            p.append(f"fps/{self.fps_div}")
        return "+".join(p) or "orig"

    def is_identity(self, cfg: RadarConfig) -> bool:
        e = self.cfg_for(cfg)
        return e.range_res_mm == cfg.range_res_mm and e.veloc_res_mm_s == cfg.veloc_res_mm_s

    def apply(self, cube: np.ndarray, cfg: RadarConfig) -> np.ndarray:
        """cube (n_dop, n_rx, R) of the ORIGINAL config -> cube of cfg_for(cfg)."""
        d0, e = cfg.derive(), self.cfg_for(cfg)
        d1 = e.derive()
        x = cube.astype(np.complex128)
        if d1.n_samples < d0.n_samples:
            x = _resample_axis(x, -1, d0.n_samples, d1.n_samples, d1.range_fft_len, shifted=False)
        if cfg.fft_mode == 2 and d1.n_intervals < d0.n_intervals and cfg.dop_fft and x.shape[0] > 1:
            x = _resample_axis(x, 0, d0.n_intervals, d1.n_intervals, d1.doppler_len, shifted=True)
        return x.astype(np.complex64)


def _resample_axis(x, axis, n_old, n_new, out_len, shifted):
    """Hann-windowed FFT of n_old samples (zero padded to x.shape[axis]) -> of the central n_new samples."""
    x = np.moveaxis(x, axis, -1)
    if shifted:
        x = np.fft.ifftshift(x, axes=-1)
    t = np.fft.ifft(x, axis=-1)[..., :n_old]
    i0 = (n_old - n_new) // 2
    w_old = np.hanning(n_old)[i0:i0 + n_new]
    w_new = np.hanning(n_new)
    ratio = np.where(w_old > 1e-3, w_new / np.maximum(w_old, 1e-3), 0.0)
    seg = t[..., i0:i0 + n_new] * ratio
    # keep the chip's FFT scaling convention (unnormalised sum): amplitude of a tone ~ sum(window)
    y = np.fft.fft(seg, n=out_len, axis=-1)
    if shifted:
        y = np.fft.fftshift(y, axes=-1)
    return np.moveaxis(y, -1, axis)


def parse(spec: str) -> "Emulation | None":
    """'orig' | '56mm' | 'K16' | 'fps/2' | combinations with '+', e.g. '150mm+K16+fps/2'."""
    spec = spec.strip()
    if spec in ("", "orig"):
        return None
    e = Emulation()
    for part in spec.split("+"):
        part = part.strip()
        if part.endswith("mm"):
            e.range_res_mm = int(part[:-2])
        elif part.upper().startswith("K"):
            e.n_intervals = int(part[1:])
        elif part.startswith("fps/"):
            e.fps_div = int(part[4:])
        else:
            raise ValueError(f"unknown emulation part {part!r} (use 56mm, K16, fps/2)")
    return e


def config_grid(fft_mode: int = 2, frame_periods=(40, 80), max_util: float = 0.6, vmax_mm_s: int = 2000):
    """EXP-013: every chip config on a coarse grid that passes validation (no ERROR) and fits the link.
    Returns dicts sorted by resolution. range_max = res x 2^k (no zero padding)."""
    rows = []
    for res in (28, 30, 35, 40, 45, 50, 56, 60, 70, 80, 100, 150, 200):
        for nb in (64, 128, 256, 512):
            rmax = res * nb
            if not 3200 <= rmax <= 16000:
                continue
            for K in ((8, 16, 32) if fft_mode == 2 else (1,)):
                for fp in frame_periods:
                    c = RadarConfig(name=f"grid-{res}mm-{rmax}-K{K}-{fp}ms", fft_mode=fft_mode, freq_start_mhz=58_000,
                                    range_max_mm=rmax, range_res_mm=res, chirp_num=1 if fft_mode == 2 else 4,
                                    veloc_max_mm_s=vmax_mm_s if fft_mode == 2 else None,
                                    veloc_res_mm_s=int(2 * vmax_mm_s / K) if fft_mode == 2 else None,
                                    dop_fft=fft_mode == 2, frame_period_ms=fp, link="spi")
                    d = c.derive()
                    iss = [str(i) for i in c.validate()]
                    if any(i.startswith("ERROR") for i in iss) or d.link_utilisation > max_util:
                        continue
                    rows.append({"name": c.name, "res_mm": res, "rmax_m": rmax / 1000, "K": d.n_intervals,
                                 "v_res": (d.v_res_mm_s or 0) / 1000, "fps": 1000 / fp, "cube_kb": d.cube_bytes / 1024,
                                 "util": d.link_utilisation, "hpf_m": d.hpf_corner_range_m, "B_ghz": d.bandwidth_hz / 1e9,
                                 "warn": [i.split(":")[0].split()[-1] for i in iss if i.startswith("WARNING")]})
    return sorted(rows, key=lambda r: (r["res_mm"], r["rmax_m"], -r["K"], -r["fps"]))
