"""Shared pre-processing: cube -> per-range features used by every detector.

Deliberately simple and transparent (first iteration). Steps:

1. Optional complex background/leakage template subtraction (captured with nothing in
   front of the radar; see :meth:`Preprocess.start_background_capture`).
2. Reduce the Doppler/interval axis (``dop_reduce``).
3. Combine RX channels into one detection power (``combine``).
4. Track a per-bin noise floor (minimum-tracking EMA in dB).
5. Elevation from the RX-hi/RX-ref cross-spectrum phase; height = sensor height +
   r * sin(elevation + pitch).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..config import RadarConfig
from ..frames import Frame
from ..logsetup import get_logger
from .base import ParamSpec, Tunable
from .detectors import DETECTORS, Detector, DetectorOutput
try:
    from . import advanced  # registers the second-generation detectors in DETECTORS (private add-on)
except ImportError:
    advanced = None

log = get_logger("proc")
EPS = 1e-12


@dataclass
class Features:
    r_m: np.ndarray            # (R,) bin centre range
    p_rx_db: np.ndarray        # (n_rx, R) per-RX power after Doppler reduction
    p_db: np.ndarray           # (R,) combined detection power
    floor_db: np.ndarray       # (R,) tracked noise floor
    snr_db: np.ndarray         # (R,)
    elev_deg: np.ndarray       # (R,)
    height_m: np.ndarray       # (R,)
    valid: np.ndarray          # (R,) bool, inside near/far crop
    rd_db: np.ndarray | None   # (n_dop, R) combined, for display (None in 1D)
    bin_m: float
    # --- added 2026-09-25 (D-018) for the second-generation detectors (processing/advanced.py) ---
    p_lin: np.ndarray | None = None      # (R,) combined detection power, linear (same as p_db)
    p_dyn: np.ndarray | None = None      # (R,) power of (cube - slow complex background): what changes vs the radar
    noise: float = float("nan")          # thermal noise level of p_lin / p_dyn (linear, same units)
    rd_pow: np.ndarray | None = None     # (n_dop, R) Doppler power (chip FFT, or host FFT of slow time), centred
    rd_noise: float = float("nan")       # thermal noise per rd_pow cell
    v_axis: np.ndarray | None = None     # (n_dop,) radial velocity of each Doppler bin, m/s (sign not verified)
    t: float = 0.0                       # frame time, s (recording time in replay, arrival time live)
    p_beam: np.ndarray | None = None     # (R,) RX-hi + RX-ref coherent sum, steered (D-019); noise level = noise
    p_beam3: np.ndarray | None = None    # (R,) 3-RX coherent sum, steered in both planes
    ang_w: np.ndarray | None = None      # (R,) soft angle window 0..1 around the steering direction
    az_deg: np.ndarray | None = None     # (R,) angle in the az/ref plane (after calibration)
    rd_beam: np.ndarray | None = None    # (n_dop, R) steered RX-hi+RX-ref beam per range-Doppler cell (noise = rd_noise)
    rd_beam3: np.ndarray | None = None   # (n_dop, R) 3-RX steered beam per cell
    rd_angw: np.ndarray | None = None    # (n_dop, R) soft angle window per cell (rd_sigma_el/az_deg)
    el_conf: np.ndarray | None = None    # (R,) 0..1 coherence of the elevation estimate (1 = stable single scatterer)
    rd_self: np.ndarray | None = None    # (R,) learned zero-Doppler self-clutter power (same units as rd_pow), D-025
    rd_motion: float = 0.0               # fraction of rd power outside zero Doppler this frame
    rd_el: np.ndarray | None = None      # (n_dop, R) elevation of each range-Doppler cell (deg, calibrated)
    rd_angw_az: np.ndarray | None = None # (n_dop, R) horizontal-only angle window (pitch-independent)
    bin_scale: float = 1.0               # EXP-011: detector bin-count params x this (1D: ref_bin_mm / bin)
    rd_az_deg: np.ndarray | None = None  # (n_dop, R) horizontal angle per cell, 3x3 + az_tau_s smoothed (EXP-010)
    rd_az_conf: np.ndarray | None = None # (n_dop, R) 0..1 coherence of that angle (1 = one stable scatterer)
    az_conf: np.ndarray | None = None    # (R,) 1D: coherence of az_deg over the frame's intervals
    sensor_h: float = 0.9                # Preprocess sensor_height_m (for per-detection heights)
    pitch_deg: float = 0.0
    dt: float = 0.0                      # time since the previous frame, s


@dataclass
class ProcessResult:
    frame_seq: int
    frame_index: int
    t_host: float
    features: Features
    outputs: dict[str, DetectorOutput]
    primary: str
    proc_ms: float
    truth: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def primary_out(self) -> DetectorOutput:
        return self.outputs[self.primary]


class Preprocess(Tunable):
    SPECS = [
        ParamSpec("dop_reduce", "sum", kind="choice", choices=["sum", "max", "dc", "no_dc"],
                  help="How the Doppler/interval axis is collapsed (2D modes). 'dc' keeps only zero velocity."),
        ParamSpec("combine", "rx12", kind="choice", choices=["rx1", "rx2", "rx0", "rx12", "all", "delta12", "sum12"],
                  help="Detection power: single RX, non-coherent sums, or coherent sum/difference of RX1,RX2 "
                       "(after phase offset)."),
        ParamSpec("leak_mode", "ema", kind="choice", choices=["off", "ema"],
                  help="TX->RX leakage removal. 'ema': slow complex average of the near bins (< leak_max_m) and the "
                       "last leak_tail_bins bins, subtracted every frame (bench 2026-09-25: leakage is ~9800 LSB at "
                       "0.2 m, phase-stable to <0.6 deg; removal leaves ~1 %). A static target inside leak_max_m is "
                       "absorbed after ~leak_tau_s."),
        ParamSpec("leak_max_m", 0.5, 0.0, 1.5, 0.05, help="Leakage region: bins closer than this."),
        ParamSpec("leak_tail_bins", 2, 0, 8, 1, kind="int",
                  help="Leakage also wraps into the LAST bins (complex FFT, negative beat frequencies)."),
        ParamSpec("leak_tau_s", 20.0, 0.5, 300.0, 0.5, help="Time constant of the leakage average."),
        ParamSpec("subtract_background", False, kind="bool",
                  help="Subtract the captured complex background/leakage template."),
        ParamSpec("near_crop_m", 0.15, 0.0, 1.0, 0.025, help="Ignore bins closer than this."),
        ParamSpec("far_crop_frac", 0.2, 0.0, 0.5, 0.05, help="Ignore the last fraction of bins (vendor: last 20% unusable)."),
        ParamSpec("floor_mode", "global_pct", kind="choice", choices=["global_pct", "tracked"],
                  help="global_pct: per-frame percentile over valid bins (static targets never vanish). "
                       "tracked: per-bin minimum-tracking over time (static targets are slowly absorbed)."),
        ParamSpec("floor_pct", 30.0, 1.0, 90.0, 1.0, help="Percentile used by global_pct."),
        ParamSpec("floor_up", 0.005, 0.0, 1.0, 0.001, help="Noise floor rise rate per frame (slow). NOTE: a static obstacle is slowly absorbed into the floor."),
        ParamSpec("floor_down", 0.3, 0.0, 1.0, 0.05, help="Noise floor fall rate per frame (fast)."),
        ParamSpec("elev_hi_rx", 2, 0, 2, 1, kind="int", help="RX index above the reference (V4 orientation: RX2)."),
        ParamSpec("elev_ref_rx", 1, 0, 2, 1, kind="int", help="Reference RX (corner of the L: RX1)."),
        ParamSpec("elev_sign", 1.0, -1.0, 1.0, 2.0, help="+1 or -1: flips elevation if the mounting is inverted."),
        ParamSpec("phase_offset_deg", 166.0, -180.0, 180.0, 1.0,
                  help="RX-hi vs RX-ref phase at boresight (calibration). Default 166 = measured on the P1812 in two "
                       "wall runs, assuming the wall was straight ahead (EXP-007); 'Calibrate phase' overwrites it."),
        ParamSpec("az_rx", 0, 0, 2, 1, kind="int", help="RX forming the second pair with the reference RX "
                  "(corner of the L). Default RX0 (RX0-RX1)."),
        ParamSpec("az_offset_deg", -10.0, -180.0, 180.0, 1.0,
                  help="RX-az vs RX-ref phase at boresight (calibration; -10 = measured with the wall, EXP-007)."),
        ParamSpec("steer_el_deg", 0.0, -60.0, 60.0, 1.0,
                  help="Digital beam steering of the RX-hi/RX-ref pair (beam channels): angle of the main lobe."),
        ParamSpec("steer_az_deg", 0.0, -60.0, 60.0, 1.0, help="Same for the RX-az/RX-ref pair (beam3 channel)."),
        ParamSpec("ang_sigma_el_deg", 15.0, 2.0, 90.0, 1.0,
                  help="Angle window, hi/ref plane (vertical): returns BELOW el_min_deg fade with this Gaussian sigma; "
                       "above it the weight is 1, because head-height obstacles sit at +30..+70 deg (EXP-008)."),
        ParamSpec("el_min_deg", -20.0, -90.0, 30.0, 1.0,
                  help="Lower edge of the accepted elevation sector (sign: the desk/floor return read about -60 deg, "
                       "EXP-007; verify with a reflector moved up/down)."),
        ParamSpec("ang_sigma_az_deg", 90.0, 2.0, 90.0, 1.0, help="Same for the az/ref plane (90 = off)."),
        ParamSpec("rd_sigma_el_deg", 15.0, 2.0, 90.0, 1.0,
                  help="2D: per-cell vertical window, same one-sided shape as ang_sigma_el_deg / el_min_deg."),
        ParamSpec("rd_sigma_az_deg", 17.0, 2.0, 90.0, 1.0,
                  help="2D: angle window per cell, RX-az/RX-ref plane (horizontal in the V4 mounting): a narrow "
                       "horizontal FOV is acceptable (owner, 2026-09-25). 17 = tuned, EXP-008."),
        ParamSpec("self_tau_s", 8.0, 0.5, 120.0, 0.5,
                  help="2D self-clutter map (D-025): zero-Doppler power per range bin, learned ONLY in frames where the "
                       "scene moves relative to the radar (then whatever stays at zero Doppler moves WITH the radar: "
                       "hand, handle, mount, leakage residue). Time constant of that map."),
        ParamSpec("self_down_tau_s", 0.5, 0.05, 30.0, 0.05,
                  help="EXP-010: the self map is a LOWER ENVELOPE (dB): it follows a drop of zero-Doppler power with "
                       "this time constant and a rise with self_tau_s. Self clutter (hand, handle) is at zero Doppler in "
                       "every moving frame; a wall passes zero Doppler only when the approach reverses, so it no "
                       "longer leaks into the map. self_down_tau_s = self_tau_s gives the old plain average."),
        ParamSpec("self_motion_frac", 0.35, 0.0, 1.0, 0.05,
                  help="Fraction of power outside zero Doppler above which a frame counts as 'moving' for the self map."),
        ParamSpec("az_tau_s", 0.2, 0.0, 3.0, 0.02,
                  help="EXP-010: averaging time of the per-cell horizontal angle used by the FOV output levels "
                       "(power-weighted cross-spectrum, 3x3 range-Doppler neighbourhood; glint reduction)."),
        ParamSpec("el_tau_s", 0.3, 0.02, 3.0, 0.02,
                  help="Averaging time of the elevation/height estimate (power-weighted cross-spectrum of the dominant "
                       "cell per range bin; D-024)."),
        ParamSpec("ang_tau_s", 0.1, 0.0, 2.0, 0.01,
                  help="Smoothing of the per-bin cross-spectrum used for the angle window (0 = per frame)."),
        ParamSpec("sensor_height_m", 0.9, 0.0, 2.0, 0.05, help="Radar height above ground."),
        ParamSpec("pitch_deg", 0.0, -60.0, 60.0, 1.0, help="Radar boresight pitch (+ up)."),
        ParamSpec("dyn_tau_s", 2.0, 0.2, 60.0, 0.1,
                  help="Time constant of the slow complex background behind p_dyn (returns that do not change "
                       "relative to the radar - leakage, hand, handle - fade into it)."),
        ParamSpec("noise_tau_s", 2.0, 0.1, 60.0, 0.1, help="Smoothing of the thermal-noise estimate (rise time)."),
        ParamSpec("noise_down_tau_s", 0.3, 0.02, 60.0, 0.02,
                  help="EXP-011: the thermal-noise estimate is a LOWER ENVELOPE: it falls with this time constant "
                       "and rises with noise_tau_s. Hand-held, every bin of a short-range 1D config holds room "
                       "returns that change frame to frame, which inflated the frame-difference estimate by "
                       "~10 dB and made it swing with motion. = noise_tau_s gives the old plain average."),
        ParamSpec("ref_bin_mm", 30.0, 0.0, 200.0, 1.0,
                  help="EXP-011, 1D only: bin-count parameters of the detectors (CFAR guard/train, persistence "
                       "tolerance, peak width, tbd range slack) are defined at this bin size and scaled to the "
                       "actual bin (1.5 cm bins -> x2). 0 = no scaling. 2D parameters are not scaled."),
    ]

    def __init__(self, cfg: RadarConfig, **kw):
        super().__init__(**kw)
        self.cfg = cfg
        self.d = cfg.derive()
        self.floor_db: np.ndarray | None = None
        self.bg: np.ndarray | None = None
        self._bg_acc: np.ndarray | None = None
        self._bg_left = 0
        self._bg_n = 0
        self.leak: np.ndarray | None = None
        self._dyn_bg: np.ndarray | None = None
        self._prev_cube: np.ndarray | None = None
        self._sig2: float | None = None
        self._rd_noise: float | None = None
        self._t_prev: float | None = None
        self._cr = None
        self._el = None
        self._self = None

    # ------------------------------------------------------------ calibration
    def start_background_capture(self, n_frames: int = 40):
        self._bg_acc, self._bg_left, self._bg_n = None, n_frames, 0
        log.info("background capture started (%d frames) - keep the scene empty", n_frames)

    @property
    def capturing(self) -> bool:
        return self._bg_left > 0

    def calibrate_phase_offset(self, feats: Features, cube: np.ndarray) -> float | None:
        """Set phase offset so the strongest valid bin reads 0 deg elevation (target at boresight)."""
        hi, ref = self.p["elev_hi_rx"], self.p["elev_ref_rx"]
        cross = (cube[:, hi, :] * np.conj(cube[:, ref, :])).sum(axis=0)
        p = np.where(feats.valid & (feats.r_m >= 0.7), np.abs(cross), 0)   # skip the near zone (hand, mount)
        if p.max() <= 0:
            return None
        k = int(np.argmax(p))
        off = math.degrees(float(np.angle(cross[k])))
        self.set("phase_offset_deg", off)
        azr = self.p["az_rx"] % cube.shape[1]
        if azr != ref:
            caz = (cube[:, azr, :] * np.conj(cube[:, ref, :])).sum(axis=0)
            self.set("az_offset_deg", math.degrees(float(np.angle(caz[k]))))
        log.info("phase offset calibrated to %.1f deg at bin %d (%.2f m)", off, k, feats.r_m[k])
        return off

    def _remove_leak(self, cube: np.ndarray) -> np.ndarray:
        nd, nrx, R = cube.shape
        bin_m = self.d.bin_mm / 1000.0 if R == self.d.range_fft_len else (self.cfg.range_max_mm / 1000.0) / R
        k_near = min(R, int(math.ceil(self.p["leak_max_m"] / max(bin_m, 1e-6))))
        tail = min(R - k_near, int(self.p["leak_tail_bins"]))
        idx = np.r_[0:k_near, R - tail:R] if tail > 0 else np.arange(k_near)
        if idx.size == 0:
            return cube
        x = cube[:, :, idx]
        if self.leak is None or self.leak.shape != x.shape:
            self.leak = x.astype(np.complex64).copy()
        else:
            a = min(1.0, (self.cfg.frame_period_ms / 1000.0) / max(1e-3, self.p["leak_tau_s"]))
            self.leak += a * (x - self.leak)
        out = cube.copy()
        out[:, :, idx] = x - self.leak
        return out

    # ------------------------------------------------------------ main
    def process(self, fr: Frame) -> Features:
        cube = fr.cube
        nd, nrx, R = cube.shape
        if self._bg_left > 0:
            self._bg_acc = cube.astype(np.complex128) if self._bg_acc is None else self._bg_acc + cube
            self._bg_n += 1
            self._bg_left -= 1
            if self._bg_left == 0:
                self.bg = (self._bg_acc / self._bg_n).astype(np.complex64)
                self.p["subtract_background"] = True
                log.info("background template captured from %d frames (peak %.1f dBFS); subtract_background -> ON",
                         self._bg_n, 20 * np.log10(np.abs(self.bg).max() + EPS))
        if self.p["leak_mode"] == "ema":
            cube = self._remove_leak(cube)
        if self.p["subtract_background"] and self.bg is not None and self.bg.shape == cube.shape:
            cube = cube - self.bg

        bin_m = self.d.bin_mm / 1000.0
        if R != self.d.range_fft_len:          # inferred shape
            bin_m = (self.cfg.range_max_mm / 1000.0) / R
        r_m = np.arange(R) * bin_m
        pw = np.abs(cube) ** 2                 # (nd, nrx, R)
        mode = self.p["dop_reduce"]
        if nd > 1 and not self.cfg.dop_fft:
            # Slow time (chip Doppler FFT off): the 2nd axis holds interval samples (EXP-003).
            # 'dc' = coherent mean over intervals (zero Doppler); 'no_dc' = power after removing that mean (MTI).
            if mode == "sum":
                prx = pw.sum(axis=0)
            elif mode == "max":
                prx = pw.max(axis=0)
            elif mode == "dc":
                prx = np.abs(cube.mean(axis=0)) ** 2 * nd
            else:
                prx = (np.abs(cube - cube.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
        elif nd > 1:
            dc = nd // 2                        # chip Doppler output is centred (legacy data: static at bin 8 of 16)
            if mode == "sum":
                prx = pw.sum(axis=0)
            elif mode == "max":
                prx = pw.max(axis=0)
            elif mode == "dc":
                prx = pw[dc]
            else:
                prx = np.delete(pw, [max(0, dc - 1), dc, min(nd - 1, dc + 1)], axis=0).sum(axis=0)
        else:
            prx = pw[0]
        hi, ref = self.p["elev_hi_rx"] % nrx, self.p["elev_ref_rx"] % nrx
        cal = np.exp(-1j * math.radians(self.p["phase_offset_deg"]))
        comb = self.p["combine"]
        t_now = float(fr.extra.get("t_rec", fr.t_mono)) if fr.extra else float(fr.t_mono)
        dt = self.cfg.frame_period_ms / 1000.0 if self._t_prev is None else min(1.0, max(1e-3, t_now - self._t_prev))
        self._t_prev = t_now
        if comb in ("rx0", "rx1", "rx2"):
            p = prx[int(comb[2]) % nrx]
        elif comb == "rx12":
            p = 0.5 * (prx[hi] + prx[ref])
        elif comb == "all":
            p = prx.mean(axis=0)
        else:
            a, b = cube[:, hi, :] * cal, cube[:, ref, :]
            z = (a + b) if comb == "sum12" else (a - b)
            pz = np.abs(z) ** 2 / 2
            p = pz.sum(axis=0) if nd > 1 else pz[0]
        p_db = 10 * np.log10(p + EPS)
        p_rx_db = 10 * np.log10(prx + EPS)

        valid = np.zeros(R, dtype=bool)
        lo = int(math.ceil(self.p["near_crop_m"] / max(bin_m, 1e-6)))
        hi_i = int(R * (1 - self.p["far_crop_frac"]))
        valid[lo:hi_i] = True
        vals = p_db[valid] if valid.any() else p_db
        g = float(np.percentile(vals, self.p["floor_pct"]))
        if self.p["floor_mode"] == "global_pct":
            self.floor_db = np.full_like(p_db, g)
        else:
            if self.floor_db is None or self.floor_db.shape != p_db.shape:
                self.floor_db = np.full_like(p_db, g)     # start from the global level, not from the scene
            diff = p_db - self.floor_db
            rate = np.where(diff > 0, self.p["floor_up"], self.p["floor_down"])
            self.floor_db = self.floor_db + rate * diff
        snr = p_db - self.floor_db

        cross = (cube[:, hi, :] * np.conj(cube[:, ref, :])).sum(axis=0) * cal
        dphi = np.angle(cross) * self.p["elev_sign"]
        s = np.clip(dphi / (2 * math.pi * self.d.d_over_lambda), -1, 1)
        el = np.degrees(np.arcsin(s))
        h = self.p["sensor_height_m"] + r_m * np.sin(np.radians(el + self.p["pitch_deg"]))

        # ---- second-generation features (D-018) ----
        a_dyn = 1.0 - math.exp(-dt / max(1e-3, self.p["dyn_tau_s"]))
        if self._dyn_bg is None or self._dyn_bg.shape != cube.shape:
            self._dyn_bg = cube.astype(np.complex64).copy()
        xd = cube - self._dyn_bg
        self._dyn_bg += a_dyn * xd
        pdw = np.abs(xd) ** 2
        p_dyn = 0.5 * (pdw[:, hi, :] + pdw[:, ref, :]).sum(axis=0)
        # thermal noise per complex sample from frame-to-frame differences (static parts cancel); the 20th
        # percentile of an exponential variable is 0.223 x its mean, which keeps moving targets out of it.
        if self._prev_cube is not None and self._prev_cube.shape == cube.shape:
            dd = np.abs(cube[:, :, valid] - self._prev_cube[:, :, valid]) ** 2 / 2
            if dd.size:
                s2 = float(np.percentile(dd, 20)) / 0.223
                if self._sig2 is None:
                    self._sig2 = s2
                else:   # EXP-011: lower envelope in dB (fast down, slow up)
                    tau = self.p["noise_down_tau_s"] if s2 < self._sig2 else self.p["noise_tau_s"]
                    a_n = 1.0 - math.exp(-dt / max(1e-3, tau))
                    self._sig2 = float(self._sig2 * (s2 / self._sig2) ** a_n) if s2 > 0 else self._sig2
        self._prev_cube = cube.copy()
        noise = nd * self._sig2 if self._sig2 else float("nan")
        # ---- interferometry: calibrated phases, digitally steered beams, soft angle window (D-019) ----
        azr = self.p["az_rx"] % nrx
        dl = self.d.d_over_lambda
        ph_el = cube[:, hi, :] * np.conj(cube[:, ref, :]) * cal                  # (nd, R)
        cal_az = np.exp(-1j * math.radians(self.p["az_offset_deg"]))
        ph_az = cube[:, azr, :] * np.conj(cube[:, ref, :]) * cal_az
        st_el = np.exp(-1j * 2 * math.pi * dl * math.sin(math.radians(self.p["steer_el_deg"])))
        st_az = np.exp(-1j * 2 * math.pi * dl * math.sin(math.radians(self.p["steer_az_deg"])))
        x_ref = cube[:, ref, :]
        x_hi = cube[:, hi, :] * cal * st_el
        x_az = cube[:, azr, :] * cal_az * st_az
        # scaled so that thermal noise has the same level as p_lin (nd * sigma^2): a target in the steered
        # direction gains +3 dB (2 RX) / +4.8 dB (3 RX); off-axis returns lose up to all of it
        p_beam = (np.abs(x_ref + x_hi) ** 2 / 2).sum(axis=0)
        p_beam3 = (np.abs(x_ref + x_hi + x_az) ** 2 / 3).sum(axis=0) if nrx >= 3 else p_beam
        cr_el, cr_az = ph_el.sum(axis=0), ph_az.sum(axis=0)
        if self.p["ang_tau_s"] > 0:
            a_c = 1.0 - math.exp(-dt / self.p["ang_tau_s"])
            if getattr(self, "_cr", None) is None or self._cr[0].shape != cr_el.shape:
                self._cr = [cr_el.copy(), cr_az.copy()]
            else:
                self._cr[0] += a_c * (cr_el - self._cr[0])
                self._cr[1] += a_c * (cr_az - self._cr[1])
            cr_el, cr_az = self._cr
        def _ang(cr, steer):
            sv = np.clip(np.angle(cr) / (2 * math.pi * dl), -1, 1)
            return np.degrees(np.arcsin(sv)) - steer
        th_el = _ang(cr_el, self.p["steer_el_deg"])
        th_az = _ang(cr_az, self.p["steer_az_deg"])
        emin = self.p["el_min_deg"]
        ang_w = np.exp(-0.5 * (np.minimum(0.0, th_el - emin) / self.p["ang_sigma_el_deg"]) ** 2)
        if self.p["ang_sigma_az_deg"] < 89.5:
            ang_w = ang_w * np.exp(-0.5 * (th_az / self.p["ang_sigma_az_deg"]) ** 2)

        rd_pow = rd_noise = v_axis = rd_beam = rd_beam3 = rd_angw = rd_self = rd_el = rd_angw_az = None
        rd_az_deg = rd_az_conf = None
        az_conf = np.abs(ph_az.sum(axis=0)) / np.maximum(np.abs(ph_az).sum(axis=0), EPS)
        rd_motion = 0.0
        if nd > 1:
            if self.cfg.dop_fft:
                S = cube
            else:
                w = np.hanning(nd).astype(np.float32)
                S = np.fft.fftshift(np.fft.fft(cube * w[:, None, None], axis=0), axes=0) / math.sqrt(float((w ** 2).sum()))
            rd_pow = 0.5 * (np.abs(S[:, hi, :]) ** 2 + np.abs(S[:, ref, :]) ** 2)
            # per range-Doppler cell interferometry (D-021): steered beam + soft angle window, both planes.
            # Returns at the same range but different Doppler (e.g. floor vs obstacle while moving) get their
            # own angle here, which 1D interferometry cannot give them.
            s_ref, s_hi, s_az = S[:, ref, :], S[:, hi, :] * cal * st_el, S[:, azr, :] * cal_az * st_az
            rd_beam = np.abs(s_ref + s_hi) ** 2 / 2
            rd_beam3 = np.abs(s_ref + s_hi + s_az) ** 2 / 3 if nrx >= 3 else rd_beam
            e_el = _ang(S[:, hi, :] * np.conj(s_ref) * cal, self.p["steer_el_deg"])
            rd_el = np.degrees(np.arcsin(np.clip(np.angle(S[:, hi, :] * np.conj(s_ref) * cal) * self.p["elev_sign"]
                                                 / (2 * math.pi * self.d.d_over_lambda), -1, 1)))
            e_az = _ang(S[:, azr, :] * np.conj(s_ref) * cal_az, self.p["steer_az_deg"])
            rd_angw_az = np.exp(-0.5 * (e_az / self.p["rd_sigma_az_deg"]) ** 2)
            # EXP-010: smoothed per-cell horizontal angle + coherence for the FOV output levels
            from scipy.ndimage import uniform_filter
            c_az = S[:, azr, :] * np.conj(s_ref) * cal_az
            cs = uniform_filter(c_az.real, 3, mode="nearest") + 1j * uniform_filter(c_az.imag, 3, mode="nearest")
            ca = uniform_filter(np.abs(c_az), 3, mode="nearest")
            a_z = 1.0 if self.p["az_tau_s"] <= 0 else 1.0 - math.exp(-dt / self.p["az_tau_s"])
            if getattr(self, "_az", None) is None or self._az[0].shape != cs.shape:
                self._az = [cs.astype(np.complex128), ca.astype(np.float64)]
            else:
                self._az[0] += a_z * (cs - self._az[0])
                self._az[1] += a_z * (ca - self._az[1])
            rd_az_deg = _ang(self._az[0], 0.0)
            rd_az_conf = np.abs(self._az[0]) / np.maximum(self._az[1], EPS)
            rd_angw = rd_angw_az * np.exp(-0.5 * (np.minimum(0.0, e_el - emin) / self.p["rd_sigma_el_deg"]) ** 2)
            c = nd // 2
            far = np.abs(np.arange(nd) - c) >= max(2, nd // 4)
            cells = rd_pow[far][:, valid] if valid.any() else rd_pow[far]
            if cells.size:
                rn = float(np.percentile(cells, 20)) / 0.223
                a_n = 1.0 - math.exp(-dt / max(1e-3, self.p["noise_tau_s"]))
                self._rd_noise = rn if self._rd_noise is None else self._rd_noise + a_n * (rn - self._rd_noise)
            rd_noise = self._rd_noise
            # self-clutter map (D-025)
            dcm = np.abs(np.arange(nd) - c) <= 1
            tot = rd_pow[:, valid].sum() if valid.any() else rd_pow.sum()
            mfrac = float(rd_pow[~dcm][:, valid].sum() / tot) if tot > 0 and valid.any() else 0.0
            dc_now = rd_pow[dcm].max(axis=0)
            if getattr(self, "_self", None) is None or self._self.shape != dc_now.shape:
                self._self = np.zeros_like(dc_now)
                self._self_n = 0
            if mfrac > self.p["self_motion_frac"]:
                # EXP-010: lower envelope in dB (fast down, slow up), not a mean
                a_up = 1.0 - math.exp(-dt / max(1e-3, self.p["self_tau_s"]))
                a_dn = 1.0 - math.exp(-dt / max(1e-3, self.p["self_down_tau_s"]))
                now_db = 10 * np.log10(np.maximum(dc_now, 1e-12))
                if self._self_n == 0:
                    self._self_db = now_db
                else:
                    a = np.where(now_db < self._self_db, a_dn, a_up)
                    self._self_db = self._self_db + a * (now_db - self._self_db)
                self._self = 10 ** (self._self_db / 10)
                self._self_n += 1
            rd_self, rd_motion = self._self.copy(), mfrac
            vres = (self.cfg.veloc_res_mm_s or 0) / 1000.0
            if not vres and self.cfg.veloc_max_mm_s:
                vres = 2 * self.cfg.veloc_max_mm_s / 1000.0 / nd
            v_axis = (np.arange(nd) - c) * vres   # + = range increasing (receding): 92 % sign agreement, EXP-008

        rd = None
        if nd > 1:
            if self.cfg.dop_fft:
                rd = 10 * np.log10(pw.mean(axis=1) + EPS)
            else:                               # host-side Doppler for display only
                spec = np.fft.fftshift(np.fft.fft(cube * np.hanning(nd)[:, None, None], axis=0), axes=0)
                rd = 10 * np.log10((np.abs(spec) ** 2).mean(axis=1) / nd + EPS)
        # ---- height v2 (D-024, EXP-009): angle from the DOMINANT range-Doppler cell of each range bin (2D),
        # power-weighted average of its cross-spectrum over ~el_tau_s (glint reduction: fades, where the angle
        # of an extended target jumps, carry little power), and a coherence confidence |sum c| / sum |c|.
        # EXP-009: on a still wall at 62 dB SNR the per-frame phase wandered 14 deg (CRB 0.07 deg): glint /
        # multiple scatterers, not noise; bins at ~40 dB wandered 50-75 deg (mixtures).
        if nd > 1 and rd_pow is not None:
            kd = rd_pow.argmax(axis=0)
            cols = np.arange(R)
            cdom = S[kd, hi, cols] * np.conj(S[kd, ref, cols]) * cal
        else:
            cdom = cross
        a_e = 1.0 - math.exp(-dt / max(1e-3, self.p["el_tau_s"]))
        if getattr(self, "_el", None) is None or self._el[0].shape != cdom.shape:
            self._el = [cdom.astype(np.complex128), np.abs(cdom).astype(np.float64)]
        else:
            self._el[0] += a_e * (cdom - self._el[0])
            self._el[1] += a_e * (np.abs(cdom) - self._el[1])
        el_conf = np.abs(self._el[0]) / np.maximum(self._el[1], EPS)
        sv = np.clip(np.angle(self._el[0]) * self.p["elev_sign"] / (2 * math.pi * self.d.d_over_lambda), -1, 1)
        el = np.degrees(np.arcsin(sv))
        h = self.p["sensor_height_m"] + r_m * np.sin(np.radians(el + self.p["pitch_deg"]))
        return Features(r_m=r_m, p_rx_db=p_rx_db, p_db=p_db, floor_db=self.floor_db.copy(), snr_db=snr,
                        elev_deg=el, height_m=h, valid=valid, rd_db=rd, bin_m=bin_m,
                        bin_scale=(self.p["ref_bin_mm"] / 1000.0 / bin_m) if (nd <= 1 and self.p["ref_bin_mm"] > 0) else 1.0,
                        p_lin=p.astype(np.float64), p_dyn=p_dyn.astype(np.float64), noise=noise,
                        rd_pow=rd_pow, rd_noise=float("nan") if rd_noise is None else rd_noise, v_axis=v_axis,
                        t=t_now, dt=dt, p_beam=p_beam.astype(np.float64), p_beam3=p_beam3.astype(np.float64),
                        ang_w=ang_w, az_deg=th_az, rd_beam=rd_beam, rd_beam3=rd_beam3, rd_angw=rd_angw,
                        el_conf=el_conf, rd_self=rd_self, rd_motion=rd_motion, rd_el=rd_el, rd_angw_az=rd_angw_az,
                        rd_az_deg=rd_az_deg, rd_az_conf=rd_az_conf, az_conf=az_conf,
                        sensor_h=self.p["sensor_height_m"], pitch_deg=self.p["pitch_deg"])


class Processor:
    """Runs pre-processing and *all* registered detectors on every frame."""

    def __init__(self, cfg: RadarConfig, primary: str | None = None):
        # default primary: the add-on's best candidate for the mode if present, else the CFAR baseline
        primary = primary or (advanced.default_primary(cfg) if advanced is not None else "cfar")
        self.cfg = cfg
        self.pre = Preprocess(cfg)
        self.detectors: dict[str, Detector] = {k: cls() for k, cls in DETECTORS.items()}
        self.primary = primary if primary in self.detectors else next(iter(self.detectors))
        self._want_phase_cal = False

    def reset(self):
        self.pre.floor_db = None
        self.pre.leak = None
        self.pre._dyn_bg = self.pre._prev_cube = self.pre._sig2 = self.pre._rd_noise = self.pre._t_prev = None
        self.pre._cr = None
        self.pre._el = None
        self.pre._self = None
        self.pre._az = None
        for d in self.detectors.values():
            d.reset()

    def request_phase_calibration(self):
        self._want_phase_cal = True

    def params_snapshot(self) -> dict:
        return {"preprocess": dict(self.pre.p), "primary": self.primary,
                "primary_user": bool(getattr(self, "primary_user", False)),
                "detectors": {k: dict(v.p) for k, v in self.detectors.items()},
                "background_template": self.pre.bg is not None}

    def process(self, fr: Frame) -> ProcessResult:
        t0 = time.perf_counter()
        feats = self.pre.process(fr)
        if self._want_phase_cal:
            self._want_phase_cal = False
            self.pre.calibrate_phase_offset(feats, fr.cube)
        outs = {k: d.process(feats) for k, d in self.detectors.items()}
        ms = (time.perf_counter() - t0) * 1000
        return ProcessResult(frame_seq=fr.seq, frame_index=fr.frame_index, t_host=fr.t_host, features=feats,
                             outputs=outs, primary=self.primary, proc_ms=ms, truth=fr.extra.get("truth"),
                             extra={"max_range_m": self.detectors[self.primary].p["max_range_m"]})
