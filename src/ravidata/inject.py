"""Synthetic target injection into REAL recordings: exact ground truth on real clutter (D-022, EXP-008).

A point target on a known approach path is added to every frame of a recorded cube, with the range and
Doppler response of an FFT, the per-RX phase of its angle (including the measured calibration offsets),
1/r^4 power and a rough 2-way antenna pattern. Detectors are then scored on whether they report *that*
target: hit / clutter (something nearer) / miss, per frame, plus the warning distance.

What it is good for: ranking detectors and tuning parameters on the owner's real clutter, and Pd-vs-SNR /
angle curves, without anyone walking into a wall 300 times. What it is not: a model of real obstacles
(extended targets, multipath, the owner's own motion). Real marker runs (EXP-004) stay the final check.

Assumptions (EXP-008): Hann window in range and Doppler (legacy Doppler neighbours at -6 dB), r_dot =
+v_axis, RX roles of Preprocess (hi = RX2, ref = RX1, az = RX0), pattern: flat to 40 deg, -5 dB at 56 deg,
-12 dB beyond 60 deg (datasheet YOZ + side, rough).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import RadarConfig
from .cube import CubeShape
from .frames import Frame
from .processing.pipeline import Processor
from .sources import open_recording

C0 = 299_792_458.0


@dataclass
class Scenario:
    start_s: float = 2.0          # time in the recording when the approach starts
    r0: float = 2.6               # slant range at start (m)
    r_end: float = 0.4            # stop injecting below this slant range
    speed: float = 1.0            # walking speed towards the obstacle (m/s)
    height_m: float = 1.7         # obstacle height above ground
    sensor_h: float = 0.85        # radar height above ground (boresight horizontal)
    lateral_m: float = 0.0        # sideways offset of the obstacle
    snr1m_db: float = 25.0        # per-cell SNR the target would have at 1 m on boresight
    pattern: bool = True          # apply the rough 2-way pattern vs elevation
    label: str = ""
    # extended obstacles (EXP-009): n_scat point scatterers spread over extent_m in depth and +-spread_deg in
    # both angles; the total power is snr1m_db (so each scatterer is weaker). truth = the FRONT edge.
    extent_m: float = 0.0
    n_scat: int = 1
    spread_deg: float = 0.0
    hold_s: float = 0.0           # speed 0: a static obstacle held at r0 for hold_s (user standing still)
    kind: str = "point"           # point | extended | static | small_near (label for reports)

    def state(self, t: float):
        """(r, el_deg, az_deg, r_dot) at recording time t, or None when not active."""
        if t < self.start_s:
            return None
        if self.speed <= 0 and t > self.start_s + self.hold_s:
            return None
        dh = self.height_m - self.sensor_h
        x0 = math.sqrt(max(1e-6, self.r0 ** 2 - dh ** 2 - self.lateral_m ** 2))
        x = x0 - self.speed * (t - self.start_s)
        if x <= 0:
            return None
        r = math.sqrt(x * x + dh * dh + self.lateral_m ** 2)
        if r < self.r_end:
            return None
        el = math.degrees(math.atan2(dh, math.hypot(x, self.lateral_m)))
        az = math.degrees(math.atan2(self.lateral_m, x))
        return r, el, az, -self.speed * x / r


def pattern_db(el_deg: float) -> float:
    a = abs(el_deg)
    if a <= 40:
        return 0.0
    if a <= 56:
        return -5.0 * (a - 40) / 16
    if a <= 60:
        return -5.0 - 7.0 * (a - 56) / 4
    return -12.0


def _resp(q: float, n: int, out_len: int | None = None) -> np.ndarray:
    """Normalised (peak ~1) FFT response of a Hann-windowed tone at fractional bin q."""
    m = np.arange(n)
    w = np.hanning(n)
    tone = w * np.exp(2j * np.pi * q * m / n)
    F = np.fft.fft(tone, n=out_len or n)
    return F / w.sum()


@dataclass
class Injector:
    cfg: RadarConfig
    scenario: Scenario
    noise_sigma2: float                     # per-sample complex noise power of the cube (normalised units)
    phase_offset_deg: float = 166.0
    az_offset_deg: float = -10.0
    hi: int = 2
    ref: int = 1
    az: int = 0
    rng_phase: float = 0.0
    truth: list = field(default_factory=list)

    def __post_init__(self):
        d = self.cfg.derive()
        self.d = d
        self.bin_m = d.bin_mm / 1000.0
        self.lam = d.wavelength_mm / 1000.0
        self.dl = d.d_over_lambda
        self.vres = (self.cfg.veloc_res_mm_s or 0) / 1000.0

    def apply(self, cube: np.ndarray, t: float) -> np.ndarray:
        st = self.scenario.state(t)
        if st is None:
            self.truth.append(np.nan)
            return cube
        sc = self.scenario
        if sc.n_scat <= 1:
            self.truth.append(st[0])
            return self._add(cube, *st, 0.0)
        rng = np.random.default_rng(int(sc.start_s * 1000) + 7)
        offs = [(0.0, 0.0, 0.0, 1.0)] + [(float(rng.uniform(0, sc.extent_m)), float(rng.uniform(-sc.spread_deg, sc.spread_deg)),
                                        float(rng.uniform(-sc.spread_deg, sc.spread_deg)), float(rng.rayleigh(0.8)))
                                       for _ in range(sc.n_scat - 1)]
        wsum = sum(w * w for *_, w in offs)
        r, el, az, rdot = st
        self.truth.append(r)
        out = cube
        for dr, de, da, w in offs:
            out = self._add(out, r + dr, el + de, az + da, rdot, 10 * math.log10(w * w / wsum), phase_seed=dr * 1e3)
        return out

    def _add(self, cube, r, el, az, rdot, rel_db, phase_seed=0.0):
        nd, nrx, R = cube.shape
        snr = 10 ** ((self.scenario.snr1m_db + rel_db - 40 * math.log10(max(r, 0.1)) +
                      (pattern_db(el) if self.scenario.pattern else 0.0)) / 10)
        amp = math.sqrt(snr * self.noise_sigma2)
        rr = _resp(r / self.bin_m, R)
        carrier = np.exp(1j * (4 * np.pi * r / self.lam + self.rng_phase + phase_seed))
        if nd > 1:
            qv = rdot / self.vres if self.vres else 0.0
            if self.cfg.dop_fft:
                dd = np.fft.fftshift(_resp(qv, nd))
            else:
                dd = np.exp(2j * np.pi * qv * np.arange(nd) / nd)
        else:
            dd = np.ones(1)
        ph = np.zeros(nrx, complex)
        ph[self.ref % nrx] = 1.0
        ph[self.hi % nrx] = np.exp(1j * (2 * np.pi * self.dl * math.sin(math.radians(el)) + math.radians(self.phase_offset_deg)))
        if nrx > 2:
            ph[self.az % nrx] = np.exp(1j * (2 * np.pi * self.dl * math.sin(math.radians(az)) + math.radians(self.az_offset_deg)))
        sig = amp * carrier * dd[:, None, None] * ph[None, :, None] * rr[None, None, :]
        return (cube + sig).astype(np.complex64)


def estimate_sigma2(rec, n: int = 60) -> float:
    """Per-sample complex noise power from frame-to-frame differences (same estimator as Preprocess)."""
    k = min(rec.n - 1, n)
    vals = []
    for i in range(1, k + 1):
        a, b = rec.get_cube(i), rec.get_cube(i - 1)
        if a.shape != b.shape:
            continue
        R = a.shape[-1]
        dd = np.abs(a[..., int(R * 0.3): int(R * 0.8)] - b[..., int(R * 0.3): int(R * 0.8)]) ** 2 / 2
        vals.append(np.percentile(dd, 20) / 0.223)
    return float(np.median(vals)) if vals else 1e-9


def run_injected(path, scenarios: list[Scenario], detectors: dict[str, dict], preprocess: dict | None = None,
                 t_window: float | None = None, lo: float = 0.6, hi: float = 1.8,
                 inj_az_err_deg: float = 0.0, emulation=None, tol_m: float | None = None) -> list[dict]:
    """Replay a recording once per scenario with the target injected; score every detector.
    Only obstacles inside the detector's output FOV (Detection.in_fov, EXP-010) count as reported.
    inj_az_err_deg: the injected target's RX-az phase offset differs from the pipeline's by this much
    (simulates a stale horizontal calibration).
    emulation: an ``emulate.Emulation``: targets are injected into the original cube, then the cube is turned
    into the emulated chip config (coarser range / fewer intervals / fewer frames) before processing (EXP-013).
    tol_m: fixed hit tolerance (default max(0.1 m, 2 bins)); use a fixed value when comparing bin sizes, else
    coarse configs get a looser test."""
    rec = open_recording(path)
    s2 = estimate_sigma2(rec)
    pre = {"phase_offset_deg": 166.0, "az_offset_deg": -10.0, **(preprocess or {})}
    pre.setdefault("sensor_height_m", scenarios[0].sensor_h if scenarios else 0.85)
    out = []
    from .processing.detectors import DETECTORS
    for sc in scenarios:
        inj = Injector(rec.cfg, sc, s2, pre["phase_offset_deg"], pre["az_offset_deg"] + inj_az_err_deg,
                       rng_phase=float(np.random.default_rng(int(sc.start_s * 1000)).uniform(0, 2 * np.pi)))
        pcfg = emulation.cfg_for(rec.cfg) if emulation is not None else rec.cfg
        proc = Processor(pcfg)
        for k, v in pre.items():
            proc.pre.set(k, v)
        dets = {}
        for name, spec in detectors.items():
            spec = dict(spec)
            kind = spec.pop("kind", name)
            dets[name] = DETECTORS[kind](**spec)
        proc.detectors, proc.primary = dets, next(iter(dets))
        t0 = rec.times[0]
        span = t_window or ((sc.r0 - sc.r_end) / sc.speed + 2.0 if sc.speed > 0 else sc.hold_s + 1.0)
        i0 = int(np.searchsorted(rec.times - t0, max(0.0, sc.start_s - 1.5)))
        i1 = int(np.searchsorted(rec.times - t0, sc.start_s + span))
        truth, outs, alls, hts = [], {n: [] for n in dets}, {n: [] for n in dets}, {n: [] for n in dets}
        step = emulation.fps_div if emulation is not None else 1
        for i in range(i0, min(i1, rec.n), max(1, step)):
            t = float(rec.times[i] - t0)
            cube = inj.apply(rec.get_cube(i), t)
            if emulation is not None:
                cube = emulation.apply(cube, rec.cfg)
            fr = Frame(i, int(rec.frame_index[i]), 0.0, t, cube, CubeShape(*cube.shape), extra={"t_rec": t})
            res = proc.process(fr)
            truth.append(inj.truth[-1])
            for n, o in res.outputs.items():
                outs[n].append(o.nearest.range_m if o.nearest is not None else np.nan)
                alls[n].append([d.range_m for d in o.detections if d.cls == "obstacle" and d.in_fov])
                tr_ = inj.truth[-1]
                near_d = [d for d in o.detections if d.cls == "obstacle" and d.in_fov and tr_ == tr_
                          and abs(d.range_m - tr_) <= 0.2]
                hts[n].append(near_d[0].height_m if near_d else np.nan)
        truth = np.array(truth)
        bin_m = pcfg.derive().bin_mm / 1000.0
        tol = tol_m if tol_m is not None else max(0.1, 2 * bin_m)
        for n, y in outs.items():
            y = np.array(y)
            m = ~np.isnan(truth) & (truth >= min(lo, 0.45 if sc.kind != "point" else lo)) & (truth <= hi)
            if not m.any():
                continue
            yy, tt = y[m], truth[m]
            hit = ~np.isnan(yy) & (yy >= tt - tol) & (yy <= tt + sc.extent_m + tol)
            clutter = ~np.isnan(yy) & (yy < tt - tol)
            # warning distance: truth range when 3 consecutive hits first occur (0 if never)
            act = ~np.isnan(truth)
            h_all = np.array([bool(act[j]) and any((truth[j] - tol) <= x <= (truth[j] + sc.extent_m + tol) for x in alls[n][j])
                              for j in range(len(truth))])
            warn = 0.0
            for j in range(2, len(h_all)):
                if h_all[j] and h_all[j - 1] and h_all[j - 2]:
                    warn = float(truth[j - 2])
                    break
            idx = np.flatnonzero(m)
            ext = sc.extent_m
            on = lambda x, tr: (tr - tol) <= x <= (tr + ext + tol)
            anyh = np.array([any(on(x, truth[j]) for x in alls[n][j]) for j in idx])
            extra = np.array([sum(not on(x, truth[j]) for x in alls[n][j]) for j in idx])
            # any-hit: the target is among the confirmed obstacles (independent of real obstacles nearer than it,
            # which are not errors); extra: other confirmed obstacles per frame (real scene or false)
            hh = np.array(hts[n])[m]
            herr = hh[~np.isnan(hh)] - sc.height_m
            out.append({"run": rec.path.parent.name, "detector": n,
                        "emulation": emulation.name() if emulation is not None else "orig", **{k: v for k, v in asdict(sc).items()},
                        "height_err_med": float(np.median(herr)) if herr.size else float("nan"),
                        "height_err_mad": float(np.median(np.abs(herr - np.median(herr)))) if herr.size else float("nan"),
                        "any_hit": float(anyh.mean()), "extra_per_frame": float(extra.mean()),
                        "hit": float(hit.mean()), "clutter": float(clutter.mean()),
                        "miss": float(1 - hit.mean() - clutter.mean()), "warn_m": warn, "frames": int(m.sum())})
    return out


def default_scenarios(duration_s: float, n: int = 6, seed: int = 0, snr1m_db=(25.0, 18.0)) -> list[Scenario]:
    """Spread approaches over the recording: heights 1.2-1.9 m (head/chest), speeds 0.8-1.3 m/s, some lateral."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        speed = float(rng.uniform(0.8, 1.3))
        span = (2.6 - 0.4) / speed + 2.0
        start = float(rng.uniform(1.5, max(1.6, duration_s - span)))
        out.append(Scenario(start_s=round(start, 2), speed=round(speed, 2),
                            height_m=float(rng.choice([1.2, 1.5, 1.7, 1.9])),
                            lateral_m=float(rng.choice([0.0, 0.0, 0.15, -0.15])),
                            snr1m_db=float(snr1m_db[k % len(snr1m_db)]), label=f"s{k}"))
    return out


def scenario_mix(duration_s: float, n: int = 12, seed: int = 0) -> list[Scenario]:
    """EXP-009 mix: point approaches (18-30 dB@1m), SMALL obstacles near (10-16 dB@1m, 0.5-1.4 m, slow),
    EXTENDED close obstacles (desk-like: 12 scatterers over 0.4 m, +-20 deg, strong) and STATIC obstacles
    held 3 s while the user stands still."""
    rng = np.random.default_rng(seed)
    out = []
    kinds = ["point", "small_near", "extended", "static"]
    for k in range(n):
        kind = kinds[k % 4]
        if kind == "point":
            sc = Scenario(speed=float(rng.uniform(0.8, 1.3)), height_m=float(rng.choice([1.2, 1.5, 1.7, 1.9])),
                          lateral_m=float(rng.choice([0.0, 0.15, -0.15])), snr1m_db=float(rng.choice([18.0, 24.0, 30.0])))
        elif kind == "small_near":
            sc = Scenario(r0=1.5, r_end=0.4, speed=float(rng.uniform(0.3, 0.7)), height_m=float(rng.choice([1.3, 1.6, 1.8])),
                          snr1m_db=float(rng.choice([10.0, 13.0, 16.0])))
        elif kind == "extended":
            sc = Scenario(r0=1.6, r_end=0.4, speed=float(rng.uniform(0.4, 0.9)), height_m=0.9, extent_m=0.4, n_scat=12,
                          spread_deg=20.0, snr1m_db=float(rng.choice([32.0, 38.0])))
        else:
            sc = Scenario(r0=float(rng.uniform(0.7, 1.4)), speed=0.0, hold_s=3.0, height_m=float(rng.choice([1.0, 1.5])),
                          snr1m_db=float(rng.choice([22.0, 30.0])), extent_m=0.2, n_scat=4, spread_deg=8.0)
        span = (sc.r0 - sc.r_end) / sc.speed + 2.0 if sc.speed > 0 else sc.hold_s + 2.0
        sc.start_s = round(float(rng.uniform(1.5, max(1.6, duration_s - span))), 2)
        sc.kind, sc.label = kind, f"{kind}{k}"
        out.append(sc)
    return out


def summarise(rows: list[dict]) -> str:
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["detector"]].append(r)
    L = ["| detector | scenarios | target found % (any) | extra obstacles / frame | nearest = target % | nearer reported % | miss % | warning dist median (m) | warn >= 1.2 m |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, rs in sorted(agg.items(), key=lambda kv: -np.mean([x["any_hit"] for x in kv[1]])):
        L.append(f"| {n} | {len(rs)} | {100*np.mean([x['any_hit'] for x in rs]):.0f} | {np.mean([x['extra_per_frame'] for x in rs]):.2f} | "
                 f"{100*np.mean([x['hit'] for x in rs]):.0f} | {100*np.mean([x['clutter'] for x in rs]):.0f} | "
                 f"{100*np.mean([x['miss'] for x in rs]):.0f} | {np.median([x['warn_m'] for x in rs]):.2f} | "
                 f"{sum(x['warn_m'] >= 1.2 for x in rs)}/{len(rs)} |")
    return "\n".join(L)


def summarise_by_kind(rows: list[dict]) -> str:
    import collections
    kinds = sorted({r.get("kind", "point") for r in rows})
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["detector"], r.get("kind", "point"))].append(r)
    dets = sorted({r["detector"] for r in rows})
    L = ["| detector | " + " | ".join(f"{k}: found % / warn m" for k in kinds) + " | extra/frame |",
         "|---|" + "---|" * (len(kinds) + 1)]
    for d in sorted(dets, key=lambda d: -np.mean([r["any_hit"] for r in rows if r["detector"] == d])):
        cells = []
        for k in kinds:
            rs = agg.get((d, k), [])
            cells.append(f"{100*np.mean([x['any_hit'] for x in rs]):.0f} / {np.median([x['warn_m'] for x in rs]):.2f}" if rs else "-")
        ex = np.mean([r["extra_per_frame"] for r in rows if r["detector"] == d])
        L.append(f"| {d} | " + " | ".join(cells) + f" | {ex:.2f} |")
    return "\n".join(L)


def wall_relative(rec, scenarios: list[Scenario], ref_db: float = 48.0, r_lo: float = 1.4, r_hi: float = 2.1
                  ) -> list[Scenario]:
    """EXP-011: rescale scenario SNRs so targets are a fixed number of dB below the recording's strongest
    return at r_lo..r_hi (the owner's wall, p90 over the run): snr1m_db -> S_wall - (ref_db - snr1m_db).
    Makes injected targets physically comparable across chip configs, whose noise estimates and FFT scaling
    differ (a 7.7 m 1D config hand-held has no empty bins, so its noise estimate is ~10 dB high)."""
    import dataclasses
    bin_m = rec.cfg.derive().bin_mm / 1000.0
    s2 = estimate_sigma2(rec)
    amps = []
    for i in range(0, rec.n, max(1, rec.n // 300)):
        c = rec.get_cube(i)
        r = np.arange(c.shape[-1]) * bin_m
        m = (r > r_lo) & (r < r_hi)
        if m.any():
            amps.append(np.abs(c[:, min(1, c.shape[1] - 1), m]).max())
    if not amps or s2 <= 0:
        return scenarios
    r_ref = 0.5 * (r_lo + r_hi)
    S = 20 * math.log10(float(np.percentile(amps, 90)) * r_ref ** 2 / math.sqrt(s2))
    return [dataclasses.replace(sc, snr1m_db=S - (ref_db - sc.snr1m_db)) for sc in scenarios]


def summarise_by_emulation(rows: list[dict]) -> str:
    """EXP-013 table: per detector and emulated config, found % by scenario kind + the tune score."""
    kinds = ["point", "small_near", "extended", "static"]
    L = ["| detector | config | n | found % point / small near / extended / static | nearest = target | "
         "something nearer | extra/frame | score |", "|---|---|---|---|---|---|---|---|"]
    dets = list(dict.fromkeys(r["detector"] for r in rows))
    emus = list(dict.fromkeys(r.get("emulation", "orig") for r in rows))
    for d in dets:
        for e in emus:
            v = [r for r in rows if r["detector"] == d and r.get("emulation", "orig") == e]
            if not v:
                continue
            k = [100 * np.mean([r["any_hit"] for r in v if r["kind"] == kk]) if any(r["kind"] == kk for r in v)
                 else float("nan") for kk in kinds]
            sc = np.mean([r["any_hit"] + 0.3 * min(r["warn_m"], 1.8) / 1.8 - 0.15 * r["extra_per_frame"]
                          - 0.25 * r["clutter"] for r in v])
            L.append(f"| {d} | {e} | {len(v)} | {' / '.join(f'{x:.0f}' for x in k)} | "
                     f"{100 * np.mean([r['hit'] for r in v]):.0f} % | {100 * np.mean([r['clutter'] for r in v]):.0f} % | "
                     f"{np.mean([r['extra_per_frame'] for r in v]):.1f} | {sc:+.2f} |")
    return "\n".join(L)
