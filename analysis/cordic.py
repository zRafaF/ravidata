"""E1: CORDIC in fixed point (bit-true numpy model of the STM32 CORDIC's algorithm) against float64.

Tables:
  cordic_error  sin/cos (rotation mode) and atan2 / magnitude (vectoring mode) errors over a dense grid, per
                word length (q1.15, q1.31) and number of iterations;
  cordic_phase  interferometric phase of real radar returns (x_b * conj(x_a) at the detections): CORDIC
                vectoring on the products (raw scale and normalised to full scale) and float32 atan(imag/real),
                each against float64 np.angle -- the 'tiny numbers' effect, measured;
  cordic_grid   the occupancy-grid pipeline with float64 trig vs CORDIC trig: final metrics and how often the
                obstacle output differs.
cordic_reference.json carries published cycle counts (ST AN5325) so the plots can put the error next to the cost."""

from __future__ import annotations

import numpy as np

from . import common as C

REFERENCE = {
    "source": "ST AN5325 'Getting started with the CORDIC accelerator using STM32CubeG4 MCU Package', "
              "STM32G4 at 170 MHz (cycles per sin/cos pair)",
    "cycles": {"CORDIC q1.31 (zero-overhead, 6 cycles + wait)": 29, "CORDIC q1.31, results kept local": 17,
               "CORDIC pipelined, per value": 8, "CORDIC with DMA, per value": 15.5,
               "arm_sin_cos_f32 (CMSIS-DSP)": 405, "sinf + cosf (libm)": 416, "arm_sin_cos_q31 (CMSIS-DSP)": 742},
    "q31_error_24_iterations": 1.9e-6,
    "note": "The STM32H563 used in this work has the same CORDIC peripheral; its own cycle counts are not "
            "measured here.",
}


def _atan_table(n):
    return np.arctan(2.0 ** -np.arange(n))


def _gain(n):
    return np.prod(1 / np.sqrt(1 + 2.0 ** (-2 * np.arange(n))))


def sincos(theta, n_iter=24, frac=31):
    """CORDIC rotation mode in integer arithmetic. theta (rad, any) -> (sin, cos) as floats (quantised)."""
    th = np.angle(np.exp(1j * np.asarray(theta, float)))               # wrap to [-pi, pi]
    flip = np.abs(th) > np.pi / 2                                       # reduce to [-pi/2, pi/2]
    th = np.where(flip, th - np.sign(th) * np.pi, th)
    one = 1 << frac
    sc = 1 << (frac - 2)                                                # angles in units of pi / 4 * 2^-... (q2.frac)
    z = np.round(th / np.pi * 4 * sc).astype(np.int64)
    at = np.round(_atan_table(n_iter) / np.pi * 4 * sc).astype(np.int64)
    x = np.full(z.shape, int(round(_gain(n_iter) * (one - 1))), np.int64)
    y = np.zeros(z.shape, np.int64)
    for i in range(n_iter):
        d = np.where(z >= 0, 1, -1)
        x, y = x - d * (y >> i), y + d * (x >> i)
        z = z - d * at[i]
    s, c = y / one, x / one
    s = np.where(flip, -s, s)
    c = np.where(flip, -c, c)
    return s, c


def atan2_mag(y, x, n_iter=24, frac=31, normalise=True):
    """CORDIC vectoring mode. Inputs are quantised to q1.frac (after scaling to full scale if normalise)."""
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    scale = np.maximum(np.abs(x), np.abs(y)).max() if normalise else 1.0
    scale = scale if scale > 0 else 1.0
    one = 1 << frac
    xi = np.round(x / scale * (one - 1) * 0.5).astype(np.int64)        # keep 1 guard bit (gain ~1.65)
    yi = np.round(y / scale * (one - 1) * 0.5).astype(np.int64)
    flip = xi < 0                                                       # rotate by pi into the right half plane
    xi, yi = np.where(flip, -xi, xi), np.where(flip, -yi, yi)
    sc = 1 << (frac - 2)
    at = np.round(_atan_table(n_iter) / np.pi * 4 * sc).astype(np.int64)
    z = np.zeros(xi.shape, np.int64)
    for i in range(n_iter):
        d = np.where(yi < 0, 1, -1)
        xi, yi = xi - d * (yi >> i), yi + d * (xi >> i)
        z = z - d * at[i]
    ang = z / (4 * sc) * np.pi
    ang = np.where(flip, np.where(ang <= 0, ang + np.pi, ang - np.pi), ang)
    ang = np.where((np.abs(x) == 0) & (np.abs(y) == 0), 0.0, ang)
    mag = xi * _gain(n_iter) / ((one - 1) * 0.5) * scale
    return ang, mag


def trig_fn(n_iter, frac):
    return lambda a: sincos(a, n_iter, frac)


def error_rows(iters=(4, 6, 8, 10, 12, 14, 16, 20, 24), fracs=(15, 31)):
    th = np.linspace(-np.pi, np.pi, 20001)
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi, np.pi, 20000)
    v = np.stack([np.cos(phi), np.sin(phi)], 1)                         # unit vectors: the algorithm's own error
    rows = []
    for q in fracs:
        for n in iters:
            if q == 15 and n > 16:
                continue
            s, c = sincos(th, n, q)
            e = np.maximum(np.abs(s - np.sin(th)), np.abs(c - np.cos(th)))
            a, m = atan2_mag(v[:, 1], v[:, 0], n, q)
            ea = np.abs(np.angle(np.exp(1j * (a - np.arctan2(v[:, 1], v[:, 0])))))
            rows.append(dict(op="sincos", q=f"q1.{q}", iterations=n, max_err=float(e.max()),
                             rms_err=float(np.sqrt((e ** 2).mean())), max_err_deg=np.nan))
            rows.append(dict(op="atan2", q=f"q1.{q}", iterations=n, max_err=float(ea.max()),
                             rms_err=float(np.sqrt((ea ** 2).mean())), max_err_deg=float(np.degrees(ea.max()))))
    # dynamic range: a vector 10^-k below the full-scale value that sets the input scaling
    for q in fracs:
        for k in range(0, 9):
            a, _ = atan2_mag(np.r_[1.0, v[:2000, 1] * 10.0 ** -k], np.r_[0.0, v[:2000, 0] * 10.0 ** -k], 24 if q == 31
                             else 16, q)
            ea = np.abs(np.angle(np.exp(1j * (a[1:] - phi[:2000]))))
            rows.append(dict(op="atan2_dynamic", q=f"q1.{q}", iterations=24 if q == 31 else 16, rel_level_db=-20.0 * k,
                             max_err=float(ea.max()), rms_err=float(np.sqrt((ea ** 2).mean())),
                             max_err_deg=float(np.degrees(ea.max()))))
    return rows


def phase_rows(recs, iters=(8, 12, 16, 20, 24)):
    """Interferometric phase of real detections: CORDIC (raw / normalised) and float32 atan vs float64."""
    from . import pipeline as PL
    rows = []
    for rec in recs:
        d, _, F = PL.detect(rec, "ca_cfar", {}, C.default_max_frames())
        if d is None or not len(d["f"]):
            continue
        s = F.S_az[d["f"], d["b"]]
        ref = np.angle(s)                                               # float64
        re32, im32 = s.real.astype(np.float32), s.imag.astype(np.float32)
        naive = np.arctan(im32 / re32).astype(np.float64)              # atan(imag/real), float32
        quad = np.where(re32 < 0, naive + np.pi, naive)
        for name, a in (("float32 atan(im/re), no quadrant", naive), ("float32 atan(im/re) + quadrant", quad),
                        ("float32 arctan2", np.arctan2(im32, re32).astype(float))):
            e = np.degrees(np.abs(np.angle(np.exp(1j * (a - ref)))))
            rows.append(dict(rec=rec.key, method=name, q="", iterations=0, n=len(e), median_err_deg=np.median(e),
                             p99_err_deg=np.percentile(e, 99), max_err_deg=e.max(), magnitude_median=np.median(np.abs(s))))
        for q in (15, 31):
            for n in iters:
                for norm in (False, True):
                    a, _ = atan2_mag(s.imag, s.real, n, q, normalise=norm)
                    e = np.degrees(np.abs(np.angle(np.exp(1j * (a - ref)))))
                    rows.append(dict(rec=rec.key, method="CORDIC " + ("normalised" if norm else "raw scale"),
                                     q=f"q1.{q}", iterations=n, n=len(e), median_err_deg=np.median(e),
                                     p99_err_deg=np.percentile(e, 99), max_err_deg=e.max(),
                                     magnitude_median=np.median(np.abs(s))))
    return rows


def grid_rows(recs, iters=(6, 8, 10, 12, 16, 24), fracs=(15, 31)):
    from .evaluate import eval_rec
    from . import score as SC
    rows = []
    for rec in recs:
        base = eval_rec(rec, "grid", {}, keep_dets=False)
        if base is None:
            continue
        Fb = base[0]
        sb = SC.summary(Fb)
        for q in fracs:
            for n in iters:
                if q == 15 and n > 16:
                    continue
                r = eval_rec(rec, "grid", {}, keep_dets=False, trig=trig_fn(n, q))
                Fr = r[0]
                s = SC.summary(Fr)
                same = (Fr.n_det.to_numpy() == Fb.n_det.to_numpy()) & (Fr.tp50.to_numpy() == Fb.tp50.to_numpy())
                rows.append(dict(rec=rec.key, take=rec.take, config=rec.config, q=f"q1.{q}", iterations=n,
                                 frames=len(Fr), same_output_frac=float(same.mean()),
                                 f1_50=s.get("f1_50"), f1_50_float=sb.get("f1_50"), recall_50=s.get("recall_50"),
                                 recall_50_float=sb.get("recall_50"), gospa=s.get("gospa"), gospa_float=sb.get("gospa")))
    return rows


def run(takes=None, max_recs=None):
    import pandas as pd
    C.save_table(pd.DataFrame(error_rows()), "cordic_error")
    recs = [r for t in C.available_takes(takes) for r in C.recordings(t)]
    truth = C.load_table("truth_frames", required=False)
    if truth is not None:
        recs = [r for r in recs if r.key in set(truth.rec)]
    if max_recs:
        recs = recs[:max_recs]
    C.save_table(pd.DataFrame(phase_rows(recs)), "cordic_phase")
    C.save_table(pd.DataFrame(grid_rows(recs)), "cordic_grid")
    C.save_json(REFERENCE, "cordic_reference")
    E = C.load_table("cordic_error")
    C.log("\n" + E.pivot_table(index="iterations", columns=["op", "q"], values="max_err").to_string())
