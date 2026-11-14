"""GT3 + A4: radar-camera extrinsic and radar calibration from the static takes.

Every non-floor AprilTag of the static scenes (median pose over the recording) is a target: the strongest radar
return within +-0.25 m of the tag's range (empty-scene reference removed when the take has one) gives the
measured range and the interferometric phases. Fits:
  * range: r_meas = |p_tag - t_rc| + b  (robust least squares, prior |t_rc| ~ 5 cm) -> t_rc, range bias b;
  * angles: per antenna pair, sign and phase offset (circular mean of sign*phase - expected phase); the mean
    resultant length R (1 = perfectly consistent) is reported, so is the residual per target.
Tables: calib_targets (one row per recording x tag) and extrinsic.json / angle_calib.json."""

from __future__ import annotations

import numpy as np

from . import common as C
from . import geometry as G
from . import radar as R


def target_rows(tags, takes=None, window_m=0.25):
    rows = []
    for take in C.available_takes(takes or C.STATIC):
        if take not in C.STATIC:
            continue
        for rec in C.recordings(take):
            if rec.part == "S0":
                continue
            t = C.radar_times(rec, C.default_max_frames())
            q = tags[(tags["take"] == take) & (tags.t_host >= t[0]) & (tags.t_host <= t[-1]) & (tags.reproj_px < 2)
                     & (tags.view_deg < 70) & tags.kind.isin(["chair", "broom"])]   # still objects only
            if not len(q):
                continue
            F = R.features(rec, {"bg": "empty"}, C.default_max_frames())
            F.r = F.r + R.range_bias()                    # the fit needs the uncorrected range axis
            P = F.P.mean(0)
            noise = np.median(F.noise)
            for i, g in q.groupby("id"):
                if len(g) < 5 or g[["tx", "ty", "tz"]].std().max() > 0.03:       # moved during the recording
                    continue
                p = g[["tx", "ty", "tz"]].median().to_numpy()
                r, az, el = G.polar(p)
                w = np.flatnonzero(np.abs(F.r - r) <= window_m)
                if not len(w):
                    continue
                b = int(w[np.argmax(P[w])])
                y = 10 * np.log10(np.maximum(P[[max(b - 1, 0), b, min(b + 1, len(P) - 1)]], 1e-300))
                den = y[0] - 2 * y[1] + y[2]
                rb = F.r[b] + (0.5 * (y[0] - y[2]) / den if abs(den) > 1e-9 else 0.0) * (F.r[1] - F.r[0])
                rows.append(dict(take=take, rec=rec.key, config=rec.config, part=rec.part, id=int(i),
                                 kind=C.TAG_KIND.get(int(i), "other"), x=p[0], y=p[1], z=p[2], r_true=r, az_true=az,
                                 el_true=el, n_tag=len(g), r_meas=rb, bin=b, p_db=10 * np.log10(P[b]),
                                 snr_db=10 * np.log10(P[b] / noise), ph_az=float(np.angle(F.S_az[:, b].sum())),
                                 ph_el=float(np.angle(F.S_el[:, b].sum())),
                                 coh_az=float(np.abs(F.S_az[:, b].sum()) / np.abs(F.S_az[:, b]).sum())))
    import pandas as pd
    return pd.DataFrame(rows)


def fit_range(T, prior_m=0.05, free_offset=False):
    """Range bias b (robust median) with the radar at the camera (t_rc = 0: the chip is 4 cm from the lens, a
    <= 4 cm effect on range). free_offset=True also fits t_rc (badly conditioned with targets that are all ahead:
    a first full run moved the radar 23 cm up to absorb range errors)."""
    from scipy.optimize import least_squares
    P = T[["x", "y", "z"]].to_numpy()
    rm = T.r_meas.to_numpy()
    if len(T) < 5 or not free_offset:
        b = float(np.median(rm - np.linalg.norm(P, axis=1)))
        return np.zeros(3), b, rm - np.linalg.norm(P, axis=1) - b

    def res(v):
        t, b = v[:3], v[3]
        return np.r_[np.linalg.norm(P - t, axis=1) + b - rm, 0.04 / prior_m * t]
    sol = least_squares(res, np.zeros(4), loss="soft_l1", f_scale=0.05)
    t, b = sol.x[:3], sol.x[3]
    return t, float(b), np.linalg.norm(P - t, axis=1) + b - rm


def fit_angle(ph, ang_true_deg):
    expd = 2 * np.pi * R.D_OVER_LAMBDA * np.sin(np.radians(ang_true_deg))
    best = None
    for sg in (1.0, -1.0):
        z = np.exp(1j * (sg * ph - expd))
        m = z.mean()
        if best is None or abs(m) > best[2]:
            best = (sg, float(np.angle(m)), float(abs(m)))
    sg, off, Rl = best
    pred = R.phase_to_angle(ph, sg, off)
    return {"sign": sg, "offset_rad": off, "R": Rl, "resid_deg": (pred - ang_true_deg).tolist()}


def run(takes=None):
    tags = C.load_table("tags")
    T = target_rows(tags, takes)
    if not len(T):
        C.log("no tag targets in the static takes (run the tags step on them first)")
        return None
    t, b, res = fit_range(T)
    T["range_resid_m"] = res
    a = fit_angle(T.ph_az.to_numpy(), T.az_true.to_numpy())
    e = fit_angle(T.ph_el.to_numpy(), T.el_true.to_numpy())
    T["az_resid_deg"] = a.pop("resid_deg")
    T["el_resid_deg"] = e.pop("resid_deg")
    C.save_table(T, "calib_targets", takes=C.available_takes(takes or C.STATIC))
    ext = {"t_rc": [float(v) for v in t], "range_bias_m": b, "n_targets": int(len(T)),
           "range_resid_rms_m": float(np.sqrt(np.mean(res ** 2))),
           "note": "radar origin in camera coordinates (x right, y down, z forward); r_meas = |p - t_rc| + b"}
    C.save_json(ext, "extrinsic")
    C.save_json({"az": a, "el": e, "note": "angle = arcsin(wrap(sign*phase - offset) / (2 pi d/lambda))"},
                "angle_calib")
    R.angle_calib.cache_clear()
    R.range_bias.cache_clear()
    R._FCACHE.clear()
    C.log(f"  t_rc = {np.round(t, 3)} m, range bias {b:+.3f} m, residual rms {ext['range_resid_rms_m']:.3f} m "
          f"({len(T)} targets)")
    C.log(f"  azimuth: sign {a['sign']:+.0f} offset {a['offset_rad']:+.2f} rad, R {a['R']:.2f}, "
          f"|resid| median {np.median(np.abs(T.az_resid_deg)):.1f} deg")
    C.log(f"  elevation: sign {e['sign']:+.0f} offset {e['offset_rad']:+.2f} rad, R {e['R']:.2f}, "
          f"|resid| median {np.median(np.abs(T.el_resid_deg)):.1f} deg")
    return T
