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


def moving_rows(tags, takes=None, window_bins=3, min_snr_db=10.0, max_dt=0.06):
    """Calibration targets from the tagged person walking in front of the still radar (static takes).

    The recording's complex mean is removed (the still scene), so the person is the only return in their range bin:
    one clean scatterer per tag sighting, over the whole field of view, instead of a handful of still objects whose
    range bins also hold walls and furniture. Per sighting: the strongest moving bin within +-window_bins of the
    tag's range (kept when >= min_snr_db over the frame's median moving power) gives the measured range and the
    two phase differences (same convention as radar.features: x_b conj(x_a), summed over 3 bins). The whole
    recording is used, also in quick mode."""
    import pandas as pd
    rows = []
    person = [i for i, k in C.TAG_KIND.items() if k == "person"]
    for take in C.available_takes(takes or C.STATIC):
        if take not in C.STATIC:
            continue
        for rec in C.recordings(take):
            if rec.part == "S0":
                continue
            rd = C.load_radar(rec)
            q = tags[(tags["take"] == take) & tags.id.isin(person) & (tags.t_host >= rd.t[0]) & (tags.t_host <= rd.t[-1])
                     & (tags.reproj_px < 2) & (tags.view_deg < 70)]
            if not len(q) or rd.n < 20:
                continue
            x = rd.cube.astype(np.complex128)
            x = x - x.mean(0, keepdims=True)
            Pm = (np.abs(x) ** 2).sum(1).mean(1)                           # (n, B) moving power
            B = Pm.shape[1]
            dr = float(rd.range_m[1] - rd.range_m[0])
            for g in q.itertuples():
                p = np.array([g.tx, g.ty, g.tz])
                r, az, el = G.polar(p)
                f = int(np.argmin(np.abs(rd.t - g.t_host)))
                b0 = int(round((r - rd.range_m[0]) / dr))
                if abs(rd.t[f] - g.t_host) > max_dt or b0 - window_bins < 1 or b0 + window_bins > B - 2:
                    continue
                w = np.arange(b0 - window_bins, b0 + window_bins + 1)
                b = int(w[np.argmax(Pm[f, w])])
                snr = 10 * np.log10(Pm[f, b] / max(np.median(Pm[f]), 1e-300))
                if snr < min_snr_db:
                    continue
                seg = x[f][:, :, b - 1:b + 2]                               # (D, 3, 3 bins)
                a_, b_ = R.PAIRS["az"]
                s_az = (seg[:, b_] * np.conj(seg[:, a_])).sum()
                a_, b_ = R.PAIRS["el"]
                s_el = (seg[:, b_] * np.conj(seg[:, a_])).sum()
                rows.append(dict(take=take, rec=rec.key, config=rec.config, part=rec.part, id=int(g.id), kind="person",
                                 x=p[0], y=p[1], z=p[2], r_true=float(r), az_true=float(az), el_true=float(el),
                                 r_meas=float(rd.range_m[b]), bin=b, snr_db=float(snr),
                                 ph_az=float(np.angle(s_az)), ph_el=float(np.angle(s_el))))
    return pd.DataFrame(rows)


MIN_MOVING = 100             # sightings of the walking person needed to calibrate on them


def fit_range(T):
    """Range bias b (median of measured - true range) with the radar at the camera (t_rc = 0: the chip is 4 cm from
    the lens; a free offset is badly conditioned with targets that are all ahead)."""
    r_true = np.linalg.norm(T[["x", "y", "z"]].to_numpy(), axis=1)
    b = float(np.median(T.r_meas.to_numpy() - r_true))
    return np.zeros(3), b, T.r_meas.to_numpy() - r_true - b


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
    tags = C.load_tags()
    T = target_rows(tags, takes)
    if not len(T):
        C.log("no tag targets in the static takes (run the tags step on them first)")
        return None
    M = moving_rows(tags, takes)
    use_moving = len(M) >= MIN_MOVING
    Fit = M if use_moving else T                # calibrate on the walking person; the still objects are the check
    method = "moving person (mean-removed cube)" if use_moving else "still tagged objects"
    if not use_moving:
        C.log(f"  calibration on the still objects ({len(M)} sightings of the walking person, {MIN_MOVING} needed):"
              " unreliable (bins shared with walls and furniture), treat the angles as uncalibrated")
    t, b, _ = fit_range(Fit)
    a = fit_angle(Fit.ph_az.to_numpy(), Fit.az_true.to_numpy())
    e = fit_angle(Fit.ph_el.to_numpy(), Fit.el_true.to_numpy())
    a.pop("resid_deg"); e.pop("resid_deg")
    for D_ in (T, M):                           # residuals of both sets against the adopted calibration
        if len(D_):
            D_["range_resid_m"] = D_.r_meas - np.linalg.norm(D_[["x", "y", "z"]].to_numpy() - t, axis=1) - b
            D_["az_resid_deg"] = R.phase_to_angle(D_.ph_az.to_numpy(), a["sign"], a["offset_rad"]) - D_.az_true
            D_["el_resid_deg"] = R.phase_to_angle(D_.ph_el.to_numpy(), e["sign"], e["offset_rad"]) - D_.el_true
    res = Fit.range_resid_m.to_numpy()
    C.save_table(T, "calib_targets", takes=C.available_takes(takes or C.STATIC))
    if len(M):
        C.save_table(M, "calib_moving", takes=C.available_takes(takes or C.STATIC))
    ext = {"t_rc": [float(v) for v in t], "range_bias_m": b, "n_targets": int(len(Fit)), "method": method,
           "range_resid_rms_m": float(np.sqrt(np.mean(res ** 2))),
           "range_resid_mad_m": float(np.median(np.abs(res - np.median(res)))),
           "note": "radar origin in camera coordinates (x right, y down, z forward); r_meas = |p - t_rc| + b"}
    C.save_json(ext, "extrinsic")
    C.save_json({"az": a, "el": e, "method": method,
                 "note": "angle = arcsin(wrap(sign*phase - offset) / (2 pi d/lambda)); the elevation targets span "
                         "only a few degrees and a standing person is tall: the elevation offset is approximate"},
                "angle_calib")
    R.angle_calib.cache_clear()
    R.range_bias.cache_clear()
    R._FCACHE.clear()
    C.log(f"  t_rc = {np.round(t, 3)} m, range bias {b:+.3f} m, residual rms {ext['range_resid_rms_m']:.3f} m "
          f"({len(Fit)} targets, {method})")
    C.log(f"  azimuth: sign {a['sign']:+.0f} offset {a['offset_rad']:+.2f} rad, R {a['R']:.2f}, "
          f"|resid| median {np.median(np.abs(Fit.az_resid_deg)):.1f} deg")
    C.log(f"  elevation: sign {e['sign']:+.0f} offset {e['offset_rad']:+.2f} rad, R {e['R']:.2f}, "
          f"|resid| median {np.median(np.abs(Fit.el_resid_deg)):.1f} deg")
    return T
