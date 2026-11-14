"""Where the early alarms of the carried takes come from (step `floor`, after eval-tuned).

Three tables, all from the existing eval tables and the truth's local maps (no detector is rerun):

  floor_margin   outcome by "floor margin" = range at which the floor enters the beam (truth arc_floor_r) minus
                 the true nearest obstacle range. Negative margin: the floor is in the beam before the obstacle.
  floor_pitch    outcome by true obstacle distance x radar pitch (phone gravity), per environment.
  phantom        frames with nothing within 2 m that still raise an alarm: the alarm's range compared with
                 what the camera map has there (floor in the beam, obstacles at any elevation, floor at any
                 angle), per environment; plus the still radar of the static empty scenes as a control.
"""

from __future__ import annotations

import numpy as np

from . import common as C

MARGIN_BINS = [-10, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 10]
PITCH_BINS = [-90, -25, -15, -5, 90]
DIST_BINS = [0.3, 0.75, 1.25, 2.0, 8.0]
VARIANTS = {"default": "", "tuned arc": "_tuned_pooled_arc"}
FAMS = ["ca_cfar", "thr"]


def _outcome(at, ad):
    found = np.isfinite(at) & np.isfinite(ad) & (np.abs(ad - at) <= 0.25)
    early = np.isfinite(ad) & (~np.isfinite(at) | (ad < at - 0.25))
    return found, early


def _frames(variant, fams):
    import pandas as pd
    Fr = C.load_table("eval_frames" + variant, required=False)
    if Fr is None:
        return None
    Fr = Fr[Fr["take"].isin(C.CARRIED) & Fr.arc_covered.astype(bool) & Fr.family.isin(fams)]
    TF = C.load_table("truth_frames").set_index(["rec", "f"])
    idx = list(zip(Fr.rec, Fr.f))
    D = pd.DataFrame(dict(rec=Fr.rec.to_numpy(), f=Fr.f.to_numpy(), take=Fr["take"].to_numpy(),
                          env=Fr.env.to_numpy(), family=Fr.family.to_numpy(),
                          at=Fr.arc_truth.to_numpy(), ad=Fr.arc_det.to_numpy(),
                          floor_r=TF.loc[idx, "arc_floor_r"].to_numpy(), t=TF.loc[idx, "t"].to_numpy(),
                          radar_h=-TF.loc[idx, "floor_h"].to_numpy()))
    D["found"], D["early"] = _outcome(D["at"].to_numpy(), D.ad.to_numpy())
    return D


def _pitch(D):
    from . import imu
    from .geometry import up_from_gravity
    p = np.full(len(D), np.nan)
    for take in D["take"].unique():
        m = (D["take"] == take).to_numpy()
        g = imu.at(take, D.t.to_numpy()[m], "Gravity", 0.3)
        if g is not None:
            p[m] = np.degrees(np.arcsin(np.clip(up_from_gravity(g)[:, 2], -1, 1)))
    return p


def _agg(D, keys):
    return (D.groupby(keys, observed=True)
            .agg(frames=("found", "size"), found=("found", "mean"), early=("early", "mean")).reset_index())


def margin_table():
    import pandas as pd
    out = []
    for lab, v in VARIANTS.items():
        D = _frames(v, FAMS)
        if D is None:
            continue
        D = D[D["at"] > 0.75]                                  # outside the near blind zone
        D["margin"] = pd.cut(D.floor_r - D["at"], MARGIN_BINS).astype(str)
        out.append(_agg(D, ["family", "env", "margin"]).assign(variant=lab))
    T = pd.concat(out, ignore_index=True)
    C.save_table(T, "floor_margin")
    return T


def pitch_table():
    import pandas as pd
    D = _frames(VARIANTS["tuned arc"], ["ca_cfar"])
    if D is None:
        return None
    D["pitch"] = _pitch(D)
    D["dist"] = pd.cut(D["at"], DIST_BINS).astype(str)
    D["pitch_bin"] = pd.cut(D.pitch, PITCH_BINS).astype(str)
    T = _agg(D, ["env", "dist", "pitch_bin"])
    C.save_table(T, "floor_pitch")
    return T


def _phantom_take(take, F, n_max):
    """Per sampled alarm frame: what the camera's local map has around the alarm's range."""
    from . import geometry as G, truth as T
    from .maps import TakeMaps, interp_poses
    tm = TakeMaps(take, "pi3x")
    trc = T.t_rc()
    clouds, rows = {}, []
    F = F.iloc[np.linspace(0, len(F) - 1, min(n_max, len(F))).astype(int)] if len(F) else F
    t_kf = tm.t[tm.kf]
    for rec, f, tt, ad, h in C.bar(list(zip(F.rec, F.f, F.t, F.ad, F.radar_h)), desc=f"phantom {take}", unit="frame"):
        i = int(tm.kf[int(np.argmin(np.abs(t_kf - tt)))])
        P, tk, hh, s_w, t_w, c2w = T.window_cloud(tm, take, tm.owner[i][0], clouds)
        if not (t_w.min() - 0.05 <= tt <= t_w.max() + 0.05):
            continue
        Tw = interp_poses(t_w, c2w, np.array([tt]))[0]
        sel = np.abs(tk - tt) <= T.ARC_WINDOW_S
        Q = G.to_radar((P[sel] - Tw[:3, 3]) @ Tw[:3, :3] * s_w, trc)
        r, az, el = G.polar(Q)
        hs = hh[sel]
        obs = (hs > T.H_OBS_MIN) & (r >= G.R_MIN)
        flo = np.abs(hs) < T.H_FLOOR_TOL
        a, e = np.abs(az), np.abs(el)
        pick = lambda m: float(r[m].min()) if m.any() else np.nan
        rows.append(dict(take=take, rec=rec, f=f, alarm_r=ad, radar_h=h,
                         floor_beam=pick(flo & (a <= T.ARC_HALF_DEG) & (e <= T.ARC_EL_HALF_DEG)),
                         floor_any=pick(flo),
                         obstacle_any_el=pick(obs & (a <= T.ARC_HALF_DEG)),
                         obstacle_side=pick(obs & (a > T.ARC_HALF_DEG) & (a <= 90))))
    return rows


def phantom_table(n_max=200):
    import pandas as pd
    D = _frames(VARIANTS["tuned arc"], ["ca_cfar"])
    if D is None:
        return None
    D = D[(D["at"] > 2.0) & np.isfinite(D.ad)]
    rows = []
    for take in C.CARRIED:
        rows += _phantom_take(take, D[D["take"] == take], n_max)
    P = pd.DataFrame(rows)
    C.save_table(P, "phantom_frames")
    env = {t: ("indoor" if t.startswith("indoor") else "outdoor") for t in C.CARRIED}
    S = []
    for take, g in P.groupby("take"):
        d = g.alarm_r
        S.append(dict(case=f"{take} (carried, tuned CA-CFAR)", env=env[take], frames=len(g),
                      alarm_median=d.median(), alarm_p25=d.quantile(.25), alarm_p75=d.quantile(.75),
                      radar_height=g.radar_h.median(), floor_in_beam=g.floor_beam.median(),
                      alarm_at_floor_in_beam=float((np.abs(d - g.floor_beam) <= 0.25).mean()),
                      nearest_floor_any_angle=g.floor_any.median(),
                      obstacle_any_elevation=g.obstacle_any_el.median(),
                      obstacle_at_60_90deg=g.obstacle_side.median()))
    Fs = C.load_table("eval_frames", required=False)               # control: the still radar, empty scene
    if Fs is not None:
        s = Fs[Fs["take"].isin(C.STATIC) & (Fs.part == "S0") & (Fs.family == "ca_cfar")]
        for take, g in s.groupby("take"):
            S.append(dict(case=f"{take} S0 (still radar, empty scene, default CA-CFAR)", env="indoor",
                          frames=len(g), alarm_median=g.arc_det.median(), alarm_p25=g.arc_det.quantile(.25),
                          alarm_p75=g.arc_det.quantile(.75)))
    S = pd.DataFrame(S)
    C.save_table(S, "phantom")
    return S


def run(n_max=200):
    C.log("  floor margin"); margin_table()
    C.log("  pitch"); pitch_table()
    C.log("  phantom alarms"); S = phantom_table(n_max)
    if S is not None:
        C.log("\n" + S.round(2).to_string())
