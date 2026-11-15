"""F07: mean power vs range, empty (S0) and furnished (S1) scene of static_pass1, temporal noise level. JSON only."""
from __future__ import annotations

import numpy as np

from .. import common as C, geometry as G, radar as R
from . import common as P


def _load(stem):
    rec = [r for r in C.recordings("static_pass1") if r.file.stem == stem][0]
    rd = C.load_radar(rec)                                   # skips the lead-in
    x = rd.cube[:, 0]                                        # (n, 3 receivers, B bins)
    r = rd.range_m - R.range_bias()
    p = (np.abs(x) ** 2).mean(0).mean(0)                     # mean over time and receivers
    var = (np.abs(x - x.mean(0)) ** 2).mean(0).mean(0)       # variance of the complex samples over time
    frame_med = np.median((np.abs(x) ** 2).mean(1)[:, (r >= G.R_MIN) & (r <= G.R_MAX)], 1)
    return r, 10 * np.log10(p), 10 * np.log10(var), 10 * np.log10(frame_med).mean(), len(x), float(rd.t[-1] - rd.t[0])


def run():
    r, empty, nz_e, med_e, n_e, d_e = _load("01_S0_tcc-1d-40mm")
    r2, furn, noise, med_f, n_f, d_f = _load("04_S1_tcc-1d-40mm")
    assert np.allclose(r, r2)
    w = (r >= 1.0) & (r <= 1.7)
    j = np.where(w)[0][np.argmax(furn[w])]
    w2 = (r >= 2.0) & (r <= G.R_MAX)
    k = np.where(w2)[0][np.argmax(furn[w2])]
    use = (r >= G.R_MIN) & (r <= G.R_MAX)
    P.save_json("f07_perfil", {
        "range_m": r, "empty_db": empty, "furnished_db": furn, "noise_db": noise, "noise_empty_db": nz_e,
        "frame_median_db": med_f, "frame_median_empty_db": med_e, "r_min": G.R_MIN, "r_max": G.R_MAX,
        "chair": {"r": r[j], "db": furn[j], "empty_db": empty[j], "above_empty_db": furn[j] - empty[j]},
        "wall": {"r": r[k], "db": furn[k], "empty_db": empty[k]},
        "clutter_over_noise_db": float(np.median(furn[use] - noise[use])),
        "frames": {"empty": n_e, "furnished": n_f}, "seconds": {"empty": d_e, "furnished": d_f}})


if __name__ == "__main__":
    run()
