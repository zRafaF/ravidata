"""Interferometry calibration check from a marked recording: ``uv run ravidata phase-check RUN`` (D-024).

Protocol (EXP-009, 10 min): a small strong reflector (metal can, foil corner) on a stick at 1.0-1.5 m.
Keep the radar still. In the lab GUI press, in any order, and hold each position ~5 s:
  "phase: boresight"   reflector straight ahead at radar height
  "phase: up"          raised ~30 cm            "phase: down"   lowered ~30 cm
  "phase: left"        moved ~30 cm to the left "phase: right"  moved ~30 cm to the right
(Markers tab, "Phase check" buttons.) The script finds the strongest return at 0.5-2.5 m in each hold
(the dominant range-Doppler cell in 2D), averages the RX-pair phases (power weighted), and reports:
  * the boresight offsets -> phase_offset_deg, az_offset_deg
  * whether up/down moves the RX-hi/RX-ref pair (vertical pair?) and with which sign -> elev_sign
  * whether left/right moves the RX-az/RX-ref pair (horizontal pair?)
  * the angle each hold implies vs the geometry (atan(0.3 / range)).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .sources import open_recording

KEYS = ("boresight", "up", "down", "left", "right")


def _markers(run_dir: Path):
    out = []
    ev = run_dir / "events.jsonl"
    for line in ev.read_text(encoding="utf-8", errors="replace").splitlines() if ev.exists() else []:
        try:
            e = json.loads(line)
        except Exception:
            continue
        txt = str(e.get("text", "")).lower()
        if e.get("type") in ("marker", "note") and txt.startswith("phase:"):
            k = txt.split(":", 1)[1].strip().split()[0]
            if k in KEYS:
                out.append((int(e.get("frame", 0)), k))
    return sorted(out)


def phase_check(path, hi=2, ref=1, az=0, hold_s=4.0) -> str:
    rec = open_recording(path)
    run_dir = Path(rec.path).parent
    mk = _markers(run_dir)
    if not mk:
        return "no 'phase: ...' markers in this run (see the protocol in src/ravidata/phasecheck.py)"
    d = rec.cfg.derive()
    dl, bin_m = d.d_over_lambda, d.bin_mm / 1000.0
    res = {}
    for j, (f0, k) in enumerate(mk):
        f1 = mk[j + 1][0] if j + 1 < len(mk) else rec.n
        t_end = rec.times[min(f0, rec.n - 1)] + hold_s
        idx = [i for i in range(f0 + 3, min(f1, rec.n)) if rec.times[i] < t_end]
        if len(idx) < 5:
            continue
        C = np.stack([rec.get_cube(i) for i in idx]).astype(np.complex128)     # n, dop, rx, R
        P = (np.abs(C[:, :, ref, :]) ** 2).mean(0)                               # dop, R
        r = np.arange(P.shape[1]) * bin_m
        P[:, (r < 0.5) | (r > 2.5)] = 0
        kd, b = np.unravel_index(int(np.argmax(P)), P.shape)
        x_ref, x_hi, x_az = C[:, kd, ref, b], C[:, kd, hi, b], C[:, kd, az, b]
        c_el, c_az = (x_hi * np.conj(x_ref)).sum(), (x_az * np.conj(x_ref)).sum()
        coh = abs(c_el) / max(1e-30, np.abs(x_hi * np.conj(x_ref)).sum())
        res[k] = {"range_m": round(b * bin_m, 2), "ph_el": math.degrees(np.angle(c_el)), "ph_az": math.degrees(np.angle(c_az)),
                  "coh": round(float(coh), 2), "frames": len(idx)}
    L = ["| hold | range m | phase hi-ref (deg) | phase az-ref (deg) | coherence |", "|---|---|---|---|---|"]
    for k in KEYS:
        if k in res:
            v = res[k]
            L.append(f"| {k} | {v['range_m']} | {v['ph_el']:.1f} | {v['ph_az']:.1f} | {v['coh']} |")
    if "boresight" in res:
        b = res["boresight"]
        L += ["", f"**Offsets:** `phase_offset_deg` = {b['ph_el']:.0f}, `az_offset_deg` = {b['ph_az']:.0f}"]

        def rel(k, which):
            return (res[k][which] - b[which] + 180) % 360 - 180
        for k, exp_pair in (("up", "ph_el"), ("down", "ph_el"), ("left", "ph_az"), ("right", "ph_az")):
            if k in res:
                de, da = rel(k, "ph_el"), rel(k, "ph_az")
                ang_geo = math.degrees(math.atan2(0.3, max(0.3, res[k]["range_m"])))
                exp_dphi = 360 * dl * math.sin(math.radians(ang_geo))
                L.append(f"- {k}: hi-ref moved {de:+.0f} deg, az-ref moved {da:+.0f} deg (a 30 cm move at "
                         f"{res[k]['range_m']} m should move the pair in that plane by about ±{exp_dphi:.0f} deg)")
        if "up" in res and "down" in res:
            s = rel("up", "ph_el") - rel("down", "ph_el")
            L.append(f"- **vertical sign:** up − down = {s:+.0f} deg → `elev_sign` = {1 if s > 0 else -1}"
                     + ("  (small: the RX-hi/RX-ref pair may be the HORIZONTAL one)" if abs(s) < 20 else ""))
        if "left" in res and "right" in res:
            s = rel("left", "ph_az") - rel("right", "ph_az")
            L.append(f"- **horizontal:** left − right = {s:+.0f} deg" + ("  (small: the RX-az/RX-ref pair may be the VERTICAL one)" if abs(s) < 20 else ""))
    return "\n".join(L)
