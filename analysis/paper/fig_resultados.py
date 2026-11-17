"""Detection on the carried takes (JSON only). (a) found vs early alarms: for every family, the best 'found' among its
tuning trials at each early-alarm budget (pooled over both walks), and the same curve for the constant answer (a fixed
distance in every frame, no radar): the chance floor. (b) tuned settings, per walk: frames whose nearest thing is the
ground vs frames with an obstacle nearer than the ground. (c) one change at a time from CA-CFAR: persistence, motion
compensation and Doppler (found with <= 10 % early alarms)."""
from __future__ import annotations

import numpy as np

from .. import common as C
from . import common as P

FAMS = ["thr", "ca_cfar", "rd_cfar", "grid", "grid_doppler"]
BUDGETS = np.round(np.arange(0.0, 0.501, 0.01), 2)
ABL = ["CA-CFAR (defaults)", "2 of 3", "3 of 5", "track (confirm 3, hold 5)", "integrate a=0.8", "log-odds grid",
       "grid + IMU", "grid + map poses", "doppler nonzero", "doppler zero"]


def _outcome(at, det, tol=0.25):
    ok, fd = np.isfinite(at), np.isfinite(det)
    return (float((ok & fd & (np.abs(det - at) <= tol)).sum() / max(1, ok.sum())),
            float((fd & (~ok | (det < at - tol))).sum() / max(1, len(at))))


def run():
    out = dict(run=C.RUN, budgets=BUDGETS)
    T = C.load_table("opt_trials_pooled_arc")
    out["curves"] = {f: [float(g[g.arc_early_rate <= b].arc_pd.max()) if (g.arc_early_rate <= b).any() else 0.0
                         for b in BUDGETS] for f, g in T.groupby("family") if f in FAMS}
    F = C.load_table("eval_frames"); F = F[F["take"].isin(C.CARRIED) & F.arc_covered.astype(bool) & (F.family == "thr")]
    at = F.arc_truth.to_numpy()
    cons = [(_outcome(at, np.full(len(at), c))) for c in np.arange(0.3, 6.0, 0.02)]
    out["constant"] = [max([f for f, e in cons if e <= b] or [0.0]) for b in BUDGETS]
    # (b) cases, tuned settings
    TF = C.load_table("truth_frames").set_index(["rec", "f"])
    Ft = C.load_table("eval_frames_tuned_pooled_arc")
    Ft = Ft[Ft["take"].isin(C.CARRIED) & Ft.arc_covered.astype(bool)]
    cases = []
    if "arc_obs_r" in TF:
        J = TF.reindex(list(zip(Ft.rec, Ft.f)))
        gr = (J.arc_floor_r.to_numpy() <= J.arc_obs_r.to_numpy())
        for take in C.CARRIED:
            for fam in FAMS:
                m = ((Ft["take"] == take) & (Ft.family == fam)).to_numpy()
                for case, s in (("ground", gr), ("obstacle", ~gr)):
                    q = m & s
                    if q.sum() < 20:
                        continue
                    f, e = _outcome(Ft.arc_truth.to_numpy()[q], Ft.arc_det.to_numpy()[q])
                    cases.append(dict(take=take, family=fam, case=case, frames=int(q.sum()), found=f, early=e))
        out["ground_share"] = {take: float(np.mean(gr[(Ft["take"] == take).to_numpy() & (Ft.family == "thr").to_numpy()]))
                               for take in C.CARRIED}
    out["cases"] = cases
    A = C.load_table("ablation_summary").set_index("variant")
    out["ablation"] = [dict(variant=v, pd10=float(A.loc[v, "pd_at_10"]), pd20=float(A.loc[v, "pd_at_20"]))
                       for v in ABL if v in A.index]
    P.save_json("fr_dados", out)
    print({f: round(c[10], 2) for f, c in out["curves"].items()}, "constant@10%", round(out["constant"][10], 2),
          "cases", len(cases), "ablation", [(a["variant"], round(a["pd10"], 2)) for a in out["ablation"]])


if __name__ == "__main__":
    run()
