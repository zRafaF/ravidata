"""Figures from the tables only (no step is re-run). Every figure is optional: a missing table skips it.
Output: derived/analysis-*/figures/<name>.png (+ .pdf with --pdf). Change a representation here and re-run
`make plots` in seconds."""

from __future__ import annotations

import json

import numpy as np

from . import common as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FAM_ORDER = ["thr_nearest", "thr", "ca_cfar", "os_cfar", "rd_cfar", "cfar_dbscan", "grid", "grid_imu",
             "grid_oracle", "app_threshold", "app_cfar"]


GROUPS = [("detectors", ["thr_nearest", "thr", "ca_cfar", "os_cfar", "rd_cfar", "cfar_dbscan"]),
          ("grids and the recording app", ["grid", "grid_imu", "grid_oracle", "app_threshold", "app_cfar"])]


def fam_colour(fam):
    """Fixed hue per family inside its group (<= 8 hues per panel, never cycled)."""
    for _, g in GROUPS:
        if fam in g:
            return SERIES[g.index(fam)]
    return INK2


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
                         "lines.linewidth": 2, "axes.axisbelow": True, "figure.dpi": 110})
    return plt


def save(fig, name, pdf=False):
    p = C.out_dir("figures") / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    if pdf:
        fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    C.log(f"  figure {p.relative_to(C.DATA)}")


def fams(df):
    return [f for f in FAM_ORDER if f in set(df.family)]


# ------------------------------------------------------------------------------------------------ sensor
def fig_noise(plt, pdf):
    N = C.load_table("noise", required=False)
    if N is None or not len(N):
        return
    S = N[N["take"].isin(C.STATIC)]
    if not len(S):
        return
    cfgs = sorted(S.config.unique())
    fig, axs = plt.subplots(1, len(cfgs), figsize=(4 * len(cfgs), 3.2), sharey=True, squeeze=False)
    for ax, cfg in zip(axs[0], cfgs):
        g = S[S.config == cfg].groupby("r")
        ax.plot(g.mean_db.median().index, g.mean_db.median(), color=SERIES[0], label="mean power (static + leak)")
        ax.plot(g.tvar_db.median().index, g.tvar_db.median(), color=SERIES[1], label="temporal noise")
        if g.dop_db.median().notna().any():
            ax.plot(g.dop_db.median().index, g.dop_db.median(), color=SERIES[2], label="|v| >= 0.75 m/s bins")
        ax.axvline(0.26, color=INK2, lw=0.8, ls=":")
        ax.set_title(cfg); ax.set_xlabel("range (m)")
    axs[0][0].set_ylabel("dB (full scale)")
    axs[0][-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Static takes: power and noise per range bin (median over recordings and RX)", fontsize=10)
    save(fig, "sensor_noise_vs_range", pdf)


def fig_targets(plt, pdf):
    T = C.load_table("targets", required=False)
    if T is None or not len(T):
        return
    T = T[T.bg == "empty"]
    g = T.groupby(["take", "config", "id"]).agg(r_true=("r_true", "first"), r_meas=("r_meas", "median"),
                                                r_sd=("r_meas", "std"), snr=("snr_db", "median"),
                                                az_true=("az_true", "first"), az_meas=("az_meas", "median"),
                                                ghost=("ghost2_snr_db", "median")).reset_index()
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, cfg in enumerate(sorted(g.config.unique())):
        q = g[g.config == cfg]
        axs[0].errorbar(q.r_true, q.r_meas - q.r_true, yerr=q.r_sd, fmt="o", ms=5, color=SERIES[k], label=cfg,
                        elinewidth=1)
        axs[1].plot(q.r_true, q.snr, "o", ms=5, color=SERIES[k])
        axs[2].plot(q.az_true, q.az_meas, "o", ms=5, color=SERIES[k])
    axs[0].axhline(0, color=INK2, lw=0.8); axs[0].set_xlabel("tag range (m)"); axs[0].set_ylabel("range error (m)")
    axs[0].set_title("Range error at tagged objects"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel("tag range (m)"); axs[1].set_ylabel("SNR (dB)"); axs[1].set_title("SNR of tagged objects")
    lim = [-35, 35]
    axs[2].plot(lim, lim, color=INK2, lw=0.8); axs[2].set_xlabel("tag azimuth (deg)")
    axs[2].set_ylabel("radar azimuth (deg)"); axs[2].set_title("Interferometric azimuth")
    save(fig, "sensor_targets", pdf)


def fig_timing(plt, pdf):
    T = C.load_table("timing", required=False)
    if T is None or not len(T):
        return
    g = T.groupby("config").agg(fps=("fps", "median"), lost=("lost_pct", "median"), jit=("dt_std_ms", "median"))
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].bar(g.index, g.lost, color=SERIES[0], width=0.5); axs[0].set_ylabel("lost frames (%)")
    axs[1].bar(g.index, g.jit, color=SERIES[0], width=0.5); axs[1].set_ylabel("inter-frame jitter, std (ms)")
    for ax in axs:
        ax.tick_params(axis="x", labelsize=7)
    save(fig, "sensor_timing", pdf)


def fig_calib(plt, pdf):
    T = C.load_table("calib_targets", required=False)
    ext = C.load_json("extrinsic", None)
    if T is None or not len(T):
        return
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    axs[0].plot(T.r_true, T.range_resid_m, "o", color=SERIES[0], ms=5)
    axs[0].axhline(0, color=INK2, lw=0.8)
    axs[0].set_xlabel("tag range (m)"); axs[0].set_ylabel("range residual after fit (m)")
    if ext:
        axs[0].set_title(f"bias {ext['range_bias_m']:+.3f} m, t_rc {np.round(ext['t_rc'], 3)}", fontsize=8)
    axs[1].plot(T.az_true, T.az_resid_deg, "o", color=SERIES[0], ms=5, label="azimuth")
    axs[1].plot(T.el_true, T.el_resid_deg, "s", color=SERIES[1], ms=5, label="elevation")
    axs[1].axhline(0, color=INK2, lw=0.8); axs[1].legend()
    axs[1].set_xlabel("tag angle (deg)"); axs[1].set_ylabel("angle residual after fit (deg)")
    save(fig, "calib_residuals", pdf)


def fig_sync(plt, pdf):
    S = C.load_table("sync_check", required=False)
    if S is None or not len(S):
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for k, (rec, g) in enumerate(S.groupby("rec")):
        ax.plot(g.lag_s, g.agree, color=SERIES[k % 8], label=rec.split("/")[-1])
    ax.set_xlabel("lag added to the radar clock (s)"); ax.set_ylabel("nearest-return agreement (< 0.2 m)")
    ax.legend(fontsize=7)
    save(fig, "sync_check", pdf)


# ------------------------------------------------------------------------------------------------ truth
def fig_truth(plt, pdf):
    F = C.load_table("truth_frames", required=False)
    M = C.load_table("map_scale", required=False)
    if F is None or not len(F):
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3))
    g = F.groupby("take").agg(covered=("covered", "mean"), obj=("n_obj", "mean"))
    axs[0].bar(g.index, g.covered, color=SERIES[0], width=0.5); axs[0].set_ylabel("frames with truth")
    axs[0].tick_params(axis="x", labelsize=7)
    if M is not None and len(M):
        for k, (take, q) in enumerate(M.groupby("take")):
            axs[1].plot(q.window, q.ratio, "o", ms=4, color=SERIES[k], label=take, alpha=0.6)
        axs[1].axhline(1, color=INK2, lw=0.8)
        axs[1].set_xlabel("map window"); axs[1].set_ylabel("tag distance / map depth"); axs[1].legend(fontsize=7)
        axs[1].set_title("Metric scale of the map from AprilTags")
    save(fig, "truth_coverage_and_scale", pdf)


# ------------------------------------------------------------------------------------------------ evaluation
def _summary(variant="default"):
    S = C.load_table("eval_summary" + ("" if variant == "default" else f"_{variant}"), required=False)
    return S


def fig_eval_bars(plt, pdf):
    S = _summary()
    if S is None or not len(S):
        return
    A = S[S["take"].isna() & S.env.isna()] if "env" in S else S[S["take"].isna()]
    if not len(A):
        return
    A = A.set_index("family").loc[fams(A)].reset_index()
    metrics = [("f1_50", "F1 at 0.5 m"), ("recall_50", "recall at 0.5 m"), ("near_pd", "nearest obstacle found"),
               ("gospa", "GOSPA (m), lower is better")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    y = np.arange(len(A))
    for ax, (m, title) in zip(axs, metrics):
        if m not in A:
            continue
        ax.barh(y, A[m], color=SERIES[0], height=0.6)
        if f"{m}_lo" in A:
            ax.errorbar(A[m], y, xerr=[A[m] - A[f"{m}_lo"], A[f"{m}_hi"] - A[m]], fmt="none", ecolor=INK, lw=1)
        ax.set_yticks(y, A.family if ax is axs[0] else [""] * len(y)); ax.invert_yaxis(); ax.set_title(title)
    fig.suptitle("All takes, default settings (95 % block-bootstrap intervals)", fontsize=10)
    save(fig, "eval_default_overview", pdf)


def fig_default_vs_tuned(plt, pdf):
    base = _summary()
    if base is None:
        return
    rows = {"default": base}
    for pr in ("strict", "pooled"):
        t = _summary(f"tuned_{pr}")
        if t is not None and len(t):
            rows[f"tuned ({pr})"] = t
    if len(rows) < 2:
        return
    allf = lambda S: S[S["take"].isna() & (S.env.isna() if "env" in S else True)].set_index("family")
    fl = fams(base)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    w = 0.8 / len(rows)
    for k, (lab, S) in enumerate(rows.items()):
        A = allf(S).reindex(fl)
        for ax, m in zip(axs, ("f1_50", "near_pd")):
            ax.bar(np.arange(len(fl)) + k * w, A[m], width=w - 0.02, color=SERIES[k], label=lab)
    for ax, t in zip(axs, ("F1 at 0.5 m", "nearest obstacle found (0.25 m)")):
        ax.set_xticks(np.arange(len(fl)) + 0.4 - w / 2, fl, rotation=35, ha="right"); ax.set_title(t)
    axs[0].legend(fontsize=7)
    fig.suptitle("Default vs tuned settings (tuned on the ground truth; all takes)", fontsize=10)
    save(fig, "eval_default_vs_tuned", pdf)


def fig_eval_env(plt, pdf):
    S = _summary()
    if S is None or "env" not in S:
        return
    E = S[S.env.notna()]
    if not len(E):
        return
    envs = sorted(E.env.unique())
    fl = fams(E)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.8 / len(envs)
    for k, e in enumerate(envs):
        q = E[E.env == e].set_index("family").reindex(fl)
        ax.bar(np.arange(len(fl)) + k * w, q.f1_50, width=w - 0.02, color=SERIES[k], label=e)
    ax.set_xticks(np.arange(len(fl)) + 0.4 - w / 2, fl, rotation=35, ha="right")
    ax.set_ylabel("F1 at 0.5 m"); ax.legend()
    save(fig, "eval_by_environment", pdf)


def fig_rings(plt, pdf):
    S = _summary()
    if S is None:
        return
    A = S[S["take"].isna() & (S.env.isna() if "env" in S else True)]
    if not len(A):
        return
    rings = [10, 25, 50, 100]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), sharey=True, sharex=True)
    for row, (gname, gf) in enumerate(GROUPS):
        for fam in [f for f in gf if f in set(A.family)]:
            q = A[A.family == fam].iloc[0]
            col = fam_colour(fam)
            axs[row, 0].plot([r / 100 for r in rings], [q.get(f"recall_{r}") for r in rings], color=col, marker="o",
                             ms=4, label=fam)
            axs[row, 1].plot([r / 100 for r in rings], [q.get(f"r_recall_{r}") for r in rings], color=col,
                             marker="o", ms=4)
        axs[row, 0].set_ylabel(f"recall ({gname})"); axs[row, 0].legend(fontsize=7)
    axs[0, 0].set_title("Plane (range + azimuth)"); axs[0, 1].set_title("Range only")
    for ax in axs.ravel():
        ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 1.0], ["0.10", "0.25", "0.50", "1.00"])
    for ax in axs[1]:
        ax.set_xlabel("ring (m)")
    save(fig, "eval_recall_vs_ring", pdf)


def fig_fp_causes(plt, pdf):
    S = _summary()
    if S is None:
        return
    A = S[S["take"].isna() & (S.env.isna() if "env" in S else True)]
    causes = [c for c in ["fp_clutter_per_frame", "fp_floor_per_frame", "fp_ghost_per_frame",
                          "fp_behind_obstacle_per_frame", "fp_unlabelled_per_frame"] if c in A]
    if not len(A) or not causes:
        return
    A = A.set_index("family").loc[fams(A)].fillna(0)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    left = np.zeros(len(A))
    for k, c in enumerate(causes):
        ax.barh(np.arange(len(A)), A[c], left=left, color=SERIES[k], height=0.6, edgecolor=SURF, linewidth=1,
                label=c.replace("fp_", "").replace("_per_frame", ""))
        left += A[c].to_numpy()
    if "det_out_fov_per_frame" in A:
        ax.plot(A.det_out_fov_per_frame, np.arange(len(A)), "|", color=INK, ms=10, label="outside camera FOV")
    ax.set_yticks(np.arange(len(A)), A.index); ax.invert_yaxis()
    ax.set_xlabel("unmatched detections per frame (0.5 m ring)"); ax.legend(fontsize=7)
    save(fig, "eval_false_positive_causes", pdf)


def fig_pr(plt, pdf):
    D = C.load_table("eval_dets", required=False)
    Fr = C.load_table("eval_frames", required=False)
    if D is None or Fr is None or not len(D):
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (gname, gf) in zip(axs, GROUPS):
        for fam in [f for f in gf if f in set(D.family)]:
            d = D[(D.family == fam) & D.complete] if "complete" in D else D[D.family == fam]
            nt = Fr[(Fr.family == fam) & Fr.covered & Fr.complete].n_truth.sum()
            if not len(d) or not nt:
                continue
            o = np.argsort(-d.score.to_numpy())
            m = d.m50.to_numpy()[o].astype(float)
            tp = np.cumsum(m)
            ax.plot(tp / nt, tp / np.arange(1, len(m) + 1), color=fam_colour(fam), lw=1.5, label=fam)
        ax.set_xlabel("recall (0.5 m)"); ax.set_title(gname); ax.legend(fontsize=7)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].set_ylabel("precision (frames with complete truth)")
    save(fig, "eval_precision_recall", pdf)


def fig_pd_range(plt, pdf):
    Fr = C.load_table("eval_frames", required=False)
    if Fr is None or not len(Fr):
        return
    bins = np.arange(0.25, 4.26, 0.5)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (gname, gf) in zip(axs, GROUPS):
        for fam in [f for f in gf if f in set(Fr.family)]:
            q = Fr[(Fr.family == fam) & Fr.covered & np.isfinite(Fr.near_truth)]
            if not len(q):
                continue
            hit = np.abs(q.near_det - q.near_truth) <= 0.25
            idx = np.digitize(q.near_truth, bins)
            pd_ = [hit[idx == i].mean() if (idx == i).sum() > 20 else np.nan for i in range(1, len(bins))]
            ax.plot((bins[:-1] + bins[1:]) / 2, pd_, color=fam_colour(fam), marker="o", ms=3, lw=1.5, label=fam)
        ax.set_xlabel("range of the nearest obstacle (m)"); ax.set_title(gname); ax.legend(fontsize=7)
    axs[0].set_ylabel("found within 0.25 m")
    save(fig, "eval_detection_vs_range", pdf)


def fig_cost(plt, pdf):
    Cst = C.load_table("eval_cost", required=False)
    S = _summary()
    if Cst is None or S is None:
        return
    A = S[S["take"].isna() & (S.env.isna() if "env" in S else True)].set_index("family")
    ms = Cst.groupby("family").ms_per_frame.median()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for k, fam in enumerate(fams(Cst)):
        if fam in A.index:
            ax.plot(ms[fam], A.loc[fam, "f1_50"], "o", ms=8, color=SERIES[0])
            ax.annotate(fam, (ms[fam], A.loc[fam, "f1_50"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Python time per frame (ms, relative)"); ax.set_ylabel("F1 at 0.5 m")
    save(fig, "eval_cost_vs_quality", pdf)


# ------------------------------------------------------------------------------------------------ optimisation
def fig_pareto(plt, pdf):
    for prot in ("strict", "pooled"):
        T = C.load_table(f"opt_trials_{prot}", required=False)
        if T is None or not len(T):
            continue
        T = T[T.split.isin(["loto", "all"])] if prot == "strict" else T
        fl = fams(T)
        n = len(fl)
        cols = min(4, n)
        rows_ = int(np.ceil(n / cols))
        fig, axs = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.8 * rows_), squeeze=False, sharex=True,
                                sharey=True)
        from .optimise import pareto
        for ax, fam in zip(axs.ravel(), fl):
            q = T[T.family == fam]
            ax.plot(q.fp_per_frame_50, 1 - q.miss_50, ".", color=GRID if False else "#b9b8b3", ms=3)
            for k, (fold, g) in enumerate(q.groupby("fold")):
                v = g[["miss_50", "fp_per_frame_50"]].fillna(1e9).to_numpy()
                pf = g[pareto(v)].sort_values("fp_per_frame_50")
                ax.step(pf.fp_per_frame_50, 1 - pf.miss_50, where="post", color=SERIES[k % 8], lw=1.2)
            d = q[q.trial == 0]
            ax.plot(d.fp_per_frame_50, 1 - d.miss_50, "x", color=INK, ms=6)
            ax.set_title(fam, fontsize=9)
        for ax in axs.ravel()[n:]:
            ax.axis("off")
        fig.supxlabel("false positives per frame (0.5 m)"); fig.supylabel("recall (0.5 m)")
        fig.suptitle(f"Training Pareto fronts per fold ({prot}); x = default setting", fontsize=10)
        save(fig, f"opt_pareto_{prot}", pdf)


def fig_universality(plt, pdf):
    for prot in ("strict", "pooled"):
        R = C.load_table(f"opt_results_{prot}", required=False)
        if R is None or not len(R):
            continue
        L = R[(R.split == "loto")]
        O = R[(R.split == "oracle") & (R.selection == "tuned_f1")]
        if not len(L) or not len(O):
            continue
        fl = fams(R)
        vals = {}
        for sel in ("default", "tuned_f1"):
            vals[sel] = L[L.selection == sel].groupby("family").f1_50.mean().reindex(fl)
        vals["oracle (per scene)"] = O.groupby("family").f1_50.mean().reindex(fl)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.26
        for k, (lab, v) in enumerate(vals.items()):
            ax.bar(np.arange(len(fl)) + k * w, v, width=w - 0.02, color=SERIES[k], label=lab)
        ax.set_xticks(np.arange(len(fl)) + w, fl, rotation=35, ha="right")
        ax.set_ylabel("F1 at 0.5 m on held-out take (mean)")
        ax.set_title("Price of universality: one setting for all scenes vs the per-scene optimum")
        ax.legend(fontsize=7)
        save(fig, f"opt_price_of_universality_{prot}", pdf)


# ------------------------------------------------------------------------------------------------ CORDIC
def fig_cordic(plt, pdf):
    E = C.load_table("cordic_error", required=False)
    if E is None or not len(E):
        return
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    k = 0
    for (op, q), g in E[E.op != "atan2_dynamic"].groupby(["op", "q"]):
        axs[0].semilogy(g.iterations, g.max_err, marker="o", ms=4, color=SERIES[k], label=f"{op} {q}")
        k += 1
    axs[0].set_xlabel("iterations"); axs[0].set_ylabel("max |error|"); axs[0].legend(fontsize=7)
    axs[0].set_title("CORDIC vs float64")
    Dn = E[E.op == "atan2_dynamic"]
    for k, (q, g) in enumerate(Dn.groupby("q")):
        axs[1].semilogy(g.rel_level_db, g.max_err_deg, marker="o", ms=4, color=SERIES[k], label=q)
    axs[1].set_xlabel("input level below full scale (dB)"); axs[1].set_ylabel("max phase error (deg)")
    axs[1].invert_xaxis(); axs[1].legend(fontsize=7); axs[1].set_title("Phase vs input level")
    G = C.load_table("cordic_grid", required=False)
    if G is not None and len(G):
        for k, (q, g) in enumerate(G.groupby("q")):
            m = g.groupby("iterations").same_output_frac.mean()
            axs[2].plot(m.index, m, marker="o", ms=4, color=SERIES[k], label=q)
        axs[2].set_xlabel("iterations"); axs[2].set_ylabel("frames with identical grid output")
        axs[2].set_title("Occupancy grid: CORDIC vs float64"); axs[2].legend(fontsize=7)
    save(fig, "cordic_accuracy", pdf)
    P = C.load_table("cordic_phase", required=False)
    if P is not None and len(P):
        g = P.groupby(["method", "q", "iterations"]).p99_err_deg.median().reset_index()
        fig, ax = plt.subplots(figsize=(7, 3.4))
        k = 0
        for (m, q), h in g[g.iterations > 0].groupby(["method", "q"]):
            ax.semilogy(h.iterations, h.p99_err_deg, marker="o", ms=4, color=SERIES[k], label=f"{m} {q}")
            k += 1
        for m, h in g[g.iterations == 0].groupby("method"):
            ax.axhline(h.p99_err_deg.iloc[0], color=SERIES[k % 8], lw=1, ls="--", label=m)
            k += 1
        ax.set_xlabel("CORDIC iterations"); ax.set_ylabel("99th pct phase error (deg)")
        ax.set_title("Interferometric phase of real radar returns vs float64"); ax.legend(fontsize=6)
        save(fig, "cordic_phase_real_data", pdf)


FIGS = [fig_noise, fig_targets, fig_timing, fig_calib, fig_sync, fig_truth, fig_eval_bars, fig_default_vs_tuned,
        fig_eval_env, fig_rings,
        fig_fp_causes, fig_pr, fig_pd_range, fig_cost, fig_pareto, fig_universality, fig_cordic]


def run(only=None, pdf=False):
    plt = style()
    for f in FIGS:
        if only and not any(o in f.__name__ for o in only):
            continue
        try:
            f(plt, pdf)
        except Exception as e:                                 # one broken figure must not stop the others
            C.log(f"  {f.__name__}: skipped ({type(e).__name__}: {e})")
