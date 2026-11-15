# Radar analyses

Everything here runs on a CPU (your PC, not the GPU pod). Each step writes **tables** (Parquet / JSON) under
`results/analysis-v2/tables/`; figures are drawn **from those tables only** by `make plots`, so a
different representation is a change in `analysis/plots.py` and a re-run of seconds, never a re-run of the
analysis. Every run appends a line to `results/analysis-v2/runs.jsonl` (step, settings, commit, duration).

```
make analysis-quick      # every step on ~7 s of every recording -> results/analysis-v2-quick (~6 min); check the figures
make analysis            # every step on the full data, in order (~3-4 h, see the table)
make plots               # figures from whatever tables exist (QUICK=1 for the quick ones)
make an-<step>           # one step; QUICK=1 for its quick version; AN_TAKES / FAMILIES / TRIALS / PROTOCOL / WORKERS
```

Every long loop shows a progress bar with the time left; each step starts with `[i/n]` and the time so far.

Results live in `results/analysis-v2/` and are kept in git (tables, figures, videos, Optuna studies, the run
log, `REPORT.md`), so a paper number can always be traced to its table and run. `results/analysis-v1/` is the
superseded first version; what changed is in `CHANGELOG.md`, which also lists the steps not yet re-run in
`analysis-v2` (ablation, plots, story, report, the plane objective, the strict protocol). The recordings are on
Hugging Face; `make data-analysis` downloads what the steps read. `RAVIDATA_RUN=<name>` writes a full run to
`results/<name>` instead (to compare with the published one). To redo only what changed: `make analysis FROM=<step>` (tags -> extrinsic -> truth -> sensor
-> eval -> ablation -> opt -> eval-tuned -> floor -> plots -> story -> report); the ablation caches every recording's
result (keyed by settings, truth table and code) and the Optuna studies resume.

Quick results go to a separate folder (`results/analysis-v2-quick`, not in git) and are never mixed with full ones. Delete that
folder whenever you like.

## Steps

| make | what | tables | full run (4 workers) |
|---|---|---|---|
| `an-tags` | AprilTag poses in every 2nd video frame and every map keyframe (OpenCV aruco 36h11, undistorted corners, IPPE_SQUARE) | `tags` | ~10-15 min |
| `an-extrinsic` | range bias and sign / phase offset of both antenna pairs, fitted on the tagged person walking in front of the still radar (mean-removed cube; still tags as a fallback below 100 sightings); residuals of the still tagged objects as a check; `t_rc` = 0 | `calib_moving`, `calib_targets`, `extrinsic.json`, `angle_calib.json` | < 1 min |
| `an-truth` | obstacle truth at every radar frame (below) | `truth_sectors`, `truth_frames`, `map_scale`, `truth_objects` | ~5 min |
| `an-sensor` | noise floor per bin / RX / config, tagged-object range error / SNR / angles per frame, frame loss and jitter, sync check | `noise`, `targets`, `timing`, `sync_check` | ~2 min |
| `an-eval` | every pipeline family with its default settings over every recording | `eval_frames`, `eval_dets`, `eval_cost`, `eval_summary` | ~15 min |
| `an-ablation` | one change at a time from CA-CFAR defaults (background, RX combining, Doppler, near cut, integration over frames, detector, M-of-N, tracker, DBSCAN, grids, IMU height gate), each over a threshold sweep; cached per recording | `ablation_points`, `ablation_summary` | ~30-60 min first time, seconds after |
| `an-opt` | multi-objective tuning of every family + generalisation folds | `opt_trials_*`, `opt_results_*`, `opt_best_params_*.json` | pooled, arc, 80 trials, 5 families: ~30 min; every family and both objectives at 150 trials: hours; strict see below |
| `an-eval-tuned` | every family again with the tuned settings | `eval_*_tuned_<protocol>` | ~15 min |
| `an-floor` | where the early alarms of the carried takes come from: outcome by floor margin (floor-in-beam range minus obstacle range) and by pitch, and what the camera map has at the range of alarms with nothing within 2 m (still-radar empty scenes as control); no detector is rerun | `floor_margin`, `floor_pitch`, `phantom`, `phantom_frames` | seconds |
| `plots` | figures: paper candidates (`--all-figs` adds the diagnostics) | `figures/*.png` (`--pdf` for PDF) | seconds |
| `story` | explanatory figures: how the truth is built at one radar frame, radar range-time maps with the truth and every detector's nearest detection | `figures/story/*.png` | ~1 min |
| `report` | every number as Markdown tables | `REPORT.md` | seconds |

Inputs besides the recordings: the Pi3X maps and `frames_<take>.json` (`derived/maps-v1`), the fused trajectory
(`derived/maps-v1/fused/pi3x`, from `make fuse`) and the still-take depth (`derived/static-depth-v1/depthpro`, from
`make static-depth TAKE=all`). `make data-analysis` downloads all of them. Without the fused trajectory the truth
uses the nearest keyframe; without the static depth the still takes only have the tagged objects (see *Ground
truth*). Both fall back silently (the log says which source was used; `truth_frames.source`). `make fuse` and
`make static-depth` remake them, but not bit for bit on another machine, so use the published files to reproduce
the published numbers.

## Watching a recording: `make play`

```
make play REC=indoor_course/01_D1_tcc-1d-40mm START=20 SECONDS=15 PLAY_FAMILIES="ca_cfar os_cfar grid"
make play REC=static_pass1/04_S1 PLAY_ARGS="--bg empty --gif"
```

An MP4 (and a GIF with `--gif`) in `results/analysis-v2/videos/`, at the radar's frame rate:

- **Left:** the synchronised camera frame. The truth points (white rings) and every family's detections are
  projected into it.
- **Right, top:** bird's-eye view of the radar plane, showing this frame's power only (no accumulation). It is a
  2-element beam over the azimuth pair, in dB over the frame's noise. The view also has the truth (nearest
  obstacle per sector), the camera FOV, the walking corridor and the detections.
- **Right, bottom:** the range profile with the first CFAR family's threshold, the detections and the truth ranges.

It uses the full-run truth table if there is one, else the quick one (first seconds only). `--params tuned_pooled`
shows the tuned settings.

## Ground truth

One representation for every take, in the **radar frame** (camera axes, origin at `t_rc`): for each radar frame
and each 4° azimuth sector inside the camera field of view (±28°), the nearest obstacle point and the nearest
floor point; plus the nearest obstacle in the walking corridor (|X| < 0.5 m, 0.3 < Z < 3 m). Positions are in the
radar's range-azimuth plane: X = r sin(az), Z = r cos(az) with r the slant range (what the radar measures).

- **Carried takes** (`indoor_course`, `outdoor_course`): keyframe depth of the feed-forward map (`MAP_MODEL`,
  default Pi3X), made metric by the AprilTags (tag distance from solvePnP over map depth at the tag centre, per
  window: table `map_scale`), moved to the exact radar frame time with the fused trajectory (`make fuse`), else the
  nearest keyframe within 0.25 s. "Up" comes from the phone gravity sensor; the floor is the densest level below the
  camera; obstacles are anything more than 0.12 m above the floor (nothing is removed overhead: a ceiling or roof
  in the beam counts, as it does for the radar). Tags on people, chairs and brooms seen within 0.1 s are added
  as obstacle points.
- **Static takes**: the tagged objects (median pose over the recording; floor tags are floor), plus the static-depth
  surfaces when they exist. With tags only the truth is *incomplete*: recall is scored, but detections that are
  not near a tag are counted as *unlabelled*, not as false positives (`complete` column).

## Pipeline families

Defaults are textbook starting values, never tuned on this data. All families share the preprocessing options
(background: none / causal EMA / empty-scene reference [static takes]; Doppler: sum / zero / non-zero / max;
RX combining: mean / coherent / max / one RX), the noise reference (`noise_q`: this quantile of the frame's power;
0.5 = the median, which is the clutter level, not the receiver noise), the range gate, an optional causal M-of-N
persistence, an optional causal running median of the nearest detection (`med` frames; 1 = off) and the
optional **IMU height gate** (radar + IMU fusion): the phone's gravity levels every detection's elevation, and
returns that land on the floor are dropped (`radar_h` assumed radar height, `h_min`, `el_off` a residual
elevation bias; all tunable). `make an-eval` runs every family twice: as is and with the height gate.

| family | detector |
|---|---|
| `thr_nearest`, `thr` | power over the frame's noise estimate; nearest bin only / every peak |
| `ca_cfar` | 1D CFAR along range, cell-averaging / greatest-of / smallest-of |
| `os_cfar` | 1D ordered-statistic CFAR |
| `rd_cfar` | 2D CA-CFAR on the range-Doppler map (2D configs) |
| `cfar_dbscan` | CA-CFAR + DBSCAN over the last N frames, nearest point per cluster |
| `grid` | CFAR / threshold front end + log-odds occupancy grid (Gaussian inverse sensor model, ray-marched free space, decay, blur / morphology, hysteresis, connected components; one target per component, or one per component and 4° sector) |
| `grid_imu` | the grid moved with the phone's yaw and an assumed walking speed while the IMU says "walking" |
| `grid_oracle` | the grid moved with the map trajectory (carried takes): the best case for motion compensation |
| `grid_doppler` | 2D configs: per range-Doppler cell CA-CFAR and its own azimuth, into a range-azimuth grid with three Doppler layers (by closing speed) weighted before reading the obstacles (`analysis/grid_doppler.py`) |
| `grid_legacy`, `grid_legacy_imu` | the earlier C firmware's energy grid: no detector in front, every range bin's energy into its 2° azimuth cell, 5 cm map, decay per second, DBSCAN on cells above a threshold, nearest cell per cluster; `_imu` turns it with the phone orientation as the firmware did |
| `app_threshold`, `app_cfar` | the recording app's own detectors (`ravidata.processing`), replayed frame by frame |

Angles come from the phase of `x_b * conj(x_a)` (float64) of the azimuth pair RX2-RX1 and the elevation pair
RX1-RX0, with the sign / offset fitted by `an-extrinsic`; ranges are corrected by the range bias it fits
(`analysis-v2`: azimuth sign +1, offset 2.99 rad, R 0.96, about 3° median error when fitted on one take and tested
on another; range bias -0.011 m; the elevation offset is approximate).

## Metrics

- **Rings**: Hungarian matching on plane distance, TP / FN / FP within 0.10, 0.25, 0.50 and 1.00 m; recall,
  precision, F1; average precision per ring (greedy by score, as nuScenes).
- **GOSPA** (c = 0.5 m, p = 2, α = 2) with its localisation, missed and false parts.
- **Range only** (no angle needed): distinct truth ranges vs distinct detection ranges (`r_*`).
- **Nearest in the camera view** (`near_*`): the nearest detection whose own azimuth / elevation is inside the
  camera view vs the camera truth: found within 0.25 m, early (a nearer false return), error.
- **Corridor**: nearest obstacle in the corridor, truth vs detection: detection rate, error, false alarms.
- **Arc** (the handheld view: only the distance to the nearest obstacle matters, not its direction): the nearest
  detection at any angle vs the nearest obstacle anywhere in the radar beam (±60° azimuth, ±30° elevation).
  `arc_pd`: found within 0.25 m; `arc_early_rate`: something reported more than 0.25 m nearer (or nothing there);
  `arc_alert_recall` / `arc_alert_false_rate`: "obstacle nearer than 2 m" as a yes / no alarm. The truth is
  concentrated (indoors near 0.9 m most of the time), so `arc_pd` has a high floor: `arc_pd_const` is the best a
  constant answer scores (at `arc_const_m`). `arc_pd_bal` averages the found rate over truth-distance classes
  (0-0.8-1.1-1.5-2.2-∞ m), with its own constant floor `arc_pd_bal_const`; compare every number with its floor. The beam is wider
  than the camera, so the arc truth comes from a local map: every keyframe of the same map window within ±4 s,
  placed with the window's own poses (a window is self-consistent; the fused map's joins can leave ghost surfaces
  and are not used for truth). Carried takes only (`arc_covered`).
- **False-positive cause** (0.5 m ring): `behind_obstacle`, `ghost` (≈ 2× an obstacle range), `floor`,
  `unlabelled` (incomplete truth), `clutter` (nothing there); plus detections outside the camera FOV.
- 95 % intervals: block bootstrap over 5 s blocks (frames are correlated).
- Cost: Python time per frame, a *relative* number only (the embedded numbers come from the firmware).

## Optimisation and generalisation

Optuna NSGA-II, two objectives (a ROC-like Pareto front), `OBJECTIVE=plane | arc | both` (default both):
`plane` = miss rate and false positives per frame at the 0.5 m ring; `arc` = nearest obstacle not found within
0.25 m, balanced over distance classes (`arc_miss_bal`), and early alarms (carried takes only). The default setting
is always trial 0; same number of trials for every family; NSGA-II seed 0. A fold's operating point is the
training-Pareto trial with the best F1 at 0.5 m (`tuned_f1`), the lowest GOSPA (`tuned_gospa`) or, for arc, the
best balanced found × (1 − early) (`tuned_arc`, `arc_score_bal`).

- `PROTOCOL=pooled` (default): one study per family over every take; folds are formed afterwards from per-take
  statistics. Cheap; the sampler saw every take, so held-out numbers are slightly optimistic.
- `PROTOCOL=strict`: one study per fold; the sampler only sees the training takes. Folds: leave-one-take-out
  (5), leave-one-environment-out (2) and `oracle` (train = test = one take: the per-scene optimum, an upper bound).
  About 12× the pooled cost: e.g. `make an-opt PROTOCOL=strict TRIALS=60 FAMILIES="thr ca_cfar grid grid_imu"`
  (~3-5 h).

**Price of universality** = per-scene optimum minus one setting for every scene on the held-out take
(`opt_price_of_universality_*.png`). A bad result is reported as it is.

Optuna keeps one SQLite file per study in `results/analysis-v2/optuna/`; a study is resumed if it exists (raise
`TRIALS` to continue it). The published studies are in git, so `make an-opt` on a fresh clone only rebuilds the
tables; use a new `RAVIDATA_RUN` to repeat the search. A study resumed after a change of the search space mixes
trials of both versions. On a network or virtual drive where SQLite locking fails, set
`RAVIDATA_OPTUNA_DIR` to a local folder.

## Numerical rules

Cube values are 1e-2 .. 1e-4 of full scale: everything is float64, phases come from the complex product
(`np.angle(x_b * conj(x_a))`), never from `atan(imag/real)`; arcsin arguments are clipped; dB only after the
linear arithmetic.

## Known limits (read before quoting a number)

- The sync offset between video and radar is the dataset's `take.json` value (dataset v1.1 or newer, required:
  indoor_course -2.44 s instead of the old 0.56 s; see `CHANGELOG.md`). `sync_check`
  shows the agreement of the radar's nearest return with the truth for lags of ±1 s but does not change it.
  outdoor_course keeps the default 0.56 s, which only weak checks support (floor return: +0.4 .. +0.6 s).
- The interferometric angles depend on the calibration from the walking person (`calib_moving`); the mean
  resultant length R in `angle_calib.json` shows how consistent it is. With several scatterers in one range bin
  (walking in clutter) the angle is still poor. Range-only metrics do not depend on it.
- While walking, the strongest near return at about the camera height (indoors 1.36 m, camera 1.34 m) is the floor
  below the radar, outside the ±30° elevation window of the floor truth, so it counts as an early alarm.
- The distance classes of `arc_pd_bal` were chosen from the indoor truth, i.e. on the test data. The pooled protocol
  lets the sampler see both carried takes; held-out numbers are optimistic.
- Static truth without static-depth is incomplete (tags only).
- Map truth is as good as the map: Pi3X keyframe depth, metric through the tags (`map_scale`).
