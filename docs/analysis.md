# Radar analyses

Everything here runs on a CPU (your PC, not the GPU pod). Each step writes **tables** (Parquet / JSON) under
`data/ravidata/derived/analysis-v1/tables/`; figures are drawn **from those tables only** by `make plots`, so a
different representation is a change in `analysis/plots.py` and a re-run of seconds, never a re-run of the
analysis. Every run appends a line to `derived/analysis-v1/runs.jsonl` (step, settings, commit, duration).

```
make analysis-quick      # every step on ~7 s of every recording -> derived/analysis-quick (~6 min); check the figures
make analysis            # every step on the full data, in order (~3-4 h, see the table); CORDIC not included
make plots               # figures from whatever tables exist (QUICK=1 for the quick ones)
make an-<step>           # one step; QUICK=1 for its quick version; AN_TAKES / FAMILIES / TRIALS / PROTOCOL / WORKERS
```

Every long loop shows a progress bar with the time left; each step starts with `[i/n]` and the time so far.

Quick results go to a separate folder (`derived/analysis-quick`) and are never mixed with full ones. Delete that
folder whenever you like.

## Steps

| make | what | tables | full run (4 workers) |
|---|---|---|---|
| `an-tags` | AprilTag poses in every 2nd video frame and every map keyframe (OpenCV aruco 36h11, undistorted corners, IPPE_SQUARE) | `tags` | ~10-15 min |
| `an-extrinsic` | radar-camera offset `t_rc`, range bias, sign and phase offset of both antenna pairs, from the tagged objects of the static takes | `calib_targets`, `extrinsic.json`, `angle_calib.json` | < 1 min |
| `an-truth` | obstacle truth at every radar frame (below) | `truth_sectors`, `truth_frames`, `map_scale` | ~3 min |
| `an-sensor` | noise floor per bin / RX / config, tagged-object range error / SNR / angles per frame, frame loss and jitter, sync check | `noise`, `targets`, `timing`, `sync_check` | ~2 min |
| `an-eval` | every pipeline family with its default settings over every recording | `eval_frames`, `eval_dets`, `eval_cost`, `eval_summary` | ~15 min |
| `an-opt` | multi-objective tuning of every family + generalisation folds | `opt_trials_*`, `opt_results_*`, `opt_best_params_*.json` | pooled ~1.5-3 h; strict see below |
| `an-eval-tuned` | every family again with the tuned settings | `eval_*_tuned_<protocol>` | ~15 min |
| `an-cordic` | (not in `make analysis`; to be redone on the C code) fixed-point CORDIC vs float64: function error, phase of real returns, occupancy grid | `cordic_error`, `cordic_phase`, `cordic_grid`, `cordic_reference.json` | ~30 min |
| `plots` | figures | `figures/*.png` (`--pdf` for PDF) | seconds |

`make static-depth TAKE=all` (docs/maps.md) before `an-truth` makes the static truth complete (walls, furniture
without tags); without it the static takes only have the tagged objects (see *Ground truth*).

## Watching a recording: `make play`

```
make play REC=indoor_course/01_D1_tcc-1d-40mm START=20 SECONDS=15 PLAY_FAMILIES="ca_cfar os_cfar grid"
make play REC=static_pass1/04_S1 PLAY_ARGS="--bg empty --gif"
```

An MP4 (and a GIF with `--gif`) in `derived/analysis-play/`, at the radar's frame rate:

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
  camera; obstacles are 0.12-2.0 m above the floor. Tags on people, chairs and brooms seen within 0.1 s are added
  as obstacle points.
- **Static takes**: the tagged objects (median pose over the recording; floor tags are floor), plus the static-depth
  surfaces when they exist. With tags only the truth is *incomplete*: recall is scored, but detections that are
  not near a tag are counted as *unlabelled*, not as false positives (`complete` column).

## Pipeline families

Defaults are textbook starting values, never tuned on this data. All families share the preprocessing options
(background: none / causal EMA / empty-scene reference [static takes]; Doppler: sum / zero / non-zero / max;
RX combining: mean / coherent / max), the range gate and an optional causal M-of-N persistence.

| family | detector |
|---|---|
| `thr_nearest`, `thr` | power over the frame's noise estimate; nearest bin only / every peak |
| `ca_cfar` | 1D CFAR along range, cell-averaging / greatest-of / smallest-of |
| `os_cfar` | 1D ordered-statistic CFAR |
| `rd_cfar` | 2D CA-CFAR on the range-Doppler map (2D configs) |
| `cfar_dbscan` | CA-CFAR + DBSCAN over the last N frames, nearest point per cluster |
| `grid` | CFAR / threshold front end + log-odds occupancy grid (Gaussian inverse sensor model, ray-marched free space, decay, blur / morphology, hysteresis, connected components) |
| `grid_imu` | the grid moved with the phone's yaw and an assumed walking speed while the IMU says "walking" |
| `grid_oracle` | the grid moved with the map trajectory (carried takes): the best case for motion compensation |
| `app_threshold`, `app_cfar` | the recording app's own detectors (`ravidata.processing`), replayed frame by frame |

Angles come from the phase of `x_b * conj(x_a)` (float64) of the azimuth pair RX2-RX1 and the elevation pair
RX1-RX0, with the sign / offset fitted by `an-extrinsic`.

## Metrics

- **Rings**: Hungarian matching on plane distance, TP / FN / FP within 0.10, 0.25, 0.50 and 1.00 m; recall,
  precision, F1; average precision per ring (greedy by score, as nuScenes).
- **GOSPA** (c = 0.5 m, p = 2, α = 2) with its localisation, missed and false parts.
- **Range only** (no angle needed): distinct truth ranges vs distinct detection ranges (`r_*`), and the nearest
  obstacle (`near_*`): found within 0.25 m, early (a nearer false return), error.
- **Corridor**: nearest obstacle in the corridor, truth vs detection: detection rate, error, false alarms.
- **False-positive cause** (0.5 m ring): `behind_obstacle`, `ghost` (≈ 2× an obstacle range), `floor`,
  `unlabelled` (incomplete truth), `clutter` (nothing there); plus detections outside the camera FOV.
- 95 % intervals: block bootstrap over 5 s blocks (frames are correlated).
- Cost: Python time per frame, a *relative* number only (the embedded numbers come from the firmware).

## Optimisation and generalisation

Optuna NSGA-II, two objectives: miss rate and false positives per frame at the 0.5 m ring (a ROC-like Pareto
front); the default setting is always trial 0; same number of trials for every family. A fold's operating point
is the training-Pareto trial with the best F1 at 0.5 m (`tuned_f1`) or the lowest GOSPA (`tuned_gospa`).

- `PROTOCOL=pooled` (default): one study per family over every take; folds are formed afterwards from per-take
  statistics. Cheap; the sampler saw every take, so held-out numbers are slightly optimistic.
- `PROTOCOL=strict`: one study per fold; the sampler only sees the training takes. Folds: leave-one-take-out
  (5), leave-one-environment-out (2) and `oracle` (train = test = one take: the per-scene optimum, an upper bound).
  About 12× the pooled cost: e.g. `make an-opt PROTOCOL=strict TRIALS=60 FAMILIES="thr ca_cfar grid grid_imu"`
  (~3-5 h).

**Price of universality** = per-scene optimum minus one setting for every scene on the held-out take
(`opt_price_of_universality_*.png`). A bad result is reported as it is.

Optuna keeps one SQLite file per study in `derived/analysis-v1/optuna/`; a study is resumed if it exists (raise
`TRIALS` to continue it). On a network or virtual drive where SQLite locking fails, set
`RAVIDATA_OPTUNA_DIR` to a local folder.

## CORDIC

`analysis/cordic.py` is a bit-true integer model of the CORDIC algorithm (rotation mode for sin/cos, vectoring
mode for atan2 / magnitude) for q1.15 and q1.31 and any number of iterations. It reports: the function error vs
iterations; the phase error vs input level (fixed point loses small inputs: normalise before converting); the
phase of real radar returns with CORDIC and with float32 `atan(imag/real)` (with and without the quadrant fix)
against float64; and the occupancy grid with CORDIC sin/cos against float64. Published cycle counts (ST AN5325)
are stored next to the results for the cost axis.

## Numerical rules

Cube values are 1e-2 .. 1e-4 of full scale: everything is float64, phases come from the complex product
(`np.angle(x_b * conj(x_a))`), never from `atan(imag/real)`; arcsin arguments are clipped; dB only after the
linear arithmetic.

## Known limits (read before quoting a number)

- The sync offset between video and radar is the dataset's (0.56 s, EXP-014); `sync_check` shows the agreement
  of the radar's nearest return with the truth for lags of ±1 s but does not change it.
- The interferometric angles depend on the calibration from a handful of tagged objects; `calib_residuals.png`
  and the mean resultant length R in `angle_calib.json` show how consistent it is. Range-only metrics do not
  depend on it.
- Static truth without static-depth is incomplete (tags only).
- Map truth is as good as the map: Pi3X keyframe depth, metric through the tags (`map_scale`).
