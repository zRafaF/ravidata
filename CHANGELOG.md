# Changelog: code, results and dataset versions

Results folders under `results/` are named after the analysis version that made them. Each one keeps its
`runs.jsonl` (step, settings, commit, duration of every run). The dataset on Hugging Face has its own versions
(git tags on the dataset repository, see its card).

## analysis-v2 (2026-11-15), the current results

`results/analysis-v2` is the canonical line. Code: the working tree after commit `a8253ca` (the run log records
`a8253ca` because the changes were not yet committed). Dataset: v1.0 files with the indoor_course correction
applied (then through a local override file, since folded into dataset v1.1, which the code now requires).

What changed from v1, and why:

- **indoor_course time alignment.** The video offset of `indoor_course` was an assumed default (0.56 s, the
  sound-output delay of the recorder; no radar-vs-video check passed for that take). Three independent checks put
  the video 3.0 s late, i.e. an offset of **-2.44 s**: the radar's floor return against the camera height above the
  floor tags; the radar range-time ridges against the map and AprilTag ranges (peaks at -2.7 .. -3.4 s, nothing at
  0.56); and the radial speed from the radar Doppler against the walking speed from the camera map (correlation
  -0.77 / -0.68 at zero lag with -2.44 s, about -0.1 at ±3 s). With it, the phone-minus-host clock of this take
  (-0.33 s) agrees with the other takes (-0.25 .. -0.37 s). The IMU moves with the video. The other takes are
  unchanged (outdoor_course: the floor return gives +0.4 .. +0.6 s, consistent with its 0.56 s). In dataset v1.1
  (`take.json` → `sync`, the old value in `sync.corrections`; the correction table is `SYNC_CORRECTIONS` in
  `scripts/hf_dataset.py`); the `tags` table is re-timed from `t_video` with the current sync
  (`common.load_tags`), so the video does not have to be read again.
- **Range and angle calibration from the moving person.** `extrinsic` now fits the range bias and the sign and
  offset of the antenna-pair phases on the tagged person walking in front of the still radar (the recording's mean
  removed, so the person is the only return in their bin: 859 sightings over 3 takes) instead of 39 rows from still
  tags, several of which shared a range bin with each other or with walls. Result: range bias +0.117 m → -0.011 m;
  azimuth sign -1 → +1, offset -1.49 → 2.99 rad (R 0.80 → 0.96; the old fit mirrored left and right); elevation
  offset 1.77 → 0.25 rad (still approximate: the targets span only a few degrees). Fitted on one take and tested on
  another, the azimuth error is 2.8-2.9° (median). The still-tag fit remains as a fallback below 100 sightings.
  New table `calib_moving`.
- **Metrics.** The arc metric had no skill floor: indoors the nearest obstacle is near 0.9 m most of the time, so a
  constant answer scored about 0.58 on `arc_pd`, above most pipelines. Added: `arc_pd_const` / `arc_const_m` (the
  best constant answer), `arc_pd_bal` (found rate averaged over truth-distance classes 0-0.8-1.1-1.5-2.2-∞ m) and
  `arc_pd_bal_const`. The class edges were chosen from the indoor truth distribution.
- **Optimisation.** The `arc` objective is now (`arc_miss_bal`, `arc_early_rate`), the operating point the best
  `arc_score_bal`. Search space: `noise_q` (noise reference = a quantile of the frame's power; 0.5 = the old
  median) and `med` (causal running median of the nearest detection; 1 = off). Defaults reproduce the old
  behaviour.
- **New family** `grid_doppler` (2D configs): range-azimuth grid with Doppler layers (`analysis/grid_doppler.py`).
- **Thesis figure exporters** `analysis/paper/` (text-free panels and small JSON files in `results/<run>/paper/`).

What has been run into `results/analysis-v2` (see `runs.jsonl`): `tags`, `extrinsic`, `truth`, `sensor`, `eval`
(before `grid_doppler` existed: the default and height-gate tables do not contain it), `opt` and `eval-tuned` with
`PROTOCOL=pooled OBJECTIVE=arc TRIALS=80` for `thr ca_cfar rd_cfar grid grid_doppler`, `floor`, and the paper
exporters. **Not yet run:** `ablation`, `plots`, `story`, `report` (so there is no `REPORT.md`, `FINDINGS.md` or
`figures/` yet), the `plane` objective, the `strict` protocol, the other families in `opt`.

## Code clean-up (2026-11-15, after the v2 runs; results unchanged)

- Dataset v1.1 (indoor_course clock fix, `dataset_version` in every `take.json`, card with the derived files and
  versions; `derived/maps-v1/fused` and `derived/static-depth-v1` published). `make hf-status`, `make hf-patch`,
  `make hf-release` (`scripts/hf_release.py`) update the hub from the local copy; `make static-depth-upload` is
  replaced by `make hf-release`.
- `analysis/sync_overrides.json` removed: the analysis reads the corrected `take.json` and refuses a dataset older
  than v1.1. Same `Sync` for every take as before (checked).
- The CORDIC step (`analysis/cordic.py`, `make an-cordic`, its plots and test) removed: the embedded part is no
  longer in the project's scope.
- `TRIALS` default 150 → 80, the setting of `results/analysis-v2` (more trials would extend its stored studies).

## analysis-v1 (2026-11-14), superseded

`results/analysis-v1`: commits `3a2478d` (tags, extrinsic, optimisation trials) and `1a35898` (truth, eval,
eval-tuned, ablation, floor, figures, report), dataset v1.0
with offset 0.56 s for indoor_course. Kept unchanged as the record of what the first version of the analysis
produced. Its indoor_course results are misaligned in time by 3 s, its angles are mirrored and its ranges are 3 bins
short; `FINDINGS.md` and `REPORT.md` there are superseded, and the explanation they give for the early alarms
indoors (the carrier's body) is replaced by the floor below the radar (its range follows the camera height:
1.36 m against 1.34 m).
