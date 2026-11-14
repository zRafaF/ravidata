# Mapping methods and running metrics (notes for the paper)

## Why maps

The radar is carried. Each radar frame needs the pose of the rig and the geometry in front of it: a camera pose
for every keyframe and a dense depth map. They serve as ground truth (the "oracle" pose, the clearance in front
of the radar) for the radar algorithms, and the AprilTags give their metric scale and an independent check.

## Models (feed-forward, one network pass per window, no per-scene optimisation)

| model | checkpoint | input | output scale | weights licence |
|---|---|---|---|---|
| [MapAnything](https://arxiv.org/abs/2509.13414) | `facebook/map-anything` | frames + calibrated intrinsics | metric | CC BY-NC 4.0 |
| [Pi3X](https://arxiv.org/abs/2507.13347) (π³ extended) | `yyfz233/Pi3X` | frames + calibrated intrinsics | approximately metric | CC BY-NC 4.0 |
| [Depth Anything 3](https://arxiv.org/abs/2511.10647) | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | frames (+ intrinsics if the API accepts them) | metric | CC BY-NC 4.0 |
| [VGGT](https://arxiv.org/abs/2503.11651) (baseline) | `facebook/VGGT-1B` | frames only | up to scale | non-commercial |

Repositories and commits: `pod/pins.env`; the commit that actually ran is in every `run.json`. Mixed precision
(bf16 on Ampere and newer). PyTorch 2.8, CUDA 12.8.

## Frames given to the models (`pod/frames.py`)

1. Candidates: every 2nd video frame (15 per second of the 30 fps video).
2. Motion gate: a candidate becomes a keyframe when the median optical flow (pyramidal Lucas–Kanade on up to 300
   Shi–Tomasi corners, 256 × 256 grey) since the last keyframe is ≥ 8 % of the image width, or 1 s has passed;
   never closer than 0.1 s. Standing still gives 1 keyframe per second; walking about 2; fast turns more.
3. Sharpness gate: a candidate whose variance of the Laplacian is below 0.5 × the median of the last 30
   candidates is motion-blurred and skipped (the next sharp one is taken); after 2 s without a keyframe the
   test is dropped so there are no long holes.
4. Each keyframe: turned upright, undistorted with the calibrated intrinsics (pinhole K), Log → display tone curve
   (per-video percentiles + mild S-curve, the same for every frame), resized to 518 × 518 (37 × 14 ViT patches).
5. Windows of 64 keyframes (48 where 64 does not fit), sharing 16 (12) with the next one, chained afterwards.

`frames_<take>.json` keeps, per keyframe, the video frame, its time, the flow and the sharpness, plus the counts
of candidates and of skipped blurry ones.

## Pod test on an RTX 3090 (24 GB), 2026-11-14

One window of real frames from `indoor_course` (earlier frame selection, ~5 keyframes/s; times per window do not
depend on it):

| model | window | peak GPU memory | s per window | model load |
|---|---|---|---|---|
| MapAnything | 64 | 20.6 GB | 24.3 | 58 s |
| Pi3X | 48 (64: out of memory) | 18.0 GB | 12.4 | 9 s |
| Depth Anything 3 | 48 (64: out of memory) | 19.7 GB | 17.5 | 11 s |
| VGGT | 64 | 14.1 GB | 21.8 | 55 s |

The full-run figures are in `derived/maps-v1/` (`run.json`, `metrics/`).

## Full run on the RTX 3090 (`derived/maps-v1`, 2026-11-14) — first look, preliminary

Both carried takes, 1573 keyframes (832 indoor, 741 outdoor; 2069–2077 processed with the window overlaps).
Run stage 45 min for the four models.

| model | wall time | model s / frame | load | peak GPU memory | GPU energy | J / frame |
|---|---|---|---|---|---|---|
| MapAnything | 14.2 min | 0.369 | 32 s | 21.4 GB | 159 kJ | 77 |
| Pi3X | 8.5 min | 0.201 | 21 s | 18.5 GB | 136 kJ | 65 |
| Depth Anything 3 | 10.3 min | 0.251 | 24 s | 19.7 GB | 170 kJ | 82 |
| VGGT | 11.8 min | 0.301 | 24 s | 14.7 GB | 199 kJ | 96 |

Energy = sum of the 1 s nvidia-smi power samples during each run (model loading included).

Window chaining consistency with the first, non-robust chaining (how well consecutive windows agree on their
shared keyframes after a scale + rigid fit on all of them; median over the window pairs; superseded by the robust
chaining below):

| model | indoor: position / rotation | outdoor: position / rotation |
|---|---|---|
| MapAnything | 0.44 m / 6.3° | 0.23 m / 1.0° |
| Pi3X | **0.03 m / 0.64°** | **0.05 m / 0.25°** |
| Depth Anything 3 | 0.07 m / 0.78° | 0.08 m / 0.41° |
| VGGT | 0.16 / 4.9° (model units) | 0.13 / 1.6° (model units) |

Batch by batch (indoor course, 2026-11-14; `window_consistency`: a frame's depth re-projected 3 keyframes ahead
with the window's own poses, median relative difference):

| model | inside a window: median / worst | robust join: shared keyframes that agree (median of 12) |
|---|---|---|
| MapAnything | 5.0 % / 19.4 % | 3 |
| Pi3X | **0.7 % / 1.5 %** | **12** |
| Depth Anything 3 | 1.4 % / 13.5 % | 10 |
| VGGT | 3.9 % / 28.0 % | 4 |

So the batches themselves are consistent (Pi3X especially); the bad global maps came from the joins. A single
keyframe that the two windows reconstruct differently (one turned 65° in one window and 135° in the other) used
to drag the whole join. The chaining is now robust: every shared keyframe proposes a transform, the one most
others agree with wins (camera within 5 % of the scene depth and 5°), and it is refitted on those frames only.
With that, Pi3X's worst join went from 3.8 m / 64° to 0.09 m / 2.2° (indoor). MapAnything and VGGT still
disagree between windows (few agreeing frames): their maps stay poor however they are joined. The joins accumulate
drift over a whole course; a global fit (pose graph, or the tags) is the next step.

### Global map on the PC: `make fuse` (`scripts/map_fuse.py`, no GPU, no pod rerun; all models and takes by default)

1. Robust joins (above).
2. **Loop closures**: keyframes of windows at least two apart and ≥ 20 s apart that look alike (thumbnail
   correlation, then ORB matches), verified geometrically — keyframe A's matched pixels lifted to 3D with A's
   depth give keyframe B's pose by PnP + RANSAC (≥ 60 inliers), the inliers' depth gives the scale; the best
   closure per window pair is kept.
3. **Pose graph**: one similarity per window (window 0 fixed), joins weighted by their agreeing keyframes,
   closures by their inliers, Huber loss. With calibrated intrinsics a similarity per submap is enough; VGGT-SLAM
   ([arXiv 2505.12549](https://arxiv.org/abs/2505.12549)) optimises on SL(4) because uncalibrated feed-forward
   reconstructions carry a projective ambiguity; VGGT-Long ([arXiv 2507.16443](https://arxiv.org/abs/2507.16443))
   chunks, aligns overlaps and closes loops much like this.
3b. **ICP refinement**: every join and closure is checked densely — point-to-plane ICP between the two
   windows' points as the graph places them (voxel-downsampled, normals from 16 neighbours, Huber weights,
   shrinking match distance); a good fit (≥ 30 % overlap, RMS ≤ 2.5 voxels, correction ≤ 10 voxels / 10°) replaces
   the edge, and the graph is solved again. Pi3X indoor: 45 of 69 edges refined (median correction 0.11 m /
   1.1°); closures then disagree by 0.07 m / 0.7° (0.11 m / 1.1° without ICP); the doubled corridor walls merge.
   `--no-icp` skips it.
4. **Weighted TSDF**: sparse voxels (2 % of the median depth, ~3.5 cm indoors), truncation 4 voxels, each sample
   weighted by (confidence / frame median) × (1 m / depth)²: near, confident data dominate; overlaps average instead
   of piling up; voxels seen by ≥ 3 frames, on the zero crossing. The black border the undistortion left in every
   keyframe (the "embossed" frame outlines) is masked out (valid-pixel mask from the calibration, eroded 3 px);
   the viewer masks it too.

Outputs per model and take in `derived/maps-v1/fused/<model>/<take>/`: `poses.npz` (per window similarity, per
keyframe camera-to-world + depth scale: the optimised oracle poses for the analyses), `graph.json` (closures,
residuals before / after, settings), `fused.npz` / `fused.ply`. In the viewer: *Show → fused map (TSDF)*, and
*Chaining → pose graph (map_fuse)* for the whole-take / two-window views.

First result, Pi3X indoor course (2026-11-14): 47 verified loop closures; they disagreed by 1.70 m / 8.4° with the
plain chaining, 0.11 m / 1.1° after the pose graph. From above, the plain chaining shows the room several times,
rotated (ghost walls); after the pose graph the room is one rectangle; the TSDF gives thin walls
(`docs/figures/pi3x_indoor_topdown.png`). 116 s on 2 CPU cores.

In the viewer, **View → Fuse this model + take (TSDF)** runs the same thing for the current model and take
(2–3 min, progress in the panel) and shows the result. Point size goes up to 0.3 and the pixel stride down to 1
for far, sparse scenes.

### Static takes: monocular depth on the CPU (`make static-depth TAKE=static_pass1`)

The camera does not move in the static takes, so no multi-view reconstruction is needed: `scripts/static_depth.py`
runs a monocular depth model on the CPU (default Depth Anything V2 Metric Indoor Small via `transformers`; any
`depth-estimation` model id) on a few frames per part (S0 / S1 / S2), on the same undistorted, tone-mapped frames
as the pod, and grounds each depth map on the AprilTags: each tag located by solvePnP (IPPE_SQUARE, printed size),
a 5 × 5 grid of true depths per tag, fitted by a scale (or scale + offset, `--fit affine`); per tag the error with
all tags and leave-one-tag-out. Frames without tags (S0, the empty scene) take the take's median scale (the camera
is still). Tags found in static_pass1 S1: 13 (1.29 m), 20 (1.83 m), 21 (1.97 m), 15 (3.75 m) — stable to 1 cm
between frames. Outputs: `derived/static-depth-v1/<take>/` (`.npz` metric depth, `.jpg`, `.ply`, `summary.json`).
`TAKE=all` = both passes + the pilot. Needs the static takes downloaded (`make data DATA_TAKES="static_pass1
static_pass2 pilot_chairs"`) and `uv sync --group depth` (PyTorch CPU + transformers).

Preliminary: these are self-consistency figures, not accuracy. The chained paths of the models still differ by
1–2 m RMS over a whole course after a similarity fit (drift of the chaining). Accuracy needs the AprilTag check
(floor tags 10 / 12 / 17 for scale, the others held out) — still to do.

## Viewing the maps: `scripts/map_viz.py`

```bash
uv run --group viz python scripts/map_viz.py     # opens http://localhost:8080 ; --model / --take / --port
```

(`make viz` on Linux.) The first time per take it re-makes the keyframe images from the course video (exactly the
pod's frames, for the colours). In the browser panel:

- **View → Show**: *whole take (chained)*; *one window* — one batch exactly as the model returned it, in its own
  frame, to see whether the model is consistent inside a batch; *two windows (overlap)* — window N (red) and N+1
  (blue) as the chaining placed them, to see whether a join is what breaks the map. The panel gives each window's
  self-consistency and, for a join, how many shared keyframes agree; *◀ previous / next ▶* steps through the
  batches, and the info box lists the worst joins. *Chaining*: camera poses (default) or dense points.
- **Data**: take and model; the info box gives the windows, keyframes, units, camera path length, the chaining
  consistency above and the pod's numbers (time, peak memory, frames/s).
- **Points**: keyframe step, pixel stride (point budget), drop the lowest-confidence pixels (per frame percentile),
  max depth, remove flying pixels at depth edges, colour by image / depth / height / window, point size. Heavy
  changes apply with **Apply**.
- **Cameras**: the camera path, frustums every N keyframes, with their images.
- **Compare models**: the paths of all models, each aligned to the current one (similarity fit on the shared
  keyframes), with their RMS difference and scale.
- **Measure**: *Start measuring*, then click points in the scene (a click snaps to the nearest map point along the
  clicked ray). Two-point mode or polyline (running total). Each measurement is drawn with its length and listed
  with its vertical and horizontal parts; *Undo last point*, *Clear all*; every measurement is appended to
  `derived/maps-v1/measurements.csv`. Units are metres for the metric models (VGGT: arbitrary). Quick scale check:
  measure the edge of a printed AprilTag (40 / 55 / 90 / 150 mm).

Memory: one model and take in full (~1 GB for the 48-frame models); the other models' paths are loaded without
their depth maps.

## Running metrics (kept unfiltered)

Per model and take, `run.json`:

- per window: `frames`, `seconds` (wall), split into `load_s` (reading the JPEGs), `model_s` (inference + copies to
  the host), `save_s` (compressing the npz); `fps`, `s_per_frame`; `peak_alloc_gb` and `peak_reserved_gb` (PyTorch
  peak allocated / reserved GPU memory in that window);
- per run: `model_load_s`, totals of the above, `fps_model`, overall peaks, GPU name and memory, torch / CUDA /
  cuDNN / Python versions, command line, checkpoint, repo commit, status (`running` while it runs, so a crash keeps
  what was measured).

In `metrics/`:

| file | content |
|---|---|
| `gpu_run_<model>_<take>.csv` | nvidia-smi every second during the run: utilisation (GPU, memory), memory used / total, power (W), temperature, SM clock → energy per frame = ∫ power dt / frames |
| `gpu_check_<model>_w<window>.csv` | the same during each check attempt (including the ones that ran out of memory) |
| `check/<model>_w<window>.json` | `run.json` of the check that passed |
| `stages.csv` | stage, start, end (Unix s), result |
| `gpu.csv`, `nvidia-smi-q.txt` | GPU model, driver, memory, power limit, PCIe |
| `lscpu.txt`, `memory.txt`, `disk.txt`, `nproc.txt`, `uname.txt` | the host |
| `freeze_<model>.txt` | every package of each model environment (reproducibility) |
| `pins.env`, `ravidata_commit.txt`, `runpod_env.txt` | what code ran, on what kind of pod |

Logs of every setup, check attempt and run are in `logs/`.

## Caveats to state

- One GPU type (RTX 3090); windows differ per model (memory), so per-window times are not directly comparable;
  compare per frame (`s_per_frame`, energy per frame).
- `model_s` includes the copies of the outputs to the host; VGGT loads its images inside its own preprocessing,
  so its `load_s` is ~0 and the loading is counted in `model_s`.
- Peak memory is PyTorch's allocator view; nvidia-smi's `memory.used` (in the GPU csv) also counts the CUDA
  context and the cache.
