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
