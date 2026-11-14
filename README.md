# RAVIDATA — RAdar, VIsual and inertial DATAset

A **60 GHz FMCW radar** (Possumic RS6130 on the MRS6130-P1812 board, 1 TX / 3 RX) recorded together with a
**phone camera** and the **phone's IMU**, for obstacle detection with a radar on a moving, hand-held platform. This
repository holds the tools to record, align and analyse the data, and to rebuild every result.

- **Code:** <https://github.com/zRafaF/ravidata> (PolyForm Noncommercial 1.0.0)
- **Data:** <https://huggingface.co/datasets/DoninhaD/ravidata> (CC BY-NC 4.0)

## The dataset

| | |
|---|---|
| radar | 4 cm range bins up to 5.12 m; `tcc-1d-40mm` (range only, 33 Hz), `tcc-2d-40mm-12fps` / `tcc-2d-40mm` (range-Doppler, 16 bins, 12.5 / 20 Hz); raw complex cubes, every frame stamped on the host clock |
| camera | Samsung SM-S936W, 1440 × 1440, 30 fps, 10-bit Log; calibrated intrinsics (checkerboard clip included) |
| IMU | Sensor Logger, 100 Hz: gyroscope, accelerometer, gravity, orientation |
| takes | `static_pass1/2` radar still (empty room → objects → a person and a broom with AprilTags); `indoor_course` / `outdoor_course` radar carried along a course (4.5 / 5.2 min); `pilot_chairs` three chairs |
| sync | recorder beeps in the video sound, gyro vs video motion, radar vs video motion; the offsets are in each `take.json` |
| privacy | faces blurred; sound reduced to the recorder beeps (pilot and calibration muted) |

## Workflow

| step | where | command |
|---|---|---|
| record | PC + radar + phone | `make rec` |
| build the public dataset (owners) | PC, CPU | `make dataset`, `make dataset-upload` |
| **maps (the only GPU step)** | rented GPU pod | `make pod` |
| analysis and scoring | PC, CPU | `make maps`, then the analysis tools |

The pod does nothing but turn video into maps: it downloads only the two carried videos and the calibration, runs
the models, uploads the results to the dataset and **stops itself**. Everything that only needs a CPU happens on
your own machine.

## Mapping methods

The radar is carried, so each radar frame needs to know where the radar was and what was in front of it. The
maps give that: a camera pose for every frame and a dense depth map, from which the radar's view is rebuilt.
Four **feed-forward 3D reconstruction models** are compared (one network pass per window of frames, no
per-scene optimisation):

| model | checkpoint | what we give it | scale of the output | weights licence |
|---|---|---|---|---|
| [MapAnything](https://arxiv.org/abs/2509.13414) (Meta, 2025) | `facebook/map-anything` | frames + calibrated intrinsics | metric | CC BY-NC 4.0 (`facebook/map-anything-apache`: Apache 2.0) |
| [Pi3X](https://arxiv.org/abs/2507.13347) (π³, extended) | `yyfz233/Pi3X` | frames + calibrated intrinsics | approximately metric | CC BY-NC 4.0 |
| [Depth Anything 3](https://arxiv.org/abs/2511.10647) (ByteDance, 2025) | `depth-anything/DA3NESTED-GIANT-LARGE-1.1` | frames (+ intrinsics if the API accepts them) | metric | CC BY-NC 4.0 |
| [VGGT](https://arxiv.org/abs/2503.11651) (Meta, CVPR 2025), baseline | `facebook/VGGT-1B` | frames only | up to an unknown scale | non-commercial |

The same frames go to every model (`pod/frames.py`). They are **keyframes chosen by motion**: a frame is kept
when the view has moved enough since the last kept one (median optical flow of tracked corners ≥ 8 % of the
image width), at least one per second when standing still, so near-identical frames are dropped; a
motion-blurred candidate (sharpness, the variance of the Laplacian, below half the recent median) is skipped for
the next sharp one. Walking gives about 2 keyframes per second. Each keyframe is turned upright, undistorted with
the calibrated intrinsics, tone-mapped from Log to normal contrast and resized to 518 × 518. The takes are cut
into windows of 64 keyframes that share 16 with the next one, so the windows can be chained afterwards; no model
ever sees a whole take at once.

The AprilTags are **not** given to the models: they are kept as independent ground truth. On the PC, the floor
tags (10, 12, 17, known size) give each window's metric scale and pose check; the other tags are held out to
score it. Each model runs in its own environment (`pod/envs.sh`, repositories pinned in `pod/pins.env`).

Per model and take the pod writes `win_NNN.npz` (frame numbers, camera-to-world poses in OpenCV axes, depth,
confidence, intrinsics) and `run.json`. Every running metric is kept, unfiltered, for the comparison: per window
the frames, the time split into image loading / model / saving, frames per second and peak GPU memory (allocated
and reserved); per run the model loading time, totals, checkpoint, repository commit and versions. Next to them,
in `metrics/`: GPU samples every second during every check and run (utilisation, memory, power, temperature,
clock), the check attempts (including the window sizes that ran out of memory), stage times, the machine
(`nvidia-smi -q`, CPU, RAM, disk) and the installed packages of every model environment.

The comparison, on CPU: AprilTag positions and sizes against their known size, start/end loop closure, camera
rotation against the phone gyroscope, and range to the tags against the radar. The scoring tool is being
written and will be added here as a `make` target.

## Detection baselines

`src/ravidata/processing/detectors.py` holds the public baselines: a fixed threshold and CA / OS-CFAR along range,
both reporting the nearest return; `make rec` shows them live. A range-Doppler CFAR and the occupancy grid
(floating point and CORDIC fixed point) will be added with the thesis, with a comparison table on RAVIDATA.

## Using the dataset (any PC)

Needs `make`, `git` and internet; [uv](https://docs.astral.sh/uv/) and Python are installed automatically. Nothing
here uploads anything or needs a token (the dataset is public).

```bash
git clone https://github.com/zRafaF/ravidata && cd ravidata
make help        # lists every command
make test        # optional: installs the Python env and runs the unit tests (checks the install works)
make data        # downloads the dataset -> data/ravidata   (DATA_TAKES="indoor_course" for one take)
```

## AprilTags

tag36h11; size = edge of the black square. Only these IDs were placed in the recordings (the other IDs of
`apriltags_A4.pdf` were printed but not used); the list is in `docs/protocols/targets/tags_used_2026-11-13.json`
and in the dataset's `apriltags.json` (`used`).

| IDs | on | size |
|---|---|---|
| 0, 1 | the person: 0 chest, 1 back | 150 mm |
| 10, 12, 17 | the floor | 90 mm |
| 11, 13, 15 | the chairs | 90 mm |
| 20, 21, 50, 51 | the broom | 55 mm (20, 21), 40 mm (50, 51) |

## GPU pod (maps only)

Target: **RunPod, one RTX 3090 (24 GB)**, any PyTorch / CUDA 12.8 image, **80 GB** volume at `/workspace` (the
environments, the model weights and the caches live there, so they survive a pod stop). Put a Hugging Face token
with write access in the pod's environment variables as `HF_TOKEN` (a RunPod secret), then in the pod's terminal:

```bash
apt-get update && apt-get install -y make git tmux
cd /workspace && git clone https://github.com/zRafaF/ravidata && cd ravidata
tmux new -s pod
make pod-test    # ~30 min: installs every model, runs each on one window of real frames, prints a time estimate
make pod         # the real run; reuses what pod-test installed and downloaded
```

`make pod-test` uploads nothing and never stops the pod. It ends with one line per model: pass or fail, the window
that fits, the peak GPU memory, and the expected time of the full run. Fix or drop a failing model
(`MODELS="vggt pi3x"`) before `make pod`.

`make pod` runs, in order (each also alone as `make pod-<stage>`; finished work is skipped on a re-run):

1. `setup` uv, one environment per model;
2. `data` the two carried videos and the calibration from Hugging Face;
3. `frames` the 518 × 518 frames;
4. `check` each model on one window of real frames; if 64 frames do not fit in the GPU memory it retries
   with 48, 32 and 16 and keeps the largest that fits; a model that fails is left out (log in `work/logs/`);
5. `run` every model that passed, on both takes;
6. `upload` the results to `derived/maps-v1/` in the dataset (also `work/maps-v1.tar`);
7. `stop` the pod stops itself (`runpodctl stop pod`) once the results are on Hugging Face, or at least on the
   `/workspace` volume, which survives a stop; `STOP=0` keeps it running.

`make pod` always ends with upload + stop, also when a stage fails (it uploads what exists, with the logs), so it
can run unattended; `make pod-test` prints whether the auto-stop is available on the pod. Expected on a 3090 (about
1500–3000 keyframes per model; `make pod-test` gives the real figures): setup 15–25 min, frames ~5 min, check ~10 min, the
four models 45–90 min, upload ~10 min (roughly 10 GB of maps): about 1.5–2.5 h in all. Other settings: `MODELS="vggt pi3x"`, `TAKES=indoor_course`, `WINDOW`,
`OVERLAP`, `STEP`, `RES`, `TAG` (see `make help`).

Back on your PC:

```bash
make maps        # derived/maps-v1 -> data/ravidata/derived/maps-v1/<model>/<take>/
```

## Recording

`make rec` opens the recorder: pick the SPI port, **Check link**, then the **Simple** tab runs a guided take (scene,
radar configs recorded back to back, a checklist, what to do now, a live check of the sync swipe, a verdict per
recording, a folder per take). Hardware notes:

- board switches for SPI: MODE SEL on, DIP 1, 2, 5, 6, 7, 9 on;
- the CH347 USB bridge needs WCH's driver (CH341PAR from wch-ic.com) on Windows;
- printable targets: `make targets` (AprilTag tag36h11 and a checkerboard; print at 100 %).

`make calib VIDEO=...` computes the camera intrinsics from a checkerboard clip; `make align` / `make pack` put the
phone video, the IMU and the radar on one clock.

## Licence

- Code: [PolyForm Noncommercial 1.0.0](LICENSE.md): free for research, teaching and personal use; **no commercial
  use**.
- Data: CC BY-NC 4.0 (see the dataset card).
- The map models keep their own licences (all the checkpoints used here are non-commercial).

## Citation

To be added with the thesis (UTFPR, 2026).
