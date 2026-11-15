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
| global maps, still-take depth | PC, CPU | `make maps`, `make fuse`, `make static-depth TAKE=all` (results are on the dataset) |
| radar analyses | PC, CPU | `make data-analysis`, `make analysis-quick`, `make analysis`, `make plots`, `make report` |

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

On the PC, `make fuse` joins the windows into one map per model and take (loop closures, pose graph, ICP, TSDF);
the radar ground truth uses its trajectory. The metric scale of every window comes from the floor tags (table
`map_scale` of the analyses). `make viz` compares the models' camera paths and measures distances in the maps. A
systematic accuracy check of the maps (held-out tags, radar as altimeter) has been done for the thesis but is not
yet a `make` target. Details: [docs/maps.md](docs/maps.md).

## Detection baselines

`src/ravidata/processing/detectors.py` holds the recording app's baselines: a fixed threshold and CA / OS-CFAR
along range, both reporting the nearest return; `make rec` shows them live.

`analysis/` scores these and more (threshold, CA / GO / SO / OS-CFAR, range-Doppler CFAR, CFAR + DBSCAN, a
log-odds occupancy grid alone / moved with the IMU / moved with the map trajectory, a range-azimuth grid with
Doppler layers) against ground truth from the AprilTags and the feed-forward maps, tunes every family with a
multi-objective search, measures how well one setting generalises across scenes and characterises the sensor.
Steps write tables; figures are drawn from the tables. See
[docs/analysis.md](docs/analysis.md).

## Results

| folder | status |
|---|---|
| `results/analysis-v2` | **current.** Indoor_course time alignment corrected, range / angle calibration from the walking person, balanced arc metric and its constant-answer floor. Run so far: tags, extrinsic, truth, sensor, eval, floor, and the arc-view tuning (pooled, 80 trials, five families). Not yet: ablation, plots, story, report, the plane objective, the strict protocol. |
| `results/analysis-v1` | superseded, kept as a record (its `FINDINGS.md` and `REPORT.md` are marked so). |

What changed and why: [CHANGELOG.md](CHANGELOG.md). Each folder's `runs.jsonl` lists every step that wrote to it,
with its settings and duration.

## Reproducing the results

CPU only; about 1 h for the steps below on a 4-core PC (`WORKERS=4`), plus the downloads (~2 GB).

```bash
git clone https://github.com/zRafaF/ravidata && cd ravidata
make install          # uv + the Python environment (uv.lock)
make test             # unit tests (no data needed)
make data-analysis    # recordings, Pi3X maps, fused poses, still-take depth -> data/ravidata
make analysis-quick   # every step on a few seconds per recording -> results/analysis-v2-quick (minutes)
```

Then the full run, with the settings of `results/analysis-v2` (writes into it; set `RAVIDATA_RUN=<name>` to write
to `results/<name>` instead and compare):

```bash
make an-tags an-extrinsic an-truth an-sensor an-eval
make an-opt an-eval-tuned OBJECTIVE=arc TRIALS=80 FAMILIES="thr ca_cfar rd_cfar grid grid_doppler"
make an-floor plots report
```

The Optuna studies in `results/<run>/optuna/` are resumed, not restarted: with the published studies present,
`an-opt` only rebuilds the tables. To redo the search from scratch, use a new `RAVIDATA_RUN`. A re-run of `an-eval`
also scores `grid_doppler`, which the published default tables lack (the family was added after that step ran).
`make analysis` runs
every step with the defaults (all families, both objectives, 80 trials: several hours). The analysis needs dataset
**v1.1** or newer (it carries the indoor_course time-alignment fix in `take.json`) and stops with a message on an
older local copy (`make data` updates it). The maps, fused poses and still-take depth must be the published ones
(`make fuse` on another machine does not give bit-identical poses).

## Using the dataset (any PC)

Needs `make`, `git` and internet; [uv](https://docs.astral.sh/uv/) and Python are installed automatically. Nothing
here uploads anything or needs a token (the dataset is public).

```bash
git clone https://github.com/zRafaF/ravidata && cd ravidata
make help        # lists every command
make test        # optional: installs the Python env and runs the unit tests (checks the install works)
make data        # the recordings -> data/ravidata   (DATA_TAKES="indoor_course" for one take)
make data-all    # everything, the derived maps included (~4.5 GB)
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

The maps are made once on a rented GPU (an RTX 3090, 24 GB, is enough); everything else runs on a PC.

```bash
apt-get update && apt-get install -y make git tmux
cd /workspace && git clone https://github.com/zRafaF/ravidata && cd ravidata
tmux new -s pod
make pod-test    # install + one window per model: pass/fail, memory, time estimate
make pod         # the real run: frames, maps, upload to derived/maps-v1/, then the pod stops itself
make pod-status  # progress, from a second tmux window
```

Full runbook (pod settings, tokens, stages, monitoring, a tmux cheat-sheet, troubleshooting):
[docs/pod.md](docs/pod.md). Models, frame selection, the measured memory / time and the running metrics:
[docs/maps.md](docs/maps.md). Back on the PC: `make maps`, then the 3D viewer
`uv run --group viz python scripts/map_viz.py`.

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
