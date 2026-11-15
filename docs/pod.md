# GPU pod runbook (maps only)

The pod does one thing: turn the two carried videos into 3D maps with four models, upload the results and every
running metric to the dataset, and stop itself. Scoring and analysis run on a normal PC (CPU).

## 1. Create the pod (RunPod)

| setting | value |
|---|---|
| GPU | 1 × RTX 3090 (24 GB) — every model fits (table in [maps.md](maps.md)) |
| image | any PyTorch / CUDA 12.8 template |
| volume | **80 GB at `/workspace`** (environments, model weights, caches and results live there and survive a stop) |
| environment | `HF_TOKEN` = a Hugging Face token with **write** access (best as a RunPod secret) |

`runpodctl` and `RUNPOD_POD_ID` come with every RunPod pod; they let the pod stop itself.

## 2. Commands

```bash
apt-get update && apt-get install -y make git tmux
cd /workspace && git clone https://github.com/zRafaF/ravidata && cd ravidata
tmux new -s pod
export HF_TOKEN=hf_...          # only if it is not set as a RunPod secret (do it inside tmux)
make pod-test                   # ~30 min: install + one window per model; prints pass/fail, memory, time estimate
make pod                        # the real run; ends with upload + stop
```

Before leaving it alone, check (inside the tmux window that runs `make pod`):

```bash
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN set (${HF_TOKEN:0:3}...)" || echo "HF_TOKEN missing"
uv run hf auth whoami                         # your Hugging Face user = the token works
command -v runpodctl && echo "pod: $RUNPOD_POD_ID"   # a path and an id = the pod can stop itself
```

Never paste the token into chats or commit it. After the run, delete the pod (the token stays in its shell history).

## 3. What `make pod` does

| stage | what | notes |
|---|---|---|
| `setup` | uv, one environment per model (`pod/envs.sh`, repos pinned in `pod/pins.env`) | skipped for a model already installed |
| `data` | the two course videos + calibration from Hugging Face | radar and IMU stay off the pod |
| `frames` | keyframes by motion + sharpness, upright, undistorted, tone-mapped, 518 × 518 (`pod/frames.py`) | redone automatically when `frames.py` / `STEP` / `RES` change |
| `check` | each model on one window of real frames; 64 → 48 → 32 → 16 on out-of-memory | failing models are left out |
| `run` | every model that passed, on both takes, in overlapping windows | per-window results written as they come |
| `upload` | results + logs + metrics → `derived/maps-v1/` in the dataset | without `HF_TOKEN`: a tar in `work/` |
| `stop` | `runpodctl stop pod` | only once the results are on HF or on the `/workspace` volume |

`make pod` always ends with upload + stop, also when a stage fails (it uploads what exists, with the logs), so it
can run unattended. Each stage also runs alone: `make pod-setup`, `pod-data`, `pod-frames`, `pod-check`,
`pod-run`, `pod-upload`, `pod-stop`. Finished work is skipped on a re-run. Settings: `MODELS="vggt pi3x"`,
`TAKES=indoor_course`, `WINDOW`, `OVERLAP`, `STEP`, `RES`, `TAG`, `STOP=0` (see `make help`).

Time on a 3090 (2026-11-14): setup 15–25 min the first time, frames ~5 min, check ~10 min, the four models about
45–60 min with the motion keyframes (~2 per second of video), upload 5–10 min.

## 4. Monitoring

Open a second tmux window (`Ctrl-b c`) inside `/workspace/ravidata`:

| command | shows |
|---|---|
| `make pod-status` | current stage, stage times, windows done / total per model and take, time left, GPU now, last log lines |
| `make pod-watch` | the same every 30 s; `Ctrl-C` quits the watch only, `make pod` keeps running; `make pod-watch LOG_LINES=40` for more log |
| `tail -f work/logs/pod.log` | the full log as it grows (`Ctrl-C` to quit) |
| `less +G work/logs/pod.log` | the whole log, scrollable (`q` to quit) |
| `watch -n 2 nvidia-smi` | GPU use and memory, live |
| `cat work/metrics/stages.csv` | start / end time of each stage (Unix seconds) and its result |
| `ls work/logs/` | one log per setup, check attempt and run |

## 5. tmux cheat-sheet

All shortcuts start with the prefix **`Ctrl-b`**, released before the next key.

| action | keys / command |
|---|---|
| start a named session | `tmux new -s pod` |
| leave it running (detach) | `Ctrl-b d` |
| come back (attach) | `tmux attach -t pod` (or `tmux a`) |
| list sessions | `tmux ls` |
| scroll back | `Ctrl-b [` then arrows / `PgUp` / `PgDn`; `q` to leave |
| search back while scrolling | `?` then text, `n` for the next match |
| scroll with the mouse wheel | `tmux set -g mouse on` (once per session) |
| new window / next / previous / by number | `Ctrl-b c` / `Ctrl-b n` / `Ctrl-b p` / `Ctrl-b 0..9` |
| split side by side / top-bottom | `Ctrl-b %` / `Ctrl-b "` ; move between panes with `Ctrl-b` + arrow |
| close a pane or window | `exit` (or `Ctrl-b x`, then `y`) |
| kill the whole session | `tmux kill-session -t pod` (this stops a running `make pod`) |

Closing the browser terminal or losing the connection does not stop a job inside tmux; detach, reconnect, attach.
A variable exported in one tmux window is not seen by the others: export `HF_TOKEN` in the window that runs
`make pod`.

## 6. Results

On Hugging Face, `derived/maps-v1/`:

```
<model>/<take>/win_NNN.npz   idx, c2w (n,4,4, OpenCV axes, the model's frame and scale), depth (n,H,W) float16,
                             conf (n,H,W) float16, K (n,3,3) — one file per window
<model>/<take>/run.json      checkpoint, repo commit, settings, timing and memory per window and in total
frames_<take>.json           the keyframes used (video frame, time, optical flow, sharpness), K, tone curve
check_ok.txt                 model and window size used
logs/                        every log of the pod
metrics/                     see maps.md "Running metrics"
```

On the PC: `make maps` downloads them to `data/ravidata/derived/maps-v1/`. `fused/` (the global maps of
`make fuse`) is made on the PC and uploaded next to them; the pod does not write it.

## 7. Troubleshooting (what happened on the first pod)

| symptom | cause | fix (already in the repo) |
|---|---|---|
| `No module named 'addict'` (DA3) | the model repo does not declare it | `envs.sh` installs it, and any other module an import check reports missing |
| `operator torchvision::nms does not exist` | xformers pulled a newer torch than torchvision | xformers dropped; setup checks that torch + torchvision import together |
| `No module named 'mapanything.models'; 'mapanything' is not a package` | the runner script had the package's name and shadowed it | runners are `pod/runners/run_<model>.py` |
| a model "does not fit in the GPU memory" | window too large for 24 GB | expected: the check falls back to a smaller window |
| `make: *** No rule to make target` | not in the repo folder | `cd /workspace/ravidata` |
| cannot scroll in tmux | tmux has its own scrollback | `Ctrl-b [` (or `tmux set -g mouse on`) |
