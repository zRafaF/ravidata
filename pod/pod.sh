#!/usr/bin/env bash
# The map bake-off on a GPU pod, from a fresh clone (make pod = bash pod/pod.sh all). The pod ONLY makes the maps;
# scoring and analysis run on a normal PC (make maps, then the local tools). Stages run alone too:
#   setup   uv, the project env, one venv per model (pod/envs.sh)
#   data    download the videos of the carried takes + the calibration from the Hugging Face dataset
#   frames  video -> keyframes chosen by motion (optical flow), upright, undistorted, tone-mapped, resized
#           (pod/frames.py; candidates every STEP-th video frame)
#   check   every model on one window of real frames; picks the largest window that fits in the GPU memory
#           (WINDOW, then 48 / 32 / 16 on "out of memory"); only the models that pass run in 'run'
#   run     every passing model on every take (windows overlap; WINDOW=0: all frames in one pass)
#   upload  results -> HF (derived/maps-<TAG>/) if HF_TOKEN is set; always a tar in work/
#   stop    STOP=1 (default) and the upload succeeded on RunPod: stop this pod so the GPU is not billed idle
#   test    (make pod-test) setup + the first take + its frames + check, then a time estimate for 'all';
#           uploads nothing, never stops the pod. 'all' afterwards reuses the environments and the weights.
# Settings (environment / make variables): HF_REPO TAKES MODELS STEP RES WINDOW OVERLAP TAG STOP
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
# caches inside the repo (on RunPod: the /workspace volume): kept across a pod stop, and uv hard-links one
# PyTorch into the four model environments instead of four copies
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/work/cache/uv}" HF_HUB_CACHE="${HF_HUB_CACHE:-$PWD/work/cache/hub}"
FULL_VIDEO_FRAMES=17532      # indoor_course 8162 + outdoor_course 9370 video frames (for the time estimate)
HF_REPO="${HF_REPO:-DoninhaD/ravidata}"
TAKES="${TAKES:-indoor_course outdoor_course}"
MODELS="${MODELS:-mapanything pi3x da3 vggt}"
STEP="${STEP:-2}"; RES="${RES:-518}"; WINDOW="${WINDOW:-64}"; OVERLAP="${OVERLAP:-16}"; TAG="${TAG:-v1}"
STOP="${STOP:-1}"
WORK=work; DATA=data/ravidata
mkdir -p "$WORK/logs"

py_of () { case "$1" in pi3x) echo pod/third_party/pi3/.venv/bin/python ;; *) echo "pod/third_party/$1/.venv/bin/python" ;; esac; }
repo_of () { case "$1" in pi3x) echo pod/third_party/pi3 ;; *) echo "pod/third_party/$1" ;; esac; }
peak_of () { grep -o '"peak_vram_gb": [0-9.]*' "$1/run.json" 2>/dev/null | awk '{print $2" GB"}'; }

stage_setup () {
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
    uv sync
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
    for m in $MODELS; do
        if [ -f "$(repo_of $m)/.venv/.ravidata_ok" ]; then echo "setup: $m already installed"; continue; fi
        bash pod/envs.sh "$m" 2>&1 | tee "$WORK/logs/setup_$m.log" \
            || echo "setup: $m FAILED (see $WORK/logs/setup_$m.log); the other models go on"
    done
}
stage_data () {     # only what the maps need: the videos + the calibration (radar and IMU stay off the pod)
    inc=(--include "calibration/camera.calib.json" --include "rig.json" --include "apriltags.json")
    for t in $TAKES; do inc+=(--include "takes/$t/video.mp4" --include "takes/$t/take.json"); done
    uv run hf download "$HF_REPO" --repo-type dataset --local-dir "$DATA" "${inc[@]}"
}
stage_frames () {
    for t in $TAKES; do
        [ -f "$WORK/frames/$t/frames.json" ] && { echo "frames: $t already done"; continue; }
        uv run python pod/frames.py --data "$DATA" --take "$t" --out "$WORK/frames/$t" --step "$STEP" --res "$RES"
    done
}
stage_check () {      # FRAMES_DIR: where the frames of the first take are (the test uses a short clip)
    t1=$(echo $TAKES | awk '{print $1}'); : > "$WORK/check_ok.txt"; fdir="${FRAMES_DIR:-$WORK/frames}/$t1"
    if [ "$WINDOW" -eq 0 ]; then tries=0; else tries="$WINDOW"; for w in 48 32 16; do [ "$w" -lt "$WINDOW" ] && tries="$tries $w"; done; fi
    for m in $MODELS; do
        ok=""
        if [ ! -f "$(repo_of $m)/.venv/.ravidata_ok" ]; then
            echo "check $m: SKIP (setup failed, see $WORK/logs/setup_$m.log)"; continue
        fi
        for w in $tries; do
            log="$WORK/logs/check_${m}_w$w.log"; rm -rf "$WORK/check/$m"
            if "$(py_of $m)" "pod/runners/$m.py" --frames "$fdir" --out "$WORK/check/$m" --window 0 \
                  --max-frames "$w" --repo "$(repo_of $m)" > "$log" 2>&1; then
                ok="$w"; break
            elif grep -qiE "out of memory|OutOfMemoryError" "$log"; then
                echo "check $m: window $w does not fit in the GPU memory, trying a smaller one"
            else
                echo "check $m: FAIL (see $log)"; tail -5 "$log"; break
            fi
        done
        if [ -n "$ok" ]; then
            echo "$m $ok" >> "$WORK/check_ok.txt"; echo "check $m: PASS (window $ok, peak $(peak_of "$WORK/check/$m"))"
        fi
    done
    [ -s "$WORK/check_ok.txt" ] || { echo "no model passed the check"; exit 1; }
}
stage_run () {
    [ -s "$WORK/check_ok.txt" ] || for m in $MODELS; do echo "$m $WINDOW"; done > "$WORK/check_ok.txt"
    while read -r m w; do
        ov=$(( w / 4 < OVERLAP ? w / 4 : OVERLAP ))
        for t in $TAKES; do
            out="$WORK/out/$m/$t"
            [ -f "$out/run.json" ] && { echo "run: $m/$t already done"; continue; }
            echo "run: $m on $t (window $w, overlap $ov)"
            "$(py_of $m)" "pod/runners/$m.py" --frames "$WORK/frames/$t" --out "$out" --window "$w" \
                --overlap "$ov" --repo "$(repo_of $m)" < /dev/null 2>&1 | tee "$WORK/logs/run_${m}_$t.log" \
                || echo "run: $m on $t FAILED (log kept, continuing)"
        done
    done < "$WORK/check_ok.txt"
}
stage_upload () {
    rm -f "$WORK/uploaded_$TAG"
    mkdir -p "$WORK/out"; cp -r "$WORK/logs" "$WORK/out/"; cp "$WORK/check_ok.txt" "$WORK/out/" 2>/dev/null || true
    for t in $TAKES; do cp "$WORK/frames/$t/frames.json" "$WORK/out/frames_$t.json" 2>/dev/null || true; done
    echo "results: $WORK/out ($(du -sh "$WORK/out" | cut -f1))"
    if [ -n "${HF_TOKEN:-}" ]; then
        uv run hf upload "$HF_REPO" "$WORK/out" "derived/maps-$TAG" --repo-type dataset \
            --commit-message "map bake-off $TAG ($(echo $MODELS))"
        touch "$WORK/uploaded_$TAG"
        echo "uploaded: https://huggingface.co/datasets/$HF_REPO/tree/main/derived/maps-$TAG"
    else
        tar -cf "$WORK/maps-$TAG.tar" -C "$WORK" out
        echo "HF_TOKEN not set: not uploaded (download $WORK/maps-$TAG.tar, or set HF_TOKEN and run: make pod-upload)"
    fi
}
stage_test () {
    t1=$(echo $TAKES | awk '{print $1}')
    echo "=== test: setup ==="; stage_setup
    echo "=== test: data ($t1) ==="; TAKES="$t1" stage_data
    echo "=== test: frames ($t1, kept for the real run) ==="; TAKES="$t1" stage_frames
    echo "=== test: check ==="; stage_check || true
    echo "=== test: summary ==="
    [ -s "$WORK/check_ok.txt" ] || { echo "no model passed: see $WORK/logs/check_*.log"; return 1; }
    read -r n1 total < <(uv run python -c "import json,sys; m=json.load(open(sys.argv[1])); n=len(m['frames']); \
print(n, round(n * $FULL_VIDEO_FRAMES / m.get('n_video_frames', $FULL_VIDEO_FRAMES)))" "$WORK/frames/$t1/frames.json")
    echo "keyframes: $n1 in $t1, ~$total expected over both takes"
    while read -r m w; do
        ov=$(( w / 4 < OVERLAP ? w / 4 : OVERLAP ))
        uv run python - "$WORK/check/$m/run.json" "$total" "$w" "$ov" "$m" <<'PY'
import json, math, sys
r = json.load(open(sys.argv[1])); total, w, ov, m = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
win = r["windows"][0]["seconds"] if r.get("windows") else r["seconds"]
nwin = 1 if w == 0 else math.ceil(max(total - ov, 1) / max(w - ov, 1)) + 2      # + 2: one window per take more
load = max(r["seconds"] - sum(x["seconds"] for x in r.get("windows", [])), 0)
est = (nwin * win + 2 * load) / 60
print(f"{m:12s} window {w:3d}: {win:5.1f} s per window, peak {r.get('peak_vram_gb')} GB, "
      f"load {load:4.0f} s -> full run ~{est:.0f} min ({nwin} windows, {total} frames)")
PY
    done < "$WORK/check_ok.txt"
    for m in $MODELS; do grep -q "^$m " "$WORK/check_ok.txt" || echo "$m: FAILED, see $WORK/logs/check_${m}_w*.log"; done
    echo "next: make pod   (reuses the environments and the downloaded weights)"
}
stage_stop () {
    if [ "$STOP" != "1" ]; then echo "STOP=$STOP: the pod keeps running (stop it in the RunPod console)"; return; fi
    if [ ! -f "$WORK/uploaded_$TAG" ]; then echo "results not uploaded: the pod keeps running so nothing is lost"; return; fi
    if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
        echo "results are on Hugging Face: stopping pod $RUNPOD_POD_ID"; sync; runpodctl stop pod "$RUNPOD_POD_ID"
    else
        echo "not a RunPod pod (no RUNPOD_POD_ID / runpodctl): stop the machine yourself"
    fi
}

stage="${1:-all}"
case "$stage" in
  all) for s in setup data frames check run upload stop; do echo "=== $s ==="; "stage_$s" 2>&1 | tee -a "$WORK/logs/pod.log"; done ;;
  setup|data|frames|check|run|upload|stop|test) "stage_$stage" 2>&1 | tee -a "$WORK/logs/pod.log" ;;
  *) echo "stages: setup data frames check run upload stop all test"; exit 2 ;;
esac
