#!/usr/bin/env bash
# One isolated uv venv per map model: pod/third_party/<model>/.venv  (usage: bash pod/envs.sh <model>)
# Models: mapanything pi3x da3 vggt. Never installs anything into the project's own env.
set -euo pipefail
cd "$(dirname "$0")/.."
source pod/pins.env
export PATH="$HOME/.local/bin:$PATH"
m="$1"

clone () {    # url ref dir
    if [ ! -d "$3/.git" ]; then git clone --quiet "$1" "$3"; fi
    git -C "$3" fetch --quiet origin "$2" 2>/dev/null || git -C "$3" fetch --quiet origin
    git -C "$3" checkout --quiet "$2" 2>/dev/null || git -C "$3" checkout --quiet "origin/$2"
    echo "[$m] $(basename "$3") at $(git -C "$3" rev-parse --short HEAD)"
}
torch_cu128 () {   # reinstall torch + torchvision from the CUDA 12.8 index (a repo's older pin has no sm_120)
    uv pip install --python "$1" --reinstall --index-url "$TORCH_INDEX" "$TORCH_SPEC" torchvision
}

case "$m" in
  mapanything)
    d=pod/third_party/mapanything; clone "$MAPANYTHING_URL" "$MAPANYTHING_REF" "$d"
    [ -d "$d/.venv" ] || uv venv --quiet --python 3.12 "$d/.venv"
    torch_cu128 "$d/.venv"
    uv pip install --python "$d/.venv" -e "$d"
    check='from mapanything.models import MapAnything' ;;
  pi3x)
    d=pod/third_party/pi3; clone "$PI3_URL" "$PI3_REF" "$d"
    [ -d "$d/.venv" ] || uv venv --quiet --python 3.11 "$d/.venv"
    [ -f "$d/requirements.txt" ] && uv pip install --python "$d/.venv" -r "$d/requirements.txt"
    torch_cu128 "$d/.venv"
    [ -f "$d/pyproject.toml" ] && uv pip install --python "$d/.venv" -e "$d" --no-deps
    check='from pi3.models.pi3x import Pi3X' ;;
  da3)
    d=pod/third_party/da3; clone "$DA3_URL" "$DA3_REF" "$d"
    [ -d "$d/.venv" ] || uv venv --quiet --python 3.12 "$d/.venv"
    torch_cu128 "$d/.venv"
    uv pip install --python "$d/.venv" -e "$d"
    torch_cu128 "$d/.venv"          # in case the package pulled another torch
    uv pip install --python "$d/.venv" --index-url "$TORCH_INDEX" xformers \
        || echo "[da3] xformers not installed (optional; DA3 falls back to plain attention)"
    uv pip install --python "$d/.venv" addict                  # imported by depth_anything_3.model, not declared
    check='from depth_anything_3.api import DepthAnything3' ;;
  vggt)
    d=pod/third_party/vggt; clone "$VGGT_URL" "$VGGT_REF" "$d"
    [ -d "$d/.venv" ] || uv venv --quiet --python 3.11 "$d/.venv"
    [ -f "$d/requirements.txt" ] && uv pip install --python "$d/.venv" -r "$d/requirements.txt"
    torch_cu128 "$d/.venv"
    uv pip install --python "$d/.venv" -e "$d" --no-deps
    check='from vggt.models.vggt import VGGT' ;;
  *) echo "unknown model '$m' (mapanything pi3x da3 vggt)"; exit 2 ;;
esac
uv pip install --python "$d/.venv" pillow numpy huggingface_hub
# Model repos do not always declare every import (DA3 needs 'addict'): install what the import check reports
# missing, a few rounds, then check again.
for round in 1 2 3 4 5 6 7 8; do
    if out=$("$d/.venv/bin/python" -c "$check" 2>&1); then break; fi
    mod=$(printf '%s\n' "$out" | grep -oP "No module named '\K[^'.]+" | tail -1 || true)
    if [ -z "$mod" ]; then printf '%s\n' "$out" | tail -20; echo "[$m] import check failed"; exit 1; fi
    case "$mod" in
        cv2) pkg=opencv-python-headless ;; PIL) pkg=pillow ;; sklearn) pkg=scikit-learn ;;
        yaml) pkg=pyyaml ;; skimage) pkg=scikit-image ;; *) pkg="$mod" ;;
    esac
    echo "[$m] missing module '$mod' -> installing $pkg"
    uv pip install --python "$d/.venv" "$pkg"
done
"$d/.venv/bin/python" -c "$check; import torch; print('[$m] import ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
touch "$d/.venv/.ravidata_ok"       # pod.sh setup skips this model next time
