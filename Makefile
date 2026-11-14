# RAVIDATA — every command of the project. Python is managed by uv (https://docs.astral.sh/uv/);
# `make` and `git` are the only other things a machine needs (`apt-get install -y make` on a fresh pod).
# Code: https://github.com/zRafaF/ravidata   Data: https://huggingface.co/datasets/DoninhaD/ravidata
.PHONY: help install test rec rec-sim align pack calib targets data dataset dataset-upload \
        maps pod pod-test pod-status pod-watch pod-setup pod-data pod-frames pod-check pod-run pod-upload pod-stop pod-clean

HF_REPO ?= DoninhaD/ravidata
TAKES   ?= indoor_course outdoor_course
MODELS  ?= mapanything pi3x da3 vggt
STEP    ?= 2
RES     ?= 518
WINDOW  ?= 64
OVERLAP ?= 16
TAG     ?= v1
STOP    ?= 1
UV      := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
POD_ENV := HF_REPO="$(HF_REPO)" TAKES="$(TAKES)" MODELS="$(MODELS)" STEP=$(STEP) RES=$(RES) \
           WINDOW=$(WINDOW) OVERLAP=$(OVERLAP) TAG=$(TAG) STOP=$(STOP)

help:
	@echo "RAVIDATA: 60 GHz radar + phone video + IMU. Data: https://huggingface.co/datasets/$(HF_REPO)"
	@echo ""
	@echo "  make install          Python env (uv sync; installs uv first if missing)"
	@echo "  make test             unit tests"
	@echo "  make data             download the dataset to data/ravidata (DATA_TAKES='a b' for some takes only)"
	@echo ""
	@echo "  recording (radar MRS6130-P1812 over SPI + phone):"
	@echo "  make rec / rec-sim    recorder GUI (Simple tab = guided takes) / with the simulator"
	@echo "  make calib VIDEO=...  camera intrinsics from a checkerboard video"
	@echo "  make align / pack FOLDER=...   align phone video + IMU with the radar / export a pack"
	@echo "  make targets          printable AprilTags + checkerboard (docs/protocols/targets)"
	@echo "  make dataset          (dataset owners) blur faces + build the HF dataset from the raw recordings"
	@echo ""
	@echo "  GPU pod, maps only (git clone https://github.com/zRafaF/ravidata, then only):"
	@echo "  make pod-test         first: install every model and try it on one window of frames (~30 min);"
	@echo "                        prints pass/fail, GPU memory and a time estimate; uploads nothing"
	@echo "  make pod HF_TOKEN=... setup + data + frames + check + run + upload + stop  (inside tmux)"
	@echo "  make pod-status       progress of a running 'make pod' (windows done per model, time left, GPU)"
	@echo "  make pod-watch        the same, refreshed every 30 s (Ctrl-C to quit); see docs/pod.md for tmux"
	@echo "  make pod-<stage>      one stage: setup data frames check run upload stop"
	@echo "  STOP=1 (default) stops the RunPod pod once the maps are on Hugging Face; STOP=0 keeps it"
	@echo "  variables: MODELS='$(MODELS)' TAKES='$(TAKES)'"
	@echo "             WINDOW=$(WINDOW) (largest tried; the check falls back to 48/32/16 if it does not fit;"
	@echo "             0 = all frames in one pass, needs ~96 GB) OVERLAP=$(OVERLAP) STEP=$(STEP) (keyframe candidates) RES=$(RES)"
	@echo "             HF_TOKEN=<write token> to upload results to derived/maps-$(TAG)/"
	@echo ""
	@echo "  back on your PC (CPU):"
	@echo "  make maps             download the pod's maps (derived/maps-$(TAG)) to data/ravidata"

$(UV):
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: $(UV)
	$(UV) sync

test: install
	$(UV) run pytest -q

rec: install
	$(UV) run ravidata rec

rec-sim: install
	$(UV) run ravidata rec --sim

align: install
	$(UV) run ravidata align

pack: install
	$(UV) run ravidata pack --folder "$(FOLDER)"

calib: install
	$(UV) run ravidata calib "$(VIDEO)"

targets: install
	$(UV) run ravidata print-targets

DATA_TAKES ?=
data: install
	$(UV) run hf download $(HF_REPO) --repo-type dataset --local-dir data/ravidata \
	  $(if $(DATA_TAKES),$(foreach t,$(DATA_TAKES),--include "takes/$(t)/*") --include "calibration/*" --include "*.json" --include "README.md",)

maps: install
	$(UV) run hf download $(HF_REPO) --repo-type dataset --local-dir data/ravidata --include "derived/maps-$(TAG)/*"

dataset: install
	$(UV) run python scripts/hf_dataset.py all

dataset-upload: install
	$(UV) run python scripts/hf_dataset.py upload --repo $(HF_REPO)

pod: $(UV)
	$(POD_ENV) bash pod/pod.sh all

pod-test pod-setup pod-data pod-frames pod-check pod-run pod-upload pod-stop: $(UV)
	$(POD_ENV) bash pod/pod.sh $(@:pod-%=%)

pod-status:
	@python3 pod/status.py

pod-watch:
	watch -n 30 python3 pod/status.py

pod-clean:
	rm -rf work pod/third_party
