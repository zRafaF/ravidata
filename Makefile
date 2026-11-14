# RAVIDATA — every command of the project. Python is managed by uv (https://docs.astral.sh/uv/);
# `make` and `git` are the only other things a machine needs (`apt-get install -y make` on a fresh pod).
# Code: https://github.com/zRafaF/ravidata   Data: https://huggingface.co/datasets/DoninhaD/ravidata
.PHONY: help install test rec rec-sim align pack calib targets data data-all dataset dataset-upload \
        maps fuse static-depth static-depth-refit static-depth-upload viz analysis analysis-quick plots an-tags an-extrinsic an-truth \
        an-sensor an-eval an-opt an-eval-tuned an-floor an-cordic an-ablation story report play pod pod-test pod-status pod-watch pod-setup pod-data pod-frames pod-check pod-run pod-upload pod-stop pod-clean

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
	@echo "  make data             download the recordings to data/ravidata (all takes; DATA_TAKES='a b' for some)"
	@echo "  make data-all         the whole dataset repository, derived maps included (~5 GB)"
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
	@echo "  make pod-watch        the same, refreshed every 30 s (Ctrl-C quits the watch only); LOG_LINES=40 for more log"
	@echo "  make pod-<stage>      one stage: setup data frames check run upload stop"
	@echo "  STOP=1 (default) stops the RunPod pod once the maps are on Hugging Face; STOP=0 keeps it"
	@echo "  variables: MODELS='$(MODELS)' TAKES='$(TAKES)'"
	@echo "             WINDOW=$(WINDOW) (largest tried; the check falls back to 48/32/16 if it does not fit;"
	@echo "             0 = all frames in one pass, needs ~96 GB) OVERLAP=$(OVERLAP) STEP=$(STEP) (keyframe candidates) RES=$(RES)"
	@echo "             HF_TOKEN=<write token> to upload results to derived/maps-$(TAG)/"
	@echo ""
	@echo "  back on your PC (CPU):"
	@echo "  make maps             download the pod's maps (derived/maps-$(TAG)) + the course videos to data/ravidata,"
	@echo "                        and re-make the exact keyframe images the models saw (for colours)"
	@echo "  make fuse             global map on the PC: loop closures + pose graph + weighted TSDF"
	@echo "                        every model and take by default (~2-3 min each, ~20 min in all);"
	@echo "                        MODEL=pi3x TAKE=indoor_course for one"
	@echo "  make static-depth TAKE=all   CPU depth model per frame of the static takes, scaled by the AprilTags"
	@echo "                        DEPTH_MODEL=depthpro (default; da2l, da2s or a transformers id); PER_PART=5"
	@echo "  make static-depth-refit TAKE=all   re-fit the saved depth maps on the tags (FIT=auto|scale|affine, no model)"
	@echo "  make static-depth-upload   put derived/static-depth-v1 on the HF dataset (after hf auth login)"
	@echo "  make viz              3D viewer of the maps in the browser: whole take, one window, two windows,"
	@echo "                        compare models, measure (VIZ_ARGS=\"--model vggt --take outdoor_course\"): docs/maps.md"
	@echo ""
	@echo "  radar analyses (CPU; tables, figures, videos in results/analysis-v1, kept in git): docs/analysis.md"
	@echo "  make analysis-quick   every step on a few seconds of each recording (~6 min), + plots, to"
	@echo "                        results/analysis-quick: run this first after any change"
	@echo "  make analysis         every step but CORDIC, full data, in order (hours: see docs/analysis.md);"
	@echo "                        FROM=truth starts there (the steps before it already done)"
	@echo "  make an-tags an-extrinsic an-truth an-sensor an-eval an-ablation an-opt an-eval-tuned an-floor an-cordic"
	@echo "                        one step (QUICK=1 for its quick version)"
	@echo "  make plots            figures from the existing tables only (seconds); QUICK=1 for the quick tables"
	@echo "  make story            explanatory figures (how the truth is built, radar range-time maps with the truth)"
	@echo "  make report           results/analysis-v1/REPORT.md: every number as Markdown tables"
	@echo "  make play REC=indoor_course/01_D1_tcc-1d-40mm START=20 SECONDS=15 PLAY_FAMILIES=\"ca_cfar grid\""
	@echo "                        video of one recording: camera + instantaneous radar heatmap + detections + truth"
	@echo "                        (PLAY_ARGS=\"--gif --bg empty --params tuned_pooled\"); results/analysis-v1/videos/"
	@echo "  variables: AN_TAKES='static_pass1 ...' FAMILIES='thr grid ...' TRIALS=$(TRIALS) PROTOCOL=$(PROTOCOL)"
	@echo "             (strict | pooled | both) WORKERS=$(WORKERS) TAG_STEP=$(TAG_STEP) MAP_MODEL=$(MAP_MODEL)"

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
data:                                  # the recordings (all takes, calibration, metadata), without derived/
	uv run hf download $(HF_REPO) --repo-type dataset --local-dir data/ravidata \
	  $(if $(DATA_TAKES),$(foreach t,$(DATA_TAKES),--include "takes/$(t)/*") --include "calibration/*" --include "*.json" --include "README.md",--exclude "derived/*")

data-all:                              # the whole dataset repository: recordings + derived/ (maps, ~4 GB more)
	uv run hf download $(HF_REPO) --repo-type dataset --local-dir data/ravidata

MAPS := data/ravidata/derived/maps-$(TAG)
maps: install
	$(UV) run hf download $(HF_REPO) --repo-type dataset --local-dir data/ravidata \
	  --include "derived/maps-$(TAG)/*" --include "calibration/camera.calib.json" \
	  $(foreach t,$(TAKES),--include "takes/$(t)/video.mp4" --include "takes/$(t)/take.json")
	@for t in $(TAKES); do \
	  if [ -f "$(MAPS)/frames_$$t.json" ] && [ ! -f "$(MAPS)/frames/$$t/frames.json" ]; then \
	    $(UV) run python pod/frames.py --data data/ravidata --take $$t --out "$(MAPS)/frames/$$t" \
	      --from-json "$(MAPS)/frames_$$t.json" || exit 1; \
	  fi; \
	done

MODEL ?= all
TAKE  ?= all
fuse:                                  # pose graph + ICP + weighted TSDF on the PC (no GPU), every model and take
	uv run --group viz python scripts/map_fuse.py --model $(MODEL) --take $(TAKE)

PER_PART ?= 5
DEPTH_MODEL ?= depthpro
FIT ?= auto
static-depth:                          # monocular metric depth of a static take on the CPU, grounded on the tags
	uv run --group depth python scripts/static_depth.py --take $(TAKE) --per-part $(PER_PART) --model $(DEPTH_MODEL)

static-depth-refit:                    # re-ground the saved depth maps on the tags (no model run; FIT=auto|scale|affine)
	uv run --group depth python scripts/static_depth.py --take $(TAKE) --per-part $(PER_PART) --model $(DEPTH_MODEL) \
	  --refit --fit $(FIT)

static-depth-upload:                   # derived/static-depth-v1 -> the HF dataset (needs `uv run hf auth login`)
	uv run hf upload $(HF_REPO) data/ravidata/derived/static-depth-v1 derived/static-depth-v1 --repo-type dataset \
	  --commit-message "static-depth-v1: monocular metric depth of the static takes, scaled on the AprilTags"

VIZ_ARGS ?=
viz:                                   # plain `uv`: also works from PowerShell with GNU make
	uv run --group viz python scripts/map_viz.py $(VIZ_ARGS)

# ---- radar analyses (python -m analysis <step>); plain `uv` so it also runs from PowerShell
AN_TAKES  ?=
FAMILIES  ?=
TRIALS    ?= 150
PROTOCOL  ?= pooled
OBJECTIVE ?= both
WORKERS   ?= 4
TAG_STEP  ?= 2
MAP_MODEL ?= pi3x
QUICK     ?=
FROM      ?=
AN_ARGS = $(if $(AN_TAKES),--takes $(AN_TAKES)) $(if $(FAMILIES),--families $(FAMILIES)) --trials $(TRIALS) \
          --protocol $(PROTOCOL) --objective $(OBJECTIVE) --workers $(WORKERS) --tag-step $(TAG_STEP) --model $(MAP_MODEL) $(if $(QUICK),--quick) $(if $(FROM),--from $(FROM))
AN = uv run --group analysis python -m analysis

an-tags:
	$(AN) tags $(AN_ARGS)
an-extrinsic:
	$(AN) extrinsic $(AN_ARGS)
an-truth:
	$(AN) truth $(AN_ARGS)
an-sensor:
	$(AN) sensor $(AN_ARGS)
an-eval:
	$(AN) eval $(AN_ARGS)
an-ablation:                           # one stage at a time over a threshold sweep (cached per recording)
	$(AN) ablation $(AN_ARGS)
story:                                 # explanatory figures: how the truth is built, range-time maps
	$(AN) story $(AN_ARGS)
report:                                # results/analysis-v1/REPORT.md: every number as tables
	$(AN) report $(AN_ARGS)
an-opt:
	$(AN) opt $(AN_ARGS)
an-eval-tuned:
	$(AN) eval-tuned $(AN_ARGS)
an-floor:                              # where the early alarms come from (floor in beam, phantom); minutes
	$(AN) floor $(AN_ARGS)
an-cordic:
	$(AN) cordic $(AN_ARGS)
plots:
	$(AN) plots $(AN_ARGS)
REC ?= indoor_course/01_D1_tcc-1d-40mm
START ?= 0
SECONDS ?= 15
PLAY_FAMILIES ?= ca_cfar grid
PLAY_ARGS ?=
play:                                  # video: camera + instantaneous radar heatmap + detections + truth
	uv run --group analysis python -m analysis.play --rec $(REC) --start $(START) --seconds $(SECONDS) \
	  --families $(PLAY_FAMILIES) $(PLAY_ARGS)
analysis:                              # every step, full data, in order
	$(AN) all $(AN_ARGS)
analysis-quick:                        # every step on a few seconds per recording, then the figures
	$(AN) all $(AN_ARGS) --quick --protocol both

dataset: install
	$(UV) run python scripts/hf_dataset.py all

dataset-upload: install
	$(UV) run python scripts/hf_dataset.py upload --repo $(HF_REPO)

pod: $(UV)
	$(POD_ENV) bash pod/pod.sh all

pod-test pod-setup pod-data pod-frames pod-check pod-run pod-upload pod-stop: $(UV)
	$(POD_ENV) bash pod/pod.sh $(@:pod-%=%)

LOG_LINES ?= 25
pod-status:
	@python3 pod/status.py $(LOG_LINES)

pod-watch:
	watch -n 30 python3 pod/status.py $(LOG_LINES)

pod-clean:
	rm -rf work pod/third_party
