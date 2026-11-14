PY := .venv/bin/python
ACCELERATE := .venv/bin/accelerate
GPUS ?= 0
comma := ,
NPROC = $(words $(subst $(comma), ,$(GPUS)))
# CONFIG picks a YAML under configs/, ARGS adds dotlist overrides,
# e.g. make train-encoder CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml ARGS="optim.lr=8e-6"
CONFIG ?=
ARGS ?=
DATA := data/splits/reason_long_nimble
# make sglang-merge / sglang-serve: the trained adapter, its base model, the merged checkpoint
BASE ?= Qwen/Qwen3-4B
ADAPTER ?= runs/qwen3_4b/adapter
MERGED ?= $(dir $(ADAPTER))merged
PORT ?= 30000
BLOG_PORT ?= 8765
# make torch-profiler: the profiled model and where its outputs go
PROFILE_MODEL ?= Qwen/Qwen3-0.6B
PROFILE_OUT ?= runs/profile/$(notdir $(PROFILE_MODEL))

.PHONY: help install download data train-decoder train-encoder evaluate benchmarks evidence \
	predict serve sglang-install sglang-merge sglang-serve sglang-stop torch-profiler blog \
	test lint format

help:
	@echo "make install         uv sync (CUDA 12.8 torch) with the dev extra"
	@echo "make data            download the pinned sources, build $(DATA)"
	@echo "make train-decoder   CONFIG=configs/train/qwen3_0.6b_lora.yaml GPUS=0,1,2,3 (FSDP2)"
	@echo "make train-encoder   CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml GPUS=0 (DDP)"
	@echo "make evaluate        CONFIG=configs/evaluate/encoder.yaml ARGS='model.path=runs/...'"
	@echo "make benchmarks      CONFIG=configs/evaluate/benchmarks.yaml ARGS='model.kind=... model.path=...'"
	@echo "make evidence        CONFIG=configs/evidence/ultrabert_380m.yaml"
	@echo "make predict         ARGS='model.kind=decoder model.path=... citation={} out=runs/predict/eval.jsonl'"
	@echo "make serve           ARGS='model.kind=encoder model.path=runs/...'  (page on http://127.0.0.1:8767/)"
	@echo "make sglang-install  SGLang as a uv tool (its own environment)"
	@echo "make sglang-merge    merge ADAPTER into BASE -> MERGED (CPU)"
	@echo "make sglang-serve    serve MERGED with SGLang on GPUS (http://127.0.0.1:PORT), merging first if needed"
	@echo "make sglang-stop     stop the server on PORT"
	@echo "make torch-profiler  profile training steps of PROFILE_MODEL -> PROFILE_OUT (docs/profiling.md)"
	@echo "make blog            serve blogs/ at http://127.0.0.1:$(BLOG_PORT)/ (Ctrl-C to stop)"
	@echo "make test / lint / format"

install:
	uv sync --extra dev

download:
	$(PY) -m decision_models.data.build --config configs/data/reason_long_nimble.yaml download_only=true

data: download
	$(PY) -m decision_models.data.build --config configs/data/reason_long_nimble.yaml $(ARGS)

train-decoder:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(ACCELERATE) launch --config_file configs/accelerate/fsdp2.yaml \
		--num_processes $(NPROC) -m decision_models.training.decoder --config $(CONFIG) $(ARGS)

train-encoder:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(ACCELERATE) launch --config_file configs/accelerate/ddp.yaml \
		--num_processes $(NPROC) -m decision_models.training.encoder --config $(CONFIG) $(ARGS)

evaluate:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.evaluation.evaluate --config $(CONFIG) $(ARGS)

benchmarks:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.evaluation.benchmarks --config $(CONFIG) $(ARGS)

evidence:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.evidence.train --config $(CONFIG) $(ARGS)

predict:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.inference.predict $(if $(CONFIG),--config $(CONFIG)) $(ARGS)

serve:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.inference.server $(if $(CONFIG),--config $(CONFIG)) $(ARGS)

sglang-install:
	uv tool install --prerelease=allow --python 3.11 sglang==0.5.22

$(MERGED)/config.json:
	$(PY) -m decision_models.inference.merge model=$(BASE) adapter=$(ADAPTER) out=$(MERGED)

sglang-merge: $(MERGED)/config.json

sglang-serve: $(MERGED)/config.json
	scripts/sglang_serve.sh $(MERGED) $(GPUS) $(PORT)
	@echo "waiting for http://127.0.0.1:$(PORT)/health ..."
	@for i in $$(seq 600); do curl -sf http://127.0.0.1:$(PORT)/health >/dev/null && echo ready && exit 0; \
		sleep 2; done; echo "not ready after 20 min: see logs/sglang_$(PORT).log"; exit 1

sglang-stop:
	@pidfile=logs/sglang_$(PORT).pid; [ -f $$pidfile ] || { echo "no $$pidfile"; exit 0; }; \
	pid=$$(cat $$pidfile); kill -- -$$pid 2>/dev/null && echo "stopped server group $$pid"; \
	for i in $$(seq 30); do ps -p $$pid >/dev/null || { rm -f $$pidfile; echo "gone"; exit 0; }; sleep 1; done; \
	echo "pid $$pid still alive after 30 s"

torch-profiler:
	CUDA_VISIBLE_DEVICES=$(GPUS) $(PY) -m decision_models.profiling.profile_step \
		model=$(PROFILE_MODEL) data=$(DATA) out=$(PROFILE_OUT) $(ARGS)

blog:
	cd blogs && ( [ -d node_modules ] || npm install --no-fund --no-audit ) \
		&& npx vite --port $(BLOG_PORT) --host 127.0.0.1 --strictPort

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src tests analysis && .venv/bin/ruff format --check src tests analysis

format:
	.venv/bin/ruff format src tests analysis && .venv/bin/ruff check --fix src tests analysis
