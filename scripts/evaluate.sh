#!/usr/bin/env bash
# Evaluates a model on the eval split and on the external benchmarks.
# Runs in the foreground.
#
#   scripts/evaluate.sh <kind> <model path> [gpu] [tag] [overrides ...]
#   scripts/evaluate.sh decoder Qwen/Qwen3-0.6B 1 base out=runs/qwen3_0.6b_lora
#   scripts/evaluate.sh decoder Qwen/Qwen3-0.6B 1 final model.adapter=runs/qwen3_0.6b_lora/adapter out=runs/qwen3_0.6b_lora
#   scripts/evaluate.sh encoder runs/ultrabert_380m_lr1.5e-5 0
#
# Output: <out>/eval_<tag>.json and eval_benchmarks_<tag>.json (+ examples).
set -euo pipefail
cd "$(dirname "$0")/.."
kind=${1:?decoder or encoder}; path=${2:?model path}; gpu=${3:-0}; tag=${4:-final}
shift $(( $# < 4 ? $# : 4 ))
export CUDA_VISIBLE_DEVICES=$gpu
args=(model.kind="$kind" model.path="$path" tag="$tag")
[[ $kind == encoder ]] && args+=(out="$path")
.venv/bin/python -m decision_models.evaluation.evaluate --config "configs/evaluate/$kind.yaml" \
    "${args[@]}" save_rows=true "$@"
.venv/bin/python -m decision_models.evaluation.benchmarks --config configs/evaluate/benchmarks.yaml \
    "${args[@]}" "$@"
