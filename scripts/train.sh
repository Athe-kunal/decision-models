#!/usr/bin/env bash
# Launches one training run in the background with Accelerate.
#
#   scripts/train.sh <config.yaml> [gpus] [overrides ...]
#   scripts/train.sh configs/train/qwen3_0.6b_lora.yaml 2
#   scripts/train.sh configs/train/qwen3_4b_lora.yaml 0,1,2,3 optim.lr=5e-5 run.out=runs/x
#   scripts/train.sh configs/train/ultrabert_380m_sigmoid_norm.yaml 0
#
# Decoder configs (configs/train/qwen*) run llm training with FSDP2, the others
# the encoder with DDP; ACCELERATE_CONFIG overrides the Accelerate config.
# Output: the config's run.out (config.yaml, weights, eval_*.json, train.log,
# tb/) and <run.out>.log (stdout/stderr). Prints the launcher's process id.
set -euo pipefail
cd "$(dirname "$0")/.."

cfg=${1:?config yaml}; gpus=${2:-0}; shift $(( $# < 2 ? $# : 2 ))
if [[ $(basename "$cfg") == qwen* ]]; then
  module=decision_models.training.decoder; default=configs/accelerate/fsdp2.yaml
else
  module=decision_models.training.encoder; default=configs/accelerate/ddp.yaml
fi
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-$default}
PORT=${PORT:-$((29500 + RANDOM % 1000))}
out=$(.venv/bin/python - "$cfg" "$@" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_dotlist(sys.argv[2:]))
print(cfg.run.out)
PY
)
nproc=$(( $(tr -cd , <<< "$gpus" | wc -c) + 1 ))
mkdir -p "$(dirname "$out")"
CUDA_VISIBLE_DEVICES=$gpus nohup .venv/bin/accelerate launch \
    --config_file "$ACCELERATE_CONFIG" --num_processes "$nproc" \
    --main_process_port "$PORT" -m "$module" --config "$cfg" "$@" > "$out.log" 2>&1 &
echo "training $cfg -> $out on GPUs $gpus (pid $!, log $out.log)"
