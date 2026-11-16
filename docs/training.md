# Training

A run is a YAML file under [`configs/train/`](https://github.com/Athe-kunal/decision-models/tree/main/configs/train) plus `key=value` overrides. A typo or an invalid value fails at startup. The resolved config is saved to `<run.out>/config.yaml`.

`make train-encoder CONFIG=`[`configs/train/ultrabert_380m_sigmoid_norm.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sigmoid_norm.yaml)` GPUS=0`<br>
`make train-decoder CONFIG=`[`configs/train/qwen3_0.6b_lora.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_lora.yaml)` GPUS=0,1,2,3 ARGS="optim.lr=5e-5 run.out=runs/x"`<br>
`scripts/train.sh `[`configs/train/qwen3_0.6b_full.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_full.yaml)` 0 run.seed=1     # same, in the background, log in <run.out>.log`

`GPUS` sets the processes: DDP for encoders, FSDP2 for decoders. `optim.batch_size` is the total across all GPUs.

| config | model | loss |
|---|---|---|
| [`ultrabert_380m_softmax_yes.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_softmax_yes.yaml) | UltraBERT-380M | softmax over ℓ(yes) |
| [`ultrabert_380m_softmax_yesno.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_softmax_yesno.yaml) | UltraBERT-380M | softmax over ℓ(yes) − ℓ(no) |
| [`ultrabert_380m_sigmoid_norm.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sigmoid_norm.yaml) | UltraBERT-380M | σ(s) normalised over the options |
| [`ultrabert_380m_sqrt_softplus.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sqrt_softplus.yaml) | UltraBERT-380M | √softplus(s) normalised over the options |
| [`ettin_encoder_400m.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ettin_encoder_400m.yaml) | Ettin-400M | sigmoid-norm |
| [`qwen3_0.6b_lora.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_lora.yaml), [`qwen3_4b_lora.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_4b_lora.yaml) | Qwen3, LoRA r16 | L_tree + KL anchors |
| [`qwen3_0.6b_full.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_full.yaml) | Qwen3-0.6B, full fine-tuning | L_tree + KL anchors |

## Arguments

### Shared by both families

| key | default | meaning |
|---|---|---|
| `data.source` | `data/splits/reason_long_nimble` | local folder of `train/eval.parquet`, or a Hub id (`Athekunal/llm-jev-reason-long`) |
| `data.max_tokens` | 16384 | drop longer training rows |
| `data.eval_per` | 100 | dev rows per dataset evaluated before and after training (at most 100; never test rows) |
| `optim.lr` | 1e-4 | peak learning rate |
| `optim.batch_size` | 32 | rows per optimizer step, total across all GPUs |
| `optim.warmup_steps` / `warmup_ratio` | None / 0.05 | linear warmup, then linear decay |
| `optim.min_lr_ratio` | 0.0 | final lr / peak |
| `optim.weight_decay`, `betas`, `eps`, `max_grad_norm` | 0.01, [0.9, 0.999], 1e-8, 1.0 | AdamW and clipping |
| `run.out` | required | run directory |
| `run.seed` | 20260922 | option shuffles and visiting order |
| `run.epochs` | 1 | passes over the training rows |
| `run.max_steps` | -1 | stop early, e.g. `10` for a smoke test |
| `run.eval_every` | 0 | also evaluate and checkpoint every N steps |
| `run.skip_base_eval` | false | skip the evaluation before training |
| `run.mixed_precision` | `"no"` | `bf16` in every config; `fp16` on pre-Ampere GPUs (quote `"no"` in YAML) |

### Encoder (`model.*`, `format.*`, `optim.loss`)

| key | default | meaning |
|---|---|---|
| `model.id` | required | Hub id or path of the masked LM |
| `model.revision` | None | Hub commit |
| `model.attention` | `varlen` | `varlen` (FlashAttention-2, packs rows, Ampere or newer) or `sdpa` |
| `model.pack_train` | true | pack each GPU's rows into one varlen forward, about 4× faster; `false`, or `attention: sdpa` (Ettin), trains one row per forward |
| `model.pack_tokens` | 32768 | tokens per packed forward |
| `model.gradient_checkpointing` | true | recompute activations in the backward pass |
| `format.template` | `decision_mlm` | `prompts/<name>.j2`, one `mask` per option |
| `format.yes_token` | `" yes"` | s_k = ℓ_k(yes) |
| `format.no_token` | None | set `" no"`: s_k = ℓ_k(yes) − ℓ_k(no) |
| `optim.loss` | `listwise` | `listwise` (softmax), `sigmoid_norm`, `sqrt_softplus` (the last two need `format.no_token`) |

### Decoder (`model.*`, `objective.*`)

| key | default | meaning |
|---|---|---|
| `model.id` | required | Hub id or path of the causal LM (Qwen3 / Qwen3.5 chat template) |
| `model.finetune` | `lora` | `lora` or `full` (fp32 weights, frozen bf16 reference model) |
| `model.lora_rank` | 16 | LoRA rank (alpha = 2 × rank) |
| `model.batch_tokens`, `batch_rows` | 32768, 64 | scoring batch limits (evaluation) |
| `objective.anchors` | `[mass, out, pos]` | KL anchors added to L_tree, e.g. `"objective.anchors=[out,pos]"` |
| `objective.lam` | 1.0 | anchor weight |
| `data.gen_per` | 0 | dev rows per dataset also answered as free text |

Qwen3.5 needs the `qwen35` extra (`uv sync --extra qwen35`, or the `qwen35` [Docker](docker.md) target): `ARGS="model.id=Qwen/Qwen3.5-4B run.out=runs/qwen35_4b"`.

## Outputs

```text
runs/x/
  config.yaml          resolved config
  train.log  tb/       per-step loss, accuracy, lr, grad norm (TensorBoard)
  eval_base.json       dev accuracy before training, per dataset and ALL
  eval_final.json      after training
  model/               encoder weights + tokenizer; full fine-tuned decoder
  adapter/             decoder LoRA adapter
  checkpoints/step-N/  with run.eval_every
```

## Evaluate

`scripts/evaluate.sh encoder runs/x 0                     # dev split + external benchmarks`<br>
`decision-models-evaluate --config `[`configs/evaluate/encoder.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/evaluate/encoder.yaml)` model.path=runs/x rows.part=test tag=test`<br>
`decision-models-evaluate --config `[`configs/evaluate/decoder.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/evaluate/decoder.yaml)` model.path=Qwen/Qwen3-0.6B model.adapter=runs/q/adapter out=runs/q`<br>
`decision-models-benchmarks --config `[`configs/evaluate/benchmarks.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/evaluate/benchmarks.yaml)` model.kind=encoder model.path=runs/x out=runs/x`

| key | meaning |
|---|---|
| `model.kind`, `model.path` | `encoder` + run directory, or `decoder` + Hub id/path |
| `model.adapter` | decoder LoRA adapter |
| `rows.part` | `dev` (tune here), `test` (report here), `all` |
| `rows.datasets`, `rows.per` | subset of datasets, rows per dataset |
| `out`, `tag` | writes `<out>/eval_<tag>.json` |

## From Python

[`configs/train/ultrabert_380m_sigmoid_norm.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sigmoid_norm.yaml)

```python
from decision_models import config
from decision_models.training import encoder, trainer

cfg, raw = config.from_command_line(config.EncoderTrainConfig, argv=[
    "--config", "configs/train/ultrabert_380m_sigmoid_norm.yaml", "run.out=runs/x", "run.max_steps=10"])
trainer.launch(cfg, raw, lambda device: encoder.build(cfg, device))
```
