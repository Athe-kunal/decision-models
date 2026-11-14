# Training

Every run is a YAML file under `configs/train/` plus `key=value` overrides. The resolved config is saved to `<run.out>/config.yaml`.

## Encoder (one mask per option)

```bash
make train-encoder CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml GPUS=0
scripts/train.sh configs/train/ultrabert_380m_sigmoid_norm.yaml 0 optim.lr=8e-6 run.out=runs/x  # background, logs to runs/x.log
```

Recipes (`optim.loss`): `listwise`, `sigmoid_norm`, `sqrt_softplus`. The last two need `format.no_token: " no"`.

```bash
configs/train/ultrabert_380m_softmax_yes.yaml
configs/train/ultrabert_380m_softmax_yesno.yaml
configs/train/ultrabert_380m_sigmoid_norm.yaml
configs/train/ultrabert_380m_sqrt_softplus.yaml
configs/train/ettin_encoder_400m.yaml
```

## Decoder (LLM-as-Jev: trie loss + KL anchors)

```bash
make train-decoder CONFIG=configs/train/qwen3_0.6b_lora.yaml GPUS=0,1,2,3      # FSDP2
scripts/train.sh configs/train/qwen3_0.6b_full.yaml 0                          # full fine-tuning
scripts/train.sh configs/train/qwen3_0.6b_lora.yaml 0 "objective.anchors=[out,pos]"
scripts/ablate_loss.sh 0 1 2                                                   # every anchor ablation
```

## Common overrides

```bash
run.out=runs/x  run.seed=1  run.epochs=2  run.max_steps=10  run.eval_every=100
optim.lr=1e-5  optim.batch_size=64  data.source=Athekunal/llm-jev-reason-long  data.max_tokens=8192
```

## Evaluate

```bash
scripts/evaluate.sh encoder runs/ultrabert_380m_lr1.5e-5 0                     # eval split + benchmarks
decision-models-evaluate --config configs/evaluate/encoder.yaml model.path=runs/x part=test tag=test
decision-models-evaluate --config configs/evaluate/encoder.yaml model.path=runs/x label_shares=true
decision-models-evaluate --config configs/evaluate/decoder.yaml model.adapter=runs/q/adapter out=runs/q
decision-models-benchmarks --config configs/evaluate/benchmarks.yaml model.kind=encoder model.path=runs/x
```

## From Python

```python
from decision_models import config
from decision_models.training import encoder, trainer

raw = config.load(config.EncoderTrainConfig, "configs/train/ultrabert_380m_sigmoid_norm.yaml",
                  ["run.out=runs/x", "run.max_steps=10"])
cfg = config.resolve(raw)
trainer.launch(cfg, raw, lambda device: encoder.build(cfg, device))
```
