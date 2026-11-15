# Inference

## Load a model

```python
from decision_models import config, models

encoder = models.load(config.ModelRef(kind="encoder", path="runs/ultrabert_380m_lr1.5e-5"))
decoder = models.load(config.ModelRef(kind="decoder", path="Athekunal/llm-jev-qwen3-0.6b"))
lora = models.load(config.ModelRef(kind="decoder", path="Qwen/Qwen3-0.6B", adapter="runs/q/adapter"))

scores = encoder.score(rows)                 # one tensor s [K] per row
choice = encoder.predict(rows[0])            # index into rows[0].options
```

## Decisions for a split

```bash
decision-models-predict model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5 out=runs/predict/eval.jsonl
decision-models-predict model.kind=decoder model.path=Athekunal/llm-jev-qwen3-0.6b \
    limit=100 citation={} out=runs/predict/eval_cited.jsonl      # + top-3 ContextCite sentences
```

## Local server ("Try it")

```bash
uv sync --extra serve
decision-models-serve model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5 \
    evidence=runs/evidence_ultrabert_380m          # optional evidence head
decision-models-serve model.kind=decoder model.path=Athekunal/llm-jev-qwen3-0.6b citation.removal=full
```

Open http://127.0.0.1:8767/, sample a row (by dataset, gold evidence only, shuffled options), then **Cite**.

```bash
curl "http://127.0.0.1:8767/api/sample?source=musique&spans=1"
curl "http://127.0.0.1:8767/api/row?id=<id>"
curl "http://127.0.0.1:8767/api/attribute?id=<id>"
```

## SGLang

```bash
make sglang-install                                               # once
make sglang-serve BASE=Qwen/Qwen3-4B ADAPTER=runs/qwen3_4b/adapter GPUS=0 PORT=30000
decision-models-sglang tokenizer=runs/qwen3_4b/merged "rows.datasets=[contractnli]" rows.per=5
make sglang-stop PORT=30000
```

```python
from transformers import AutoTokenizer
from decision_models.inference import sglang

tok = AutoTokenizer.from_pretrained("runs/qwen3_4b/merged")
answer, s = sglang.predict("http://127.0.0.1:30000", tok, row)
result = sglang.attribute("http://127.0.0.1:30000", tok, row)    # ContextCite by deletion
```

## Merge a LoRA adapter

```bash
decision-models-merge model=Qwen/Qwen3-4B adapter=runs/qwen3_4b/adapter out=runs/qwen3_4b/merged
```
