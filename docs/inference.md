# Inference

## Python

```python
from decision_models import config, models
from decision_models.data import schema

encoder = models.load(config.ModelRef(kind="encoder", path="runs/ultrabert_380m_lr1.5e-5_sigmoidnorm"))
decoder = models.load(config.ModelRef(kind="decoder", path="Athekunal/llm-jev-qwen3-0.6b"))
lora    = models.load(config.ModelRef(kind="decoder", path="Qwen/Qwen3-0.6B", adapter="runs/q/adapter"))

row = schema.Example(id="q:1", source="mine",
                     parts=(schema.Part("Document", "The supplier delivers by May 1."),),
                     question="When must the supplier deliver?",
                     options=("By May 1", "By June 1", "Not stated"), gold=0)
s = encoder.score([row])[0]        # tensor [K], one score per option
k = encoder.predict(row)           # argmax s: index into row.options
```

`gold` is required by `Example` but not used for scoring; set it to any valid index.

| `ModelRef` field | meaning |
|---|---|
| `kind` | `encoder` or `decoder` |
| `path` | encoder: training run directory (its `config.yaml` gives the prompt); decoder: Hub id or path |
| `adapter` | decoder LoRA adapter, merged on load |
| `attention` | encoder: override the run's attention (`sdpa` on pre-Ampere GPUs) |
| `dtype` | encoder weights, `float32` (default, as trained) or `bfloat16` |

## A split, to JSONL

```bash
decision-models-predict model.kind=encoder model.path=runs/x rows.source=data/splits/mine out=runs/predict/mine.jsonl
decision-models-predict model.kind=decoder model.path=Athekunal/llm-jev-qwen3-0.6b limit=100 citation={} out=runs/predict/cited.jsonl
```

Each line is `{id, source, answer, option, q, gold, correct}`, plus the top evidence sentences with `citation={}`.

| key | meaning |
|---|---|
| `rows.source`, `rows.split` | data folder or Hub id, and split (default `eval`) |
| `rows.datasets`, `rows.per` | only these datasets, at most N rows each |
| `limit` | first N rows |
| `citation` | `{}` adds ContextCite evidence ([citations](citation.md)) |

## Local page

```bash
uv sync --extra serve
decision-models-serve model.kind=encoder model.path=runs/x evidence=runs/evidence_x    # evidence head optional
```

Open http://127.0.0.1:8767/, pick a row, then **Cite**. The same server answers `GET /api/sample?source=musique`, `/api/row?id=<id>` and `/api/attribute?id=<id>`.

## SGLang (decoders)

```bash
make sglang-install
make sglang-serve BASE=Qwen/Qwen3-4B ADAPTER=runs/qwen3_4b/adapter GPUS=0 PORT=30000   # merges the adapter first
decision-models-sglang tokenizer=runs/qwen3_4b/merged "rows.datasets=[contractnli]" rows.per=5
make sglang-stop PORT=30000
```

```python
from transformers import AutoTokenizer
from decision_models.inference import sglang

tok = AutoTokenizer.from_pretrained("runs/qwen3_4b/merged")
answer, s = sglang.predict("http://127.0.0.1:30000", tok, row)
```

Merge an adapter on its own: `decision-models-merge model=Qwen/Qwen3-4B adapter=runs/qwen3_4b/adapter out=runs/qwen3_4b/merged`.
