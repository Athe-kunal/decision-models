# Citations and evidence

## ContextCite

```python
from decision_models import config, models
from decision_models.citation import cite

model = models.load(config.ModelRef(kind="decoder", path="Athekunal/llm-jev-qwen3-0.6b"))
settings = config.CiteConfig(removal="full")      # decoders: reader | full | delete
cite.prepare(model, settings)                     # packed attention + fused kernels, once
result = cite.cite(model, row, settings)
result.answer, result.q, result.weights           # one weight per sentence
cite.top_sentences(row, result, k=3)
```

| removal | model | how | equals deletion |
|---|---|---|---|
| `reader` | decoder | document once, ablated readers packed (FlexAttention) | no (leaks) |
| `full` | decoder | kept tokens packed, shared prefixes once (FA2 varlen) | yes |
| `delete` | any | one scored row per ablation (encoders pack them on FA2) | yes |

Lower level:

```python
from decision_models.citation import context_cite, masked
from decision_models.models import decoder, flash_attention

flash_attention.use_decoder(model.network); decoder.fuse(model.network)
masked.attribute(model, row, removal="full")
context_cite.attribute(model, row)                # any DecisionModel, by deletion
context_cite.attribute_with(my_scorer, row)       # your own (example, sentences, keep) -> scores
```

## Measure ContextCite against gold spans

```bash
scripts/attribution.sh Athekunal/llm-jev-qwen3-0.6b runs/attribution/qwen3_0.6b 0 50
decision-models-cite-eval model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5 citation.removal=delete out=runs/attribution/ub
```

Writes `rows.jsonl` and `summary.json`: span AP / top-1, drop@k, LDS, seconds.

## Evidence head (encoders)

```bash
make evidence CONFIG=configs/evidence/ultrabert_380m.yaml GPUS=0
decision-models-train-evidence --config configs/evidence/ultrabert_380m.yaml data.labels=gold out=runs/ev_gold
```

```python
from decision_models.evidence import head as head_lib
from decision_models.models import encoder

model = encoder.MaskedEncoder.from_run("runs/ultrabert_380m_lr1.5e-5_sigmoidnorm")
evidence = head_lib.load("runs/evidence_ultrabert_380m_sigmoidnorm", model)   # head + tuned threshold
pred = head_lib.predict(model, evidence.head, row)
pred.answer, pred.tokens.where, pred.probabilities      # (part, start, end) and p per token
```

## Generated evidence labels (best-of-N, SelfCite reward)

```bash
uv sync --extra spans
GPUS="0 1" PORTS="8011 8012" scripts/serve_labeller.sh
decision-models-spans --config configs/evidence/spans.yaml \
    "base_url=[http://127.0.0.1:8011/v1,http://127.0.0.1:8012/v1]" out=runs/spans/train.jsonl
```
