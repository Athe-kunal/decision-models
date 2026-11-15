# decision-models

[📝 **Blog post:** ](https://athe-kunal.github.io/decision-models/blog/) · [ 📚 **Docs:** ](https://athe-kunal.github.io/decision-models/)

🤗 **Dataset** (training mixture, held-out rows and evidence labels): [Athekunal/llm-jev-reason-long](https://huggingface.co/datasets/Athekunal/llm-jev-reason-long)

**Encoder checkpoints** (UltraBERT-380M, Apache-2.0):

- 🤗 Softmax, "yes" (with evidence head): [Athekunal/ultrabert-jev-380m-softmax-yes](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yes)
- 🤗 Softmax, "yes" − "no": [Athekunal/ultrabert-jev-380m-softmax-yesno](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yesno)
- 🤗 Sigmoid-norm (with evidence head): [Athekunal/ultrabert-jev-380m-sigmoid-norm](https://huggingface.co/Athekunal/ultrabert-jev-380m-sigmoid-norm)
- 🤗 Sqrt-softplus: [Athekunal/ultrabert-jev-380m-sqrt-softplus](https://huggingface.co/Athekunal/ultrabert-jev-380m-sqrt-softplus)

**Decoder checkpoints** (LLM-as-Jev, trained for one epoch):

- 🤗 Qwen3-0.6B: [Athekunal/llm-jev-qwen3-0.6b](https://huggingface.co/Athekunal/llm-jev-qwen3-0.6b)
- 🤗 Qwen3-4B: [Athekunal/llm-jev-qwen3-4b](https://huggingface.co/Athekunal/llm-jev-qwen3-4b)

Models that answer a question about a document by choosing one of its options, in a single scoring pass, and that show the sentences behind the choice.

Two families share one interface, `DecisionModel` (`decision_models/models/base.py`): `score(rows) -> s [K]`, `loss(row)`, `save(...)`. Training, evaluation, the benchmarks, ContextCite and the server are written against it, so they work for both.

| family | class | option k is scored by | training loss |
|---|---|---|---|
| decoder | `TrieDecoder` | log-probability of "k]" after "Best answer: [" (a candidate trie; LLM-as-Jev, arXiv 2610.02076v3) | L_tree + KL anchors |
| encoder | `MaskedEncoder` | MLM logit of " yes" (minus " no") at the mask in front of option k | softmax, sigmoid-norm or sqrt-softplus |

## Install

```bash
uv sync --extra dev                  # + --extra serve (local server), spans, profile, qwen35
```

Python 3.11+, a CUDA GPU for training. FlashAttention-2 is fetched from the Hugging Face Hub on first use.

## Quickstart

```bash
make data                                                                  # build the training mix
make train-encoder CONFIG=configs/train/ultrabert_380m_sigmoid_norm.yaml GPUS=0
make train-decoder CONFIG=configs/train/qwen3_0.6b_lora.yaml GPUS=0,1,2,3
scripts/evaluate.sh encoder runs/ultrabert_380m_lr1.5e-5_sigmoidnorm 0
decision-models-serve model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5_sigmoidnorm   # http://127.0.0.1:8767/
```

```python
from decision_models import config, models
from decision_models.data import storage

model = models.load(config.ModelRef(kind="decoder", path="Athekunal/llm-jev-qwen3-0.6b"))
rows = storage.load_examples("Athekunal/llm-jev-reason-long", split="eval")[:8]
choices = [int(s.argmax()) for s in model.score(rows)]
```

Every command takes `--config file.yaml` plus `key=value` overrides, checked against the dataclass schemas in `decision_models/config.py`.

## Docs

- [Data](docs/data.md): build the mix, load it, bring your own rows
- [Training](docs/training.md): encoder recipes, decoder (LoRA / full), evaluation
- [Inference](docs/inference.md): predict, local server, SGLang, LoRA merge
- [Citations and evidence](docs/citation.md): ContextCite (one-pass for decoders), the evidence head, span labels
- [Profiling](docs/profiling.md): torch.profiler traces and CUDA memory of a training step
- [Analyses](docs/analysis.md): latency, option coupling, AT2 (outside the package)

## Commands

| command | what |
|---|---|
| `decision-models-build-data` | build the training mix |
| `decision-models-train-encoder` / `-train-decoder` | train (run under `accelerate launch`; see `make train-*`, `scripts/train.sh`) |
| `decision-models-evaluate` / `-benchmarks` | held-out split / external benchmarks |
| `decision-models-train-evidence` | evidence head on a frozen encoder |
| `decision-models-spans` | best-of-N evidence labels from a vLLM server |
| `decision-models-predict` | decisions for a split, optional ContextCite |
| `decision-models-serve` | local web page to try a model |
| `decision-models-cite-eval` | ContextCite against gold spans |
| `decision-models-sglang` / `-merge` | SGLang client / LoRA merge |

## Layout

```
src/decision_models/
  config.py      typed configs (OmegaConf)
  data/          Example schema, sources, prompts, sampling, build pipeline
  models/        DecisionModel; TrieDecoder; MaskedEncoder; FlashAttention-2 packing
  losses/        trie loss + KL anchors; encoder recipes
  training/      one Accelerate trainer, decoder and encoder entry points
  evaluation/    held-out split and benchmarks
  citation/      ContextCite: by deletion (any model), one-pass packed (decoders), evaluation
  evidence/      sentences, metrics, evidence head, span labels
  inference/     predict, local server, SGLang client, LoRA merge
  profiling/     training-step profiler
analysis/        paper measurements (not installed)
blogs/           the blog (React; `make blog`)
configs/  scripts/  docs/  tests/
```

## Development

```bash
make test     # CPU, tiny random models, no downloads
make lint     # ruff, Google style
```

## Releases

The checkpoints and dataset are listed at the top. `scripts/release_encoders.py` builds (and with `--upload` pushes) the UltraBERT releases.

The paper's LaTeX sources are on the `ultrabert` branch.
