# decision-models

Models that answer a question about a document by choosing one of its options, in a single scoring pass, and that show the sentences behind the choice.

**[Read the blog post](blog/)** for the results, with interactive figures.

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

## Guides

- [Data](data.md): build the mix, load it, bring your own rows
- [Training](training.md): encoder recipes, decoder (LoRA / full), evaluation
- [Inference](inference.md): predict, local server, SGLang, LoRA merge
- [Citations and evidence](citation.md): ContextCite (one-pass for decoders), the evidence head, span labels
- [Profiling](profiling.md): torch.profiler traces and CUDA memory of a training step
- [Analyses](analysis.md): latency, option coupling, AT2 (outside the package)

## Releases

- Dataset (training mixture, held-out rows and evidence labels): [Athekunal/llm-jev-reason-long](https://huggingface.co/datasets/Athekunal/llm-jev-reason-long)
- Encoders (UltraBERT-380M, Apache-2.0): [softmax "yes"](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yes), [softmax "yes" − "no"](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yesno), [sigmoid-norm](https://huggingface.co/Athekunal/ultrabert-jev-380m-sigmoid-norm), [sqrt-softplus](https://huggingface.co/Athekunal/ultrabert-jev-380m-sqrt-softplus)
- Decoders (LLM-as-Jev, one epoch): [Qwen3-0.6B](https://huggingface.co/Athekunal/llm-jev-qwen3-0.6b), [Qwen3-4B](https://huggingface.co/Athekunal/llm-jev-qwen3-4b)
