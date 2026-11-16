# decision-models

Models that pick one option for a question about a document in a single forward pass, and show the sentences behind the pick. Results and figures are in the [blog post](https://athe-kunal.github.io/decision-models/blog/).

| family | class | score of option k | loss |
|---|---|---|---|
| decoder | `TrieDecoder` | log p("k]" \| prompt … "Best answer: [") | L_tree + KL anchors (LLM-as-Jev) |
| encoder | `MaskedEncoder` | ℓ(" yes") [− ℓ(" no")] at the mask in front of option k | softmax, sigmoid-norm or sqrt-softplus over the options |

## Install

```bash
uv sync --extra dev          # add --extra qwen35 (Qwen3.5), serve (local page), spans (labels), profile
```

Python 3.11+ and an NVIDIA GPU for training. Or use [Docker](docker.md): `make docker-build TARGET=encoder && make docker-run TARGET=encoder`.

## Quickstart

`make data                                                                         # or use data.source=Athekunal/llm-jev-reason-long`<br>
`make train-encoder CONFIG=`[`configs/train/ultrabert_380m_sigmoid_norm.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/ultrabert_380m_sigmoid_norm.yaml)` GPUS=0`<br>
`make train-decoder CONFIG=`[`configs/train/qwen3_0.6b_lora.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/train/qwen3_0.6b_lora.yaml)` GPUS=0,1,2,3`<br>
`scripts/evaluate.sh encoder runs/ultrabert_380m_lr1.5e-5_sigmoidnorm 0`<br>
`decision-models-predict model.kind=encoder model.path=runs/ultrabert_380m_lr1.5e-5_sigmoidnorm out=runs/predict/eval.jsonl`

```python
from decision_models import config, models
from decision_models.data import storage

model = models.load(config.ModelRef(kind="decoder", path="Athekunal/llm-jev-qwen3-0.6b"))
rows = storage.load_examples("Athekunal/llm-jev-reason-long", split="eval")[:8]
choices = [int(s.argmax()) for s in model.score(rows)]
```

Every command takes `--config file.yaml` and `key=value` overrides. The keys are the dataclass fields in `src/decision_models/config.py`.

## Guides

| page | covers |
|---|---|
| [Data](data.md) | the row format, what the encoder and decoder see, building the mix, your own data |
| [Training](training.md) | configs, every argument, outputs, evaluation |
| [Inference](inference.md) | Python, JSONL predictions, local page, SGLang |
| [Citations and evidence](citation.md) | ContextCite, the evidence head, generating evidence labels |
| [Docker](docker.md) | images per model family, GPU generations |

## Releases

- Dataset: [Athekunal/llm-jev-reason-long](https://huggingface.co/datasets/Athekunal/llm-jev-reason-long) (ReClor withheld)
- Encoders (UltraBERT-380M): [softmax "yes"](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yes), [softmax "yes" − "no"](https://huggingface.co/Athekunal/ultrabert-jev-380m-softmax-yesno), [sigmoid-norm](https://huggingface.co/Athekunal/ultrabert-jev-380m-sigmoid-norm), [sqrt-softplus](https://huggingface.co/Athekunal/ultrabert-jev-380m-sqrt-softplus)
- Decoders: [Qwen3-0.6B](https://huggingface.co/Athekunal/llm-jev-qwen3-0.6b), [Qwen3-4B](https://huggingface.co/Athekunal/llm-jev-qwen3-4b)
