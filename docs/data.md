# Data

Every dataset is stored as rows of one type, `schema.Example` (`src/decision_models/data/schema.py`). Encoders and decoders train on the same rows and differ only in how they turn a row into tokens.

## The row

| field | type | meaning |
|---|---|---|
| `id` | str | unique within its dataset |
| `source` | str | dataset name (per-dataset metrics group by it) |
| `parts` | list of `Part(name, text)` | the input text, in prompt order |
| `question` | str | what to decide |
| `options` | list of str | at least 2; option `i` is shown as `[i+1]` |
| `gold` | int | 0-based index of the correct option |
| `spans` | list of `Span(part, start, end)` | optional: characters `[start, end)` of `parts[part]` that support the gold |
| `group`, `subset`, `meta` | str, str, JSON | optional: leakage unit, stratum, provenance |

A short row from the eval split:

```python
Example(
    id="anli:451999a5f0cc679f", source="anli",
    parts=(Part("Observation 1", "Noah had a cold."),
           Part("Observation 2", "After two days, Noah felt better.")),
    question="Which hypothesis best explains what happened between the two observations?",
    options=("Noah ate cold cuts and got a lot of rest.",
             "Noah took his prescribed medicine and stayed home."),
    gold=1,
)
```

A row with evidence (ContractNLI). The span points into the contract text:

```python
row.question  # 'Statement: "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated." ...'
row.options   # ('The contract entails the statement.', 'The contract contradicts the statement.', 'The contract does not mention it.')
row.gold      # 0
row.spans     # (Span(part=0, start=1354, end=1637),)
row.parts[0].text[1354:1637]
# 'Listing Broker, Prospect, and Prospect’s Broker agree not to disclose to any other person the fact that ...'
```

## What the decoder sees

`models/decoder.py:build_prompt` renders `prompts/decision.j2` in the chat template and prefills the answer up to `Best answer: [` (Qwen3, thinking off):

```text
<|im_start|>user
State: Noah had a cold.

After two days, Noah felt better.

Which hypothesis best explains what happened between the two observations?
Options:
[1] Noah ate cold cuts and got a lot of rest.
[2] Noah took his prescribed medicine and stayed home.
Select one option. Answer only with its bracketed numeric identifier.<|im_end|>
<|im_start|>assistant
<think>

</think>

Best answer: [
```

Option k is the token suffix `k]`. All suffixes form a trie; the score of option k is

```text
s_k = Σ_t log p(suffix_k[t] | prompt, suffix_k[<t])
```

For this row the suffixes are `['1', ']']` and `['2', ']']`, and training teacher-forces the gold one (`2]`). With 12 options, Qwen splits digits, so the trie branches twice:

```text
(root) -> 1 2 3 4 5 6 7 8 9
"1"    -> 0 1 2 ]          # "1]", "10]", "11]", "12]"
"2".."9", "10".."12" -> ]
```

```python
from decision_models.models import decoder

enc = decoder.encode(row, tokenizer)            # options in release order
enc.input_ids                                   # prompt + "2]"
enc.path                                        # [(82, legal ['1','2'] ids, gold '2' id), (83, [']'], ']')]
```

## What the encoder sees

`models/encoder.py:build_prompt` renders `prompts/decision_mlm.j2` with one mask in front of every option (UltraBERT):

```text
State: Noah had a cold.

After two days, Noah felt better.

Which hypothesis best explains what happened between the two observations?
Mark the correct option with yes and every other option with no.
Options:
[1] <|mask|> Noah ate cold cuts and got a lot of rest.
[2] <|mask|> Noah took his prescribed medicine and stayed home.
```

One forward pass gives the MLM logits ℓ at every mask, and the score of option k is

```text
s_k = ℓ_k(" yes")  [ − ℓ_k(" no")  when format.no_token: " no" ]
```

 The loss normalises s over the K options ([training](training.md#encoder-model-format-optimloss)).

```python
from decision_models import config
from decision_models.models import encoder

enc = encoder.encode(row, tokenizer, config.FormatConfig(no_token=" no"))
enc.masks     # [52, 68]: token position of each option's mask
enc.gold      # 1
```

During training both models shuffle the options per row (`encode(..., rng)`), so the gold position is uniform.

## Build the training mix

`make data          # download the pinned sources, then build data/splits/reason_long_nimble`<br>
`decision-models-build-data --config `[`configs/data/reason_long_nimble.yaml`](https://github.com/Athe-kunal/decision-models/blob/main/configs/data/reason_long_nimble.yaml)` \`<br>
`    datasets=[musique,contractnli] per_dataset=500 out=data/splits/small`

| argument | default | meaning |
|---|---|---|
| `out` | `data/splits/reason_long_nimble` | writes `train.parquet`, `eval.parquet`, `manifest.json` |
| `datasets` | all 13 | subset of clinc150 massive bitext reclor logiqa2 cosmosqa piqa anli proofwriter contractnli musique quality nimble |
| `per_dataset` | 1000 | training rows per dataset |
| `eval_rows` | 500 | held-out rows per dataset |
| `max_tokens` | 10240 | longer prompts are dropped, never truncated |
| `length_tokenizer` | `Qwen/Qwen3.5-4B` | measures `max_tokens` |
| `download_only` | false | only fetch the sources |

Or skip the build and train on the released mix (ReClor withheld): `data.source=Athekunal/llm-jev-reason-long`.

## Your own data

Write an adapter that yields `adapter.Record(documents, question, options, answer)`:

| field | type | meaning |
|---|---|---|
| `documents` | str | the input text; `""` if the question stands alone |
| `question` | str | what to decide |
| `options` | list of str, or a name in `Record.option_sets` | `adapter.YES_NO` is shown as `Yes` / `No` |
| `answer` | int or str | the correct option's 0-based index, or its text (case-insensitive); not a bool |

```python
import json
from decision_models.data import adapter

def my_adapter(path):
    for line in open(path):
        r = json.loads(line)
        yield adapter.Record(r["context"], r["question"], adapter.YES_NO, r["label"])

adapter.save(
    {"train": my_adapter("train.jsonl"), "eval": my_adapter("dev.jsonl")},
    out="data/splits/mine", source="mine",
)
```

Then train with `data.source=data/splits/mine` (or evaluate with `rows.source=...`). An invalid record raises `ValueError`.

To see what a model is trained on, `adapter.collate(records, model)` returns the encodings of a `MaskedEncoder` or a `TrieDecoder`, the same ones training uses (pass `rng` to shuffle options as training does).

To extend it:

- **More fields** (several documents, evidence spans, metadata): subclass `Record` and override `to_example(source)`, or yield `schema.Example`s; `save` and `collate` accept both.
- **Another named option set**: subclass `Record` and extend `option_sets`:

  ```python
  @dataclasses.dataclass(frozen=True)
  class MyRecord(adapter.Record):
      option_sets = types.MappingProxyType(
          {**adapter.Record.option_sets, "true/false": ("True", "False")}
      )
  ```
- **Another model family**: implement `DecisionModel.encode`; `collate` and training call it.

### Add it to the build instead

To mix your dataset into `make data` (deduplication, benchmark-overlap filter, sampling), register a source. `sources/multiple_choice.py` is the pattern: `base.make_example` gives stable ids and groups, and spans are given as `(part name, start, end)`.

```python
# src/decision_models/data/sources/mine.py
from decision_models.data.sources import base

@base.register("mine")
def mine() -> base.Split:
    def row(r):
        return base.make_example(
            "mine", key=r["id"], subset="", group=r["doc_id"],
            parts=[("Document", r["context"])], question=r["question"],
            options=r["choices"], gold=r["label"],
            spans=[("Document", a, b) for a, b in r.get("spans", [])],
        )
    return {"train": [row(r) for r in my_reader("train")],      # your file reader
            "validation": [row(r) for r in my_reader("dev")]}
```

Import the module in `sources/__init__.py`, then `decision-models-build-data ... datasets=[mine]`.
