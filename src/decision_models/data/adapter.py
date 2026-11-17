"""Bring your own data: an adapter yields `Record`s, a collator formats them.

A `Record` is the smallest thing a dataset has to say: the document(s), the
question, the options (a list, or `YES_NO`) and the answer. An adapter is any
function (or iterable) that yields records from your files:

    def my_adapter(path):
        for r in read_jsonl(path):
            yield adapter.Record(documents=r["context"], question=r["q"],
                                 options=adapter.YES_NO, answer=r["label"])

`to_examples` turns records into validated `schema.Example`s, `save` writes
them as <out>/<split>.parquet (train with `data.source=<out>`), and
`Collator` formats them exactly as training does, for an encoder or a
decoder (`models.encoder.encode` / `models.decoder.encode`).
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import numbers
import pathlib
import random
from typing import Any
from typing import Literal

from decision_models import config
from decision_models.data import schema
from decision_models.data import storage

YES_NO = "yes/no"
YES_NO_OPTIONS = ("Yes", "No")  # The wording the BoolQ benchmark uses.

Options = Literal["yes/no"] | Sequence[str]
Answer = int | str | bool


def _hash(*parts: Any) -> str:
    from decision_models.data.sources import base

    return base.stable_hash(*parts)


@dataclasses.dataclass(frozen=True)
class Record:
    """One decision, as an adapter returns it.

    Attributes:
      documents: The input text: one string, or several (each becomes its
        own part, "Document 1", "Document 2", ...). Empty if the question
        is self-contained.
      question: What to decide.
      options: `YES_NO` (shown as "Yes" / "No") or at least two option texts.
      answer: The correct option: its 0-based index, its text, or, for
        `YES_NO`, a bool (True = "Yes"). Texts match case-insensitively.
      id: Unique within the dataset; default: a hash of the record.
      evidence: Optional quotes from the documents that support the answer;
        each becomes a span where it is found (quotes not found are ignored).
      group: Optional leakage unit (e.g. the document id).
    """

    documents: str | Sequence[str]
    question: str
    options: Options
    answer: Answer
    id: str | None = None
    evidence: Sequence[str] = ()
    group: str = ""

    @property
    def documents_list(self) -> list[str]:
        """The documents as a list, empty strings left out."""
        docs = (
            [self.documents]
            if isinstance(self.documents, str)
            else list(self.documents)
        )
        return [d for d in docs if d]

    @property
    def option_texts(self) -> tuple[str, ...]:
        """The options as shown."""
        if isinstance(self.options, str):
            if self.options.strip().lower() != YES_NO:
                raise ValueError(
                    f"options={self.options!r}: give a list of options or "
                    f"{YES_NO!r}"
                )
            return YES_NO_OPTIONS
        return tuple(self.options)

    @property
    def gold(self) -> int:
        """The 0-based index of the answer among `option_texts`.

        Raises:
          ValueError: If the answer names no option.
        """
        options = self.option_texts
        answer = self.answer
        if isinstance(answer, bool):
            if options != YES_NO_OPTIONS:
                raise ValueError(
                    f"answer={answer}: a bool answer needs options={YES_NO!r}"
                )
            return 0 if answer else 1
        if isinstance(answer, numbers.Integral):
            answer = int(answer)
            if not 0 <= answer < len(options):
                raise ValueError(
                    f"answer index {answer} not in 0..{len(options) - 1}"
                )
            return answer
        folded = [o.strip().casefold() for o in options]
        try:
            return folded.index(str(answer).strip().casefold())
        except ValueError:
            raise ValueError(
                f"answer {answer!r} is not one of the options {list(options)}"
            ) from None


def to_example(record: Record, source: str) -> schema.Example:
    """The validated `Example` of a record.

    Raises:
      ValueError: If the record has fewer than two options or its answer
        names no option.
    """
    docs = record.documents_list
    names = (
        ["Document"]
        if len(docs) == 1
        else [f"Document {i + 1}" for i in range(len(docs))]
    )
    spans = []
    for quote in record.evidence:
        for p, text in enumerate(docs):
            start = text.find(quote) if quote else -1
            if start >= 0:
                spans.append(schema.Span(p, start, start + len(quote)))
                break
    options = record.option_texts
    key = record.id or _hash(*docs, record.question, *options)
    return schema.Example(
        id=f"{source}:{key}",
        source=source,
        parts=tuple(
            schema.Part(n, t) for n, t in zip(names, docs, strict=True)
        ),
        question=record.question,
        options=options,
        gold=record.gold,
        spans=tuple(sorted(set(spans), key=lambda s: (s.part, s.start))),
        group=f"{source}:{record.group or key}",
        subset=str(record.gold + 1),
    )


def to_examples(
    records: Iterable[Record | schema.Example], source: str
) -> list[schema.Example]:
    """`to_example` of every record (Examples pass through unchanged).

    Raises:
      ValueError: If a record is invalid (the message names its position)
        or two records share an id.
    """
    out: list[schema.Example] = []
    for i, record in enumerate(records):
        if isinstance(record, schema.Example):
            out.append(record)
            continue
        try:
            out.append(to_example(record, source))
        except ValueError as e:
            raise ValueError(f"{source} record {i}: {e}") from None
    ids = [e.id for e in out]
    if len(set(ids)) != len(ids):
        dup = next(i for i in ids if ids.count(i) > 1)
        raise ValueError(f"{source}: duplicate id {dup!r}")
    return out


def save(
    splits: Mapping[str, Iterable[Record | schema.Example]],
    out: str | pathlib.Path,
    source: str,
) -> dict[str, int]:
    """Writes each split as <out>/<split>.parquet; returns the row counts.

    Train on it with `data.source=<out>` (splits "train" and "eval" by
    default), or evaluate with `rows.source=<out>`.
    """
    counts: dict[str, int] = {}
    for split, records in splits.items():
        examples = to_examples(records, source)
        storage.save(
            storage.to_dataset(examples),
            pathlib.Path(out) / f"{split}.parquet",
        )
        counts[split] = len(examples)
    return counts


@dataclasses.dataclass(frozen=True)
class Collated:
    """One record, formatted as the training pipeline formats it.

    Attributes:
      example: The validated row.
      text: The prompt text, options in the shown order.
      encoding: `encoder.MaskEncoding` or `decoder.TrieEncoding`.
    """

    example: schema.Example
    text: str
    encoding: Any

    @property
    def input_ids(self) -> list[int]:
        """The token ids the model reads."""
        return self.encoding.input_ids

    @property
    def gold(self) -> int:
        """The gold option's index in the shown order."""
        return self.encoding.gold


class Collator:
    """Formats records for an encoder or a decoder, as training does.

    Build it from a loaded model (`Collator(model)`, a `MaskedEncoder` or a
    `TrieDecoder`) or from a tokenizer alone, without weights
    (`Collator.for_encoder(tok, fmt)`, `Collator.for_decoder(tok)`).
    Calling it on records returns one `Collated` per record; with `rng`
    the options are shuffled as in training, without it they stay in order.

    Attributes:
      kind: "encoder" or "decoder".
      tokenizer: The model's tokenizer.
      fmt: The encoder's prompt format (None for a decoder).
      source: The `source` given to records without one.
    """

    def __init__(
        self,
        model: Any = None,
        *,
        kind: Literal["encoder", "decoder"] | None = None,
        tokenizer: Any = None,
        fmt: config.FormatConfig | None = None,
        source: str = "custom",
    ) -> None:
        if model is not None:
            from decision_models.models import encoder

            kind = (
                "encoder"
                if isinstance(model, encoder.MaskedEncoder)
                else "decoder"
            )
            tokenizer = model.tokenizer
            fmt = getattr(model, "fmt", None)
        if kind not in ("encoder", "decoder") or tokenizer is None:
            raise ValueError(
                "give a model, or kind='encoder'|'decoder' and a tokenizer"
            )
        self.kind = kind
        self.tokenizer = tokenizer
        self.fmt = fmt or (config.FormatConfig() if kind == "encoder" else None)
        self.source = source

    @classmethod
    def for_encoder(
        cls,
        tokenizer: Any,
        fmt: config.FormatConfig | None = None,
        source: str = "custom",
    ) -> "Collator":
        """A collator for a masked-LM encoder (default `FormatConfig`)."""
        return cls(kind="encoder", tokenizer=tokenizer, fmt=fmt, source=source)

    @classmethod
    def for_decoder(cls, tokenizer: Any, source: str = "custom") -> "Collator":
        """A collator for a causal-LM decoder (its chat template)."""
        return cls(kind="decoder", tokenizer=tokenizer, source=source)

    def collate_one(
        self,
        record: Record | schema.Example,
        rng: random.Random | None = None,
    ) -> Collated:
        """One record, formatted; with `rng` its options are shuffled."""
        example = (
            record
            if isinstance(record, schema.Example)
            else to_example(record, self.source)
        )
        if self.kind == "encoder":
            from decision_models.models import encoder

            assert self.fmt is not None
            enc = encoder.encode(example, self.tokenizer, self.fmt, rng)
            text, _ = encoder.build_prompt(
                example, enc.order, self.tokenizer, self.fmt
            )
            return Collated(example, text, enc)
        from decision_models.models import decoder

        state = rng.getstate() if rng is not None else None
        enc = decoder.encode(example, self.tokenizer, rng)
        order = list(range(len(example.options)))
        if state is not None:  # replay the shuffle `encode` drew
            replay = random.Random()
            replay.setstate(state)
            replay.shuffle(order)
        text, _ = decoder.build_prompt(example, order, self.tokenizer)
        return Collated(example, text, enc)

    def __call__(
        self,
        records: Iterable[Record | schema.Example],
        rng: random.Random | None = None,
    ) -> list[Collated]:
        """Every record, formatted (shared `rng` for the shuffles)."""
        return [self.collate_one(r, rng) for r in records]
