"""Bring your own data: an adapter yields `Record`s, `save` writes them."""

from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
import pathlib
import random
from typing import Any
from typing import Literal

from decision_models.data import schema
from decision_models.data import storage
from decision_models.data.sources import base

YES_NO = "yes/no"


@dataclasses.dataclass(frozen=True)
class Record:
    """One decision.

    Attributes:
      documents: The input text ("" if the question stands alone).
      question: What to decide.
      options: `YES_NO` (shown as "Yes" / "No") or at least two option texts.
      answer: The correct option: its 0-based index or its text.
    """

    documents: str
    question: str
    options: Literal["yes/no"] | Sequence[str]
    answer: int | str


def to_example(record: Record, source: str) -> schema.Example:
    """The validated `Example` of a record; raises ValueError if invalid."""
    options = (
        ("Yes", "No") if record.options == YES_NO else tuple(record.options)
    )
    gold = record.answer
    if isinstance(gold, str):
        folded = [o.casefold() for o in options]
        if gold.casefold() not in folded:
            raise ValueError(f"answer {gold!r} is not one of {list(options)}")
        gold = folded.index(gold.casefold())
    key = base.stable_hash(record.documents, record.question, *options)
    return schema.Example(
        id=f"{source}:{key}",
        source=source,
        parts=(schema.Part("Document", record.documents),)
        if record.documents
        else (),
        question=record.question,
        options=options,
        gold=gold,
        group=f"{source}:{key}",
    )


def save(
    splits: dict[str, Iterable[Record]], out: str | pathlib.Path, source: str
) -> None:
    """Writes <out>/<split>.parquet per split; train with data.source=<out>."""
    for split, records in splits.items():
        rows = [to_example(r, source) for r in records]
        storage.save(
            storage.to_dataset(rows), pathlib.Path(out) / f"{split}.parquet"
        )


def collate(
    records: Iterable[Record],
    model: Any,
    rng: random.Random | None = None,
    source: str = "custom",
) -> list[Any]:
    """Records encoded as `model` (MaskedEncoder or TrieDecoder) trains on them.

    With `rng` the options are shuffled, as in training.
    """
    return [model.encode(to_example(r, source), rng) for r in records]
