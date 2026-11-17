"""Bring your own data: an adapter yields `Record`s, `save` writes them.

To carry more than a `Record` holds (several documents, evidence spans),
subclass it and override `to_example`, or yield `schema.Example`s directly.
"""

from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
import pathlib
import random
from typing import Any

from decision_models.data import schema
from decision_models.data import storage
from decision_models.data.sources import base

YES_NO = "yes/no"
# Named option sets `Record.options` may give instead of a list.
OPTION_SETS: dict[str, tuple[str, ...]] = {YES_NO: ("Yes", "No")}


@dataclasses.dataclass(frozen=True)
class Record:
    """One decision.

    Attributes:
      documents: The input text ("" if the question stands alone).
      question: What to decide.
      options: A key of `OPTION_SETS` (e.g. `YES_NO`) or at least two texts.
      answer: The correct option: its 0-based index or its text.
    """

    documents: str
    question: str
    options: str | Sequence[str]
    answer: int | str

    def to_example(self, source: str) -> schema.Example:
        """The validated `Example`; raises ValueError if invalid."""
        if not isinstance(self.options, str):
            options = tuple(self.options)
        elif self.options in OPTION_SETS:
            options = OPTION_SETS[self.options]
        else:
            raise ValueError(f"options {self.options!r} not in {OPTION_SETS}")
        gold = self.answer
        if isinstance(gold, str):
            folded = [o.casefold() for o in options]
            if gold.casefold() not in folded:
                raise ValueError(
                    f"answer {gold!r} is not one of {list(options)}"
                )
            gold = folded.index(gold.casefold())
        key = base.stable_hash(self.documents, self.question, *options)
        return schema.Example(
            id=f"{source}:{key}",
            source=source,
            parts=(schema.Part("Document", self.documents),)
            if self.documents
            else (),
            question=self.question,
            options=options,
            gold=gold,
            group=f"{source}:{key}",
        )


Row = Record | schema.Example


def to_example(row: Row, source: str) -> schema.Example:
    """`row.to_example(source)`; Examples pass through."""
    return row if isinstance(row, schema.Example) else row.to_example(source)


def save(
    splits: dict[str, Iterable[Row]], out: str | pathlib.Path, source: str
) -> None:
    """Writes <out>/<split>.parquet per split; train with data.source=<out>."""
    for split, rows in splits.items():
        examples = [to_example(r, source) for r in rows]
        storage.save(
            storage.to_dataset(examples),
            pathlib.Path(out) / f"{split}.parquet",
        )


def collate(
    rows: Iterable[Row],
    model: Any,
    rng: random.Random | None = None,
    source: str = "custom",
) -> list[Any]:
    """Rows encoded as `model` (any `DecisionModel`) trains on them.

    With `rng` the options are shuffled, as in training.
    """
    return [model.encode(to_example(r, source), rng) for r in rows]
