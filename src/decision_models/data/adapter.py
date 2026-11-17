"""Bring your own data: an adapter yields `Record`s, `save` writes them.

To carry more than a `Record` holds (several documents, evidence spans),
subclass it and override `to_example`, or yield `schema.Example`s directly.
"""

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import pathlib
import random
import types
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from loguru import logger

from decision_models.data import schema
from decision_models.data import storage
from decision_models.data.sources import base

if TYPE_CHECKING:
    from decision_models.models import base as models_base

YES_NO = "yes/no"


@dataclasses.dataclass(frozen=True)
class Record:
    """One decision; subclass to add fields or `option_sets`.

    Attributes:
      documents: The input text ("" if the question stands alone).
      question: What to decide.
      options: A name in `option_sets` (e.g. `YES_NO`) or at least two texts.
      answer: The correct option's 0-based index or text (case-insensitive).
    """

    option_sets: ClassVar[Mapping[str, tuple[str, ...]]] = (
        types.MappingProxyType({YES_NO: ("Yes", "No")})
    )

    documents: str
    question: str
    options: str | Sequence[str]
    answer: int | str

    def to_example(self, source: str) -> schema.Example:
        """The record as a validated `Example` of dataset `source`.

        Raises:
          KeyError: If `options` names no option set.
          ValueError: If the answer is a bool or names no option, or several.
        """
        if (
            isinstance(self.options, str)
            and self.options not in self.option_sets
        ):
            raise KeyError(
                f"{self.options!r} is not in {sorted(self.option_sets)}"
            )
        options = (
            self.option_sets[self.options]
            if isinstance(self.options, str)
            else tuple(self.options)
        )
        folded = [o.strip().casefold() for o in options]
        gold = self.answer
        if isinstance(gold, str) and folded.count(gold.strip().casefold()) == 1:
            gold = folded.index(gold.strip().casefold())
        elif isinstance(gold, (str, bool)):
            raise ValueError(f"answer {gold!r} must name one of {options}")
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


def _example(row: "Record | schema.Example", source: str) -> schema.Example:
    return row if isinstance(row, schema.Example) else row.to_example(source)


def save(
    splits: Mapping[str, Iterable["Record | schema.Example"]],
    source: str,
    out: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Writes <out>/<split>.parquet per split; train with data.source=<source>.

    A row whose id (same documents, question and options) or group was
    already seen, in this split or an earlier one, is dropped with a warning.

    Args:
      splits: Rows per split, e.g. {"train": ..., "eval": ...}.
      source: The dataset name.
      out: The folder; default: `storage.splits_home()/<source>`.

    Returns:
      The folder written.

    Raises:
      ValueError: If a split has no rows left.
    """
    seen: dict[str, str] = {}  # id or group -> the split that has it
    built: dict[str, list[schema.Example]] = {}
    for split, rows in splits.items():
        built[split] = []
        for e in (_example(row, source) for row in rows):
            first = seen.get(e.id) or seen.get(e.group or e.id, split)
            if e.id in seen or first != split:
                logger.warning(
                    "dropped {} from {}: already in {}", e.id, split, first
                )
                continue
            seen[e.id] = seen[e.group or e.id] = split
            built[split].append(e)
        if not built[split]:
            raise ValueError(f"split {split!r} has no rows")
    folder = pathlib.Path(out) if out else storage.splits_home() / source
    for split, examples in built.items():
        storage.save(storage.to_dataset(examples), folder / f"{split}.parquet")
    return folder


def collate(
    rows: Iterable["Record | schema.Example"],
    model: "models_base.DecisionModel",
    rng: random.Random | None = None,
    source: str = "custom",
) -> list[Any]:
    """The rows encoded as `model` reads them; `rng` shuffles the options."""
    return [model.encode(_example(row, source), rng) for row in rows]
