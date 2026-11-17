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
          ValueError: If the answer names no option.
        """
        options = (
            self.option_sets[self.options]
            if isinstance(self.options, str)
            else tuple(self.options)
        )
        folded = [o.casefold() for o in options]
        answer = self.answer
        if isinstance(answer, bool) or (
            isinstance(answer, str) and answer.casefold() not in folded
        ):
            raise ValueError(f"answer {answer!r} is not one of {options}")
        key = base.stable_hash(self.documents, self.question, *options)
        return schema.Example(
            id=f"{source}:{key}",
            source=source,
            parts=(schema.Part("Document", self.documents),)
            if self.documents
            else (),
            question=self.question,
            options=options,
            gold=folded.index(answer.casefold())
            if isinstance(answer, str)
            else answer,
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

    Args:
      splits: Rows per split, e.g. {"train": ..., "eval": ...}.
      source: The dataset name.
      out: The folder; default: `storage.splits_home()/<source>`.

    Returns:
      The folder written.
    """
    folder = pathlib.Path(out) if out else storage.splits_home() / source
    for split, rows in splits.items():
        examples = [_example(row, source) for row in rows]
        storage.save(storage.to_dataset(examples), folder / f"{split}.parquet")
    return folder


def collate(
    rows: Iterable["Record | schema.Example"],
    model: "models_base.DecisionModel",
    rng: random.Random | None = None,
    source: str = "custom",
) -> list[Any]:
    """The rows encoded as `model` reads them; `rng` shuffles as training."""
    return [model.encode(_example(row, source), rng) for row in rows]
