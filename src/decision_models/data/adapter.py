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
    """One decision.

    Subclass to add fields (override `to_example`) or named option sets
    (override `option_sets`).

    Attributes:
      documents: The input text ("" if the question stands alone).
      question: What to decide.
      options: A name in `option_sets` (e.g. `YES_NO`) or at least two texts.
      answer: The correct option: its 0-based index or its text
        (case-insensitive).
      option_sets: Named option sets `options` may give instead of texts.
    """

    option_sets: ClassVar[Mapping[str, tuple[str, ...]]] = (
        types.MappingProxyType({YES_NO: ("Yes", "No")})
    )

    documents: str
    question: str
    options: str | Sequence[str]
    answer: int | str

    def option_texts(self) -> tuple[str, ...]:
        """The options as shown.

        Raises:
          ValueError: If `options` names no option set.
        """
        if not isinstance(self.options, str):
            return tuple(self.options)
        if self.options not in self.option_sets:
            raise ValueError(
                f"options {self.options!r} is not one of "
                f"{sorted(self.option_sets)}"
            )
        return self.option_sets[self.options]

    def to_example(self, source: str) -> schema.Example:
        """The record as a validated `Example`.

        Args:
          source: The dataset name, also the prefix of the id and group.

        Raises:
          ValueError: If the options or the answer are invalid.
        """
        options = self.option_texts()
        key = base.stable_hash(self.documents, self.question, *options)
        parts = (
            (schema.Part("Document", self.documents),) if self.documents else ()
        )
        return schema.Example(
            id=f"{source}:{key}",
            source=source,
            parts=parts,
            question=self.question,
            options=options,
            gold=_gold_index(self.answer, options),
            group=f"{source}:{key}",
        )


Row = Record | schema.Example


def _gold_index(answer: int | str, options: Sequence[str]) -> int:
    """The index of `answer` (an index or a case-insensitive option text)."""
    if isinstance(answer, bool):  # A bool is an int: True would mean "No".
        raise ValueError(f"answer {answer!r}: give an index or option text")
    if isinstance(answer, int):
        return answer  # `schema.Example` checks the range.
    folded = [o.casefold() for o in options]
    try:
        return folded.index(answer.casefold())
    except ValueError:
        raise ValueError(
            f"answer {answer!r} is not one of {list(options)}"
        ) from None


def to_example(row: Row, source: str) -> schema.Example:
    """`row.to_example(source)`; an `Example` is returned unchanged."""
    if isinstance(row, schema.Example):
        return row
    return row.to_example(source)


def save(
    splits: Mapping[str, Iterable[Row]],
    source: str,
    out: str | pathlib.Path | None = None,
) -> pathlib.Path:
    """Writes each split as <out>/<split>.parquet.

    Train on it with `data.source=<source>` (or `data.source=<out>` when
    `out` is given).

    Args:
      splits: Rows per split name, e.g. {"train": ..., "eval": ...}.
      source: The dataset name given to records.
      out: The folder to write; default: `storage.splits_home()/<source>`,
        outside the workspace.

    Returns:
      The folder written.

    Raises:
      ValueError: If a record is invalid.
    """
    folder = pathlib.Path(out) if out else storage.splits_home() / source
    for split, rows in splits.items():
        examples = [to_example(row, source) for row in rows]
        storage.save(storage.to_dataset(examples), folder / f"{split}.parquet")
    return folder


def collate(
    rows: Iterable[Row],
    model: "models_base.DecisionModel",
    rng: random.Random | None = None,
    source: str = "custom",
) -> list[Any]:
    """The rows encoded as `model` reads them in training.

    Args:
      rows: Records or examples.
      model: Any decision model (`MaskedEncoder`, `TrieDecoder`, ...).
      rng: Shuffles the options, as in training; None keeps their order.
      source: The dataset name given to records.

    Returns:
      One encoding per row, of the model's type (e.g. `MaskEncoding`).

    Raises:
      ValueError: If a record is invalid.
    """
    return [model.encode(to_example(row, source), rng) for row in rows]
