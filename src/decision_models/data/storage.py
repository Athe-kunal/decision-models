"""Examples as a `datasets.Dataset`: the container the pipeline maps over."""

from collections.abc import Iterable
import pathlib

import datasets
from loguru import logger

from decision_models.data import schema
from decision_models.data.sources import download

DEFAULT_SOURCE = "data/splits/reason_long_nimble"  # local copy or a Hub id

FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "source": datasets.Value("string"),
        "parts": [
            {"name": datasets.Value("string"), "text": datasets.Value("string")}
        ],
        "question": datasets.Value("string"),
        "options": [datasets.Value("string")],
        "gold": datasets.Value("int64"),
        "spans": [
            {
                "part": datasets.Value("int64"),
                "start": datasets.Value("int64"),
                "end": datasets.Value("int64"),
            }
        ],
        "group": datasets.Value("string"),
        "subset": datasets.Value("string"),
        "meta": datasets.Value("string"),  # JSON object
    }
)


def to_dataset(
    examples: Iterable[schema.Example], with_overlap_text: bool = False
) -> datasets.Dataset:
    """A Dataset with one row per example.

    Args:
      examples: The examples.
      with_overlap_text: Whether to add the `overlap_text` working column.
    """
    features = FEATURES.copy()
    if with_overlap_text:
        features["overlap_text"] = [datasets.Value("string")]
    return datasets.Dataset.from_list(
        [e.to_row(with_overlap_text) for e in examples], features=features
    )


def save(dataset: datasets.Dataset, path: str | pathlib.Path) -> None:
    """Writes a Dataset as one Parquet file, creating the parent folder."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(path))


def splits_home() -> pathlib.Path:
    """Where datasets saved by name live: <download.data_home()>/splits.

    Override the root with LLM_JEV_DATA (default: $HF_HOME/decision_models).
    """
    return download.data_home() / "splits"


def load_dataset(
    source: str = DEFAULT_SOURCE, split: str = "train"
) -> datasets.Dataset:
    """One split as a Dataset, from a local file or folder or the Hub.

    Args:
      source: A .parquet file, a folder holding <split>.parquet, the name
        of a dataset under `splits_home()`, or a Hugging Face dataset id
        (looked up in that order).
      split: The split to read (ignored for a .parquet file).

    Raises:
      FileNotFoundError: If `source` is a local folder without <split>.parquet.
    """
    root = pathlib.Path(source)
    if not root.exists() and (splits_home() / source).is_dir():
        root = splits_home() / source
    path = root / f"{split}.parquet" if root.is_dir() else root
    if path.suffix == ".parquet" and path.exists():
        dataset = datasets.Dataset.from_parquet(str(path))
        logger.info("{} examples from {}", len(dataset), path)
        return dataset
    if root.exists():
        raise FileNotFoundError(f"no {split}.parquet in {source}")
    dataset = datasets.load_dataset(source, split=split)
    if (
        "source_spans" in dataset.column_names
    ):  # the Hub release's name for spans
        dataset = dataset.rename_column("source_spans", "spans")
    logger.info(
        "{} examples from the Hub: {} ({})", len(dataset), source, split
    )
    return dataset


def load_examples(
    source: str = DEFAULT_SOURCE, split: str = "train"
) -> list[schema.Example]:
    """`load_dataset` as validated `Example`s."""
    return [schema.Example.from_row(row) for row in load_dataset(source, split)]
