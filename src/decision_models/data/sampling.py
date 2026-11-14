"""Seeded row selections shared by training, evaluation and the analyses."""

import collections
from collections.abc import Callable
from collections.abc import Sequence
import json
import pathlib
import random

from decision_models import config
from decision_models.data import schema
from decision_models.data import storage

DEV_PER = 100  # Rows per dataset of the dev sample settings were chosen on.
DEV_SEED = config.SEED

Predicate = Callable[[schema.Example], bool]


def by_source(
    rows: Sequence[schema.Example],
) -> dict[str, list[schema.Example]]:
    """Rows grouped by their source dataset, in first-seen order."""
    groups: dict[str, list[schema.Example]] = collections.defaultdict(list)
    for row in rows:
        groups[row.source].append(row)
    return groups


def per_dataset(
    rows: Sequence[schema.Example], per: int, seed: int
) -> list[schema.Example]:
    """A fixed sample of up to `per` rows from each source dataset.

    Datasets are drawn in sorted order from one `random.Random(seed)`.
    """
    groups = by_source(rows)
    rng = random.Random(seed)
    out: list[schema.Example] = []
    for name in sorted(groups):
        out += rng.sample(groups[name], min(per, len(groups[name])))
    return out


def select_part(
    rows: Sequence[schema.Example], part: config.Part = "all"
) -> list[schema.Example]:
    """One part of the held-out split.

    "dev" is `per_dataset(rows, DEV_PER, DEV_SEED)`, the sample settings were
    chosen on; "test" is every other row, never used for a choice; "all" is
    every row.
    """
    if part == "all":
        return list(rows)
    dev = per_dataset(rows, DEV_PER, DEV_SEED)
    if part == "dev":
        return dev
    dev_ids = {r.id for r in dev}
    return [r for r in rows if r.id not in dev_ids]


def select_rows(
    rows: Sequence[schema.Example],
    cfg: config.RowsConfig,
    where: Predicate | None = None,
) -> list[schema.Example]:
    """`cfg`'s datasets and part of `rows`, filtered by `where`, then `per`."""
    rows = select_part(rows, cfg.part)
    rows = [
        r
        for r in rows
        if (cfg.datasets is None or r.source in cfg.datasets)
        and (where is None or where(r))
    ]
    return rows if cfg.per is None else per_dataset(rows, cfg.per, cfg.seed)


def select(
    cfg: config.RowsConfig, where: Predicate | None = None
) -> list[schema.Example]:
    """Loads `cfg.source` / `cfg.split` and selects rows as `select_rows`."""
    return select_rows(storage.load_examples(cfg.source, cfg.split), cfg, where)


def read_ids(path: str | pathlib.Path | None) -> set[str] | None:
    """The row ids in a JSON list file, or None without a file."""
    if path is None:
        return None
    return set(json.loads(pathlib.Path(path).read_text()))


def balanced(
    rows: Sequence[schema.Example], n: int, seed: int
) -> list[schema.Example]:
    """`n` rows taken round-robin over the datasets after a seeded shuffle."""
    groups = by_source(rows)
    rng = random.Random(seed)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    out: list[schema.Example] = []
    i = 0
    while len(out) < n and any(i < len(v) for v in groups.values()):
        out += [
            groups[name][i] for name in sorted(groups) if i < len(groups[name])
        ][: n - len(out)]
        i += 1
    return out


def fixed_label_sources(
    rows: Sequence[schema.Example], max_labels: int = 5
) -> dict[str, list[schema.Example]]:
    """Datasets whose rows share one label set of <= `max_labels` options."""
    return {
        source: rs
        for source, rs in by_source(rows).items()
        if len({r.options for r in rs}) == 1
        and len(rs[0].options) <= max_labels
    }
