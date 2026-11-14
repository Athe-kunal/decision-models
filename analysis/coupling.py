"""How strongly an encoder's option probabilities depend on the other options.

The MoE argument for a sigmoid gate (DeepSeek-V3) is that softmax couples the
experts: raising one score moves every probability. Here the options play the
experts, and per run on the dev rows we measure:

  - confidence: raw score scale, top probability, calibration (ECE) of the top
    option, and of sigma(s_k) per option ("is this option correct on its own");
  - by K: accuracy and log p(gold) for rows with few and with many options;
  - shuffle: the options in `shuffles` more random orders; how often the
    decision (mapped back to the original option) changes, and the mean total
    variation between the probability vectors;
  - balance: predicted against gold shares of each label (datasets with one
    fixed label set) and of each position, over all orders, as total variation;
  - distractor: one random wrong option removed (rows with K >= 3, the removed
    option not the prediction); how often the decision among the remaining
    options changes, and how far s(gold), log p(gold) and sigma(s(gold)) move.

p is the run's own training normalisation (losses.option.probabilities).
Decisions are argmax s for every recipe. Writes <run>/<tag>.json.

Usage:
  python -m analysis.coupling --config configs/analysis/coupling.yaml
"""

import collections
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import math
import pathlib
import random
from typing import Any
from typing import NamedTuple

from loguru import logger
import torch

from analysis import config as analysis_config
from decision_models import config
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.losses import option
from decision_models.models import encoder

# Bucket name -> inclusive range of K.
K_BUCKETS: dict[str, tuple[float, float]] = {
    "2-2": (2, 2),
    "3-4": (3, 4),
    "5-10": (5, 10),
    ">=11": (11, math.inf),
}
ECE_BINS = 15
MAX_LABELS = 5  # A "fixed label set" dataset: one option set of <= 5.


# ---------------------------------------------------------------------------
# Results.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Confidence:
    """Score scale and calibration on the base option order."""

    mean_abs_score: float
    mean_score_range: float
    mean_top_p: float
    ece_top: float
    ece_option_sigmoid: float
    mean_sigmoid_gold: float
    mean_sigmoid_wrong: float


@dataclasses.dataclass(frozen=True)
class KBucket:
    """Rows with a K in one bucket."""

    rows: int
    accuracy: float
    log_p_gold: float


@dataclasses.dataclass(frozen=True)
class ShuffleStats:
    """How decisions move when the options are reordered."""

    decision_changes: float
    mean_tv: float


@dataclasses.dataclass(frozen=True)
class Shuffle(ShuffleStats):
    """ShuffleStats over datasets (their mean) and per dataset."""

    by_dataset: dict[str, ShuffleStats]


@dataclasses.dataclass(frozen=True)
class LabelShares:
    """Gold and predicted share of each label."""

    gold: dict[str, float]
    predicted: dict[str, float]


@dataclasses.dataclass(frozen=True)
class Balance:
    """Predicted against gold shares, as total variation."""

    label_tv: dict[str, float]
    label_shares: dict[str, LabelShares]
    position_tv: dict[str, float]


@dataclasses.dataclass(frozen=True)
class DistractorStats:
    """How decisions move when one wrong option is removed."""

    rows: int
    decision_changes: float
    abs_d_score_gold: float
    abs_d_log_p_gold: float
    abs_d_sigmoid_gold: float


@dataclasses.dataclass(frozen=True)
class Distractor(DistractorStats):
    """DistractorStats over datasets (their mean) and per dataset."""

    by_dataset: dict[str, DistractorStats]


@dataclasses.dataclass(frozen=True)
class Coupling:
    """Every coupling metric of one encoder run."""

    run: str
    loss: str
    rows: int
    shuffles: int
    accuracy: float
    confidence: Confidence
    by_k: dict[str, KBucket]
    shuffle: Shuffle
    balance: Balance
    distractor: Distractor


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


class View(NamedTuple):
    """One option order of a row, mapped back to release order.

    Attributes:
      prediction: The chosen option (release index).
      p: Probabilities over the options in release order.
      ex: The encoded row (its shown order).
    """

    prediction: int
    p: torch.Tensor
    ex: encoder.MaskEncoding


class Base(NamedTuple):
    """A row scored in its base order: scores, probabilities, encoding."""

    s: torch.Tensor
    p: torch.Tensor
    ex: encoder.MaskEncoding


def ece(
    conf: Sequence[float], hit: Sequence[float], bins: int = ECE_BINS
) -> float:
    """Expected calibration error with equal-width bins."""
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [
            i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)
        ]
        if idx:
            gap = sum(conf[i] for i in idx) - sum(hit[i] for i in idx)
            err += abs(gap) / total
    return err


def tv(a: Mapping[Any, float], b: Mapping[Any, float]) -> float:
    """Total variation between two share dicts."""
    return 0.5 * sum(
        abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)
    )


def shares(counter: collections.Counter[Any]) -> dict[Any, float]:
    """Each key's share of the counts."""
    n = sum(counter.values())
    return {k: v / n for k, v in counter.items()}


def mean(xs: Iterable[float]) -> float:
    """The mean, NaN when empty."""
    values = list(xs)
    return sum(values) / len(values) if values else float("nan")


def by_dataset_mean(per: Mapping[str, Sequence[float]]) -> float:
    """Mean over datasets of each dataset's mean, as ALL elsewhere."""
    return mean(mean(v) for v in per.values() if v)


def bucket(k: int) -> str:
    """The K_BUCKETS name of an option count."""
    return next(name for name, (lo, hi) in K_BUCKETS.items() if lo <= k <= hi)


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------


def score_views(
    model: encoder.MaskedEncoder,
    rows: Sequence[schema.Example],
    shuffles: int,
    seed: int,
) -> tuple[list[Base], list[list[View]]]:
    """Each row in its base order and `shuffles` more seeded orders."""
    jobs = [
        (i, v, model.encode(r, random.Random(f"coupling:{seed}:{r.id}:{v}")))
        for i, r in enumerate(rows)
        for v in range(shuffles + 1)
    ]
    scores = model.score_encoded([ex for _, _, ex in jobs])
    bases: dict[int, Base] = {}
    views: list[list[View]] = [[] for _ in rows]
    for (i, v, ex), s in zip(jobs, scores, strict=True):
        s = s.float().cpu()
        p = option.probabilities(s, model.loss_kind)
        p_release = torch.zeros(len(ex.order))
        p_release[torch.tensor(ex.order)] = p
        views[i].append(View(ex.order[int(s.argmax())], p_release, ex))
        if v == 0:
            bases[i] = Base(s, p, ex)
    return [bases[i] for i in range(len(rows))], views


def confidence(bases: Sequence[Base]) -> Confidence:
    """Score scale and calibration of the base orders."""
    top_conf: list[float] = []
    top_hit: list[float] = []
    option_conf: list[float] = []
    option_hit: list[float] = []
    for b in bases:
        top_conf.append(float(b.p.max()))
        top_hit.append(float(int(b.s.argmax()) == b.ex.gold))
        option_conf += torch.sigmoid(b.s).tolist()
        option_hit += [float(m == b.ex.gold) for m in range(len(b.s))]
    pairs = list(zip(option_conf, option_hit, strict=True))
    return Confidence(
        mean_abs_score=mean(float(b.s.abs().mean()) for b in bases),
        mean_score_range=mean(float(b.s.max() - b.s.min()) for b in bases),
        mean_top_p=mean(top_conf),
        ece_top=ece(top_conf, top_hit),
        ece_option_sigmoid=ece(option_conf, option_hit),
        mean_sigmoid_gold=mean(c for c, h in pairs if h),
        mean_sigmoid_wrong=mean(c for c, h in pairs if not h),
    )


def by_k(bases: Sequence[Base]) -> dict[str, KBucket]:
    """Accuracy and log p(gold) per option-count bucket."""
    hits: dict[str, list[float]] = collections.defaultdict(list)
    log_p: dict[str, list[float]] = collections.defaultdict(list)
    for b in bases:
        name = bucket(len(b.s))
        hits[name].append(float(int(b.s.argmax()) == b.ex.gold))
        log_p[name].append(float(b.p[b.ex.gold].log()))
    return {
        name: KBucket(len(hits[name]), mean(hits[name]), mean(log_p[name]))
        for name in hits
    }


def shuffle(
    rows: Sequence[schema.Example], views: Sequence[Sequence[View]]
) -> Shuffle:
    """Decision changes and probability movement under reordering."""
    flips: dict[str, list[float]] = collections.defaultdict(list)
    tvs: dict[str, list[float]] = collections.defaultdict(list)
    for r, vs in zip(rows, views, strict=True):
        flips[r.source].append(float(len({v.prediction for v in vs}) > 1))
        tvs[r.source].append(
            mean(0.5 * float((vs[0].p - v.p).abs().sum()) for v in vs[1:])
        )
    return Shuffle(
        decision_changes=by_dataset_mean(flips),
        mean_tv=by_dataset_mean(tvs),
        by_dataset={
            source: ShuffleStats(mean(v), mean(tvs[source]))
            for source, v in sorted(flips.items())
        },
    )


def balance(
    rows: Sequence[schema.Example], views: Sequence[Sequence[View]]
) -> Balance:
    """Label and position shares, predicted against gold, over all orders."""
    fixed = sampling.fixed_label_sources(rows, MAX_LABELS)
    label_gold: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    label_pred: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    pos_gold: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    pos_pred: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for r, vs in zip(rows, views, strict=True):
        for v in vs:
            k = len(v.ex.order)
            pos_gold[k][v.ex.gold] += 1
            pos_pred[k][v.ex.order.index(v.prediction)] += 1
            if r.source in fixed:
                label_gold[r.source][r.answer] += 1
                label_pred[r.source][r.options[v.prediction]] += 1
    return Balance(
        label_tv={
            s: tv(shares(label_gold[s]), shares(label_pred[s]))
            for s in sorted(fixed)
        },
        label_shares={
            s: LabelShares(shares(label_gold[s]), shares(label_pred[s]))
            for s in sorted(fixed)
        },
        position_tv={
            str(k): tv(shares(pos_gold[k]), shares(pos_pred[k]))
            for k in sorted(pos_gold)
            if sum(pos_gold[k].values()) >= 100
        },
    )


def distractor(
    model: encoder.MaskedEncoder,
    rows: Sequence[schema.Example],
    bases: Sequence[Base],
    seed: int,
) -> Distractor:
    """Decision and score movement when one wrong option is removed.

    The removed option is a seeded wrong option other than the prediction
    (rows with K >= 3); the others keep their base relative order.
    """
    jobs: list[tuple[int, int, encoder.MaskEncoding]] = []
    for i, (r, b) in enumerate(zip(rows, bases, strict=True)):
        k, pred = len(b.ex.order), int(b.s.argmax())
        wrong = [m for m in range(k) if m not in (b.ex.gold, pred)]
        if k < 3 or not wrong:
            continue
        m = random.Random(f"drop:{seed}:{r.id}").choice(wrong)
        removed = b.ex.order[m]
        keep = [o for o in range(len(r.options)) if o != removed]
        order = [keep.index(o) for o in b.ex.order if o != removed]
        sub = r.without_option(removed)
        jobs.append(
            (i, m, encoder.encode_order(sub, order, model.tokenizer, model.fmt))
        )
    scores = model.score_encoded([ex for _, _, ex in jobs])
    stats: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for (i, m, ex2), s2 in zip(jobs, scores, strict=True):
        b = bases[i]
        s2 = s2.float().cpu()
        p2 = option.probabilities(s2, model.loss_kind)
        kept = [j for j in range(len(b.s)) if j != m]  # Base index -> new.
        g, g2 = b.ex.gold, ex2.gold
        d = stats[rows[i].source]
        d["decision_changes"].append(
            float(int(s2.argmax()) != kept.index(int(b.s.argmax())))
        )
        d["abs_d_score_gold"].append(abs(float(s2[g2] - b.s[g])))
        d["abs_d_log_p_gold"].append(abs(float(p2[g2].log() - b.p[g].log())))
        d["abs_d_sigmoid_gold"].append(
            abs(float(torch.sigmoid(s2[g2]) - torch.sigmoid(b.s[g])))
        )
    keys = (
        "decision_changes",
        "abs_d_score_gold",
        "abs_d_log_p_gold",
        "abs_d_sigmoid_gold",
    )
    return Distractor(
        rows=len(jobs),
        **{
            k: by_dataset_mean({s: d[k] for s, d in stats.items()})
            for k in keys
        },
        by_dataset={
            s: DistractorStats(
                len(d["decision_changes"]), *(mean(d[k]) for k in keys)
            )
            for s, d in sorted(stats.items())
        },
    )


def analyse(
    run: pathlib.Path, rows: Sequence[schema.Example], shuffles: int, seed: int
) -> Coupling:
    """Every coupling metric of one encoder run on `rows`."""
    model = encoder.MaskedEncoder.from_run(run)
    bases, views = score_views(model, rows, shuffles, seed)
    hits: dict[str, list[float]] = collections.defaultdict(list)
    for r, b in zip(rows, bases, strict=True):
        hits[r.source].append(float(int(b.s.argmax()) == b.ex.gold))
    return Coupling(
        run=str(run),
        loss=model.loss_kind,
        rows=len(rows),
        shuffles=shuffles,
        accuracy=by_dataset_mean(hits),
        confidence=confidence(bases),
        by_k=by_k(bases),
        shuffle=shuffle(rows, views),
        balance=balance(rows, views),
        distractor=distractor(model, rows, bases, seed),
    )


def main() -> None:
    """Analyses every run and writes <run>/<tag>.json."""
    cfg, _ = config.from_command_line(analysis_config.CouplingConfig, __doc__)
    rows = sampling.select(cfg.rows)
    for run in map(pathlib.Path, cfg.runs):
        logger.info("{}: {} rows", run, len(rows))
        out = analyse(run, rows, cfg.shuffles, cfg.seed)
        path = run / f"{cfg.tag}.json"
        path.write_text(json.dumps(dataclasses.asdict(out), indent=2))
        logger.info(
            "{}: acc {:.3f} | shuffle changes {:.3f} | drop changes {:.3f} | "
            "ece {:.3f}",
            run,
            out.accuracy,
            out.shuffle.decision_changes,
            out.distractor.decision_changes,
            out.confidence.ece_top,
        )


if __name__ == "__main__":
    main()
