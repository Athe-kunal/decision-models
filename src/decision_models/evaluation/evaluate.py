"""Evaluates any decision model on a held-out split.

Usage:
  decision-models-evaluate --config configs/evaluate/encoder.yaml \
      model.path=runs/ultrabert_380m out=runs/ultrabert_380m tag=final

Metrics per dataset and their mean over datasets ("ALL"): accuracy of
argmax s and mean log q(gold), q = softmax(s). Decoders also report the
root's legal mass and, on `gen_per` rows per dataset, the validity and
accuracy of unguided greedy text (eval_<tag>_examples.jsonl shows them).
Writes eval_<tag>.json; with save_rows eval_<tag>_rows.json (per-row
correctness, for bootstrap intervals); with label_shares
eval_<tag>_labels.json (gold vs predicted label shares, see `label_shares`).
"""

import collections
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import math
import pathlib
import random
from typing import Any

from loguru import logger
import torch

from decision_models import config
from decision_models import models
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.models import base

Metrics = dict[str, dict[str, float]]


@dataclasses.dataclass(frozen=True)
class RowResult:
    """One scored row.

    Attributes:
      id: The row's id.
      source: Its dataset.
      correct: Whether argmax s is the gold option.
      log_p_gold: log softmax(s)[gold].
      prediction: argmax s (release order).
      root_legal_mass: Decoders: probability of a valid first answer token.
      gen_valid: Text mode (decoders, sampled rows): the reply is a valid "k]".
      gen_correct: Text mode: the reply names the gold option.
    """

    id: str
    source: str
    correct: bool
    log_p_gold: float
    prediction: int
    root_legal_mass: float | None = None
    gen_valid: bool | None = None
    gen_correct: bool | None = None


def score_rows(
    model: base.DecisionModel,
    rows: Sequence[schema.Example],
    gen_ids: set[str] | None = None,
    examples_out: list[dict[str, Any]] | None = None,
) -> list[RowResult]:
    """Scores the rows and grades them.

    All rows go to the model in one call: decoders batch rows of similar
    length, and in bf16 a row's scores depend on its batch's padding, so
    scoring in smaller groups would change results.

    Args:
      model: Any decision model.
      rows: The rows.
      gen_ids: Rows also answered in text mode, where the model supports it
        (decoders).
      examples_out: If given, receives one readable record per text-mode row.
    """
    gen_ids = gen_ids or set()
    out: list[RowResult] = []
    scored = model.score_with_mass(rows)
    for row, (s, mass) in zip(rows, scored, strict=True):
        prediction = int(s.argmax())
        gen_valid = gen_correct = None
        reply = model.generate_answer(row) if row.id in gen_ids else None
        if reply is not None:
            text, named = reply
            gen_valid, gen_correct = named is not None, named == row.gold
            if examples_out is not None:
                examples_out.append(_example_record(row, s, text, gen_valid))
        out.append(
            RowResult(
                id=row.id,
                source=row.source,
                correct=prediction == row.gold,
                log_p_gold=torch.log_softmax(s.float(), 0)[row.gold].item(),
                prediction=prediction,
                root_legal_mass=mass,
                gen_valid=gen_valid,
                gen_correct=gen_correct,
            )
        )
    return out


def _example_record(
    row: schema.Example, s: torch.Tensor, text: str, valid: bool
) -> dict[str, Any]:
    """A readable record of one text-mode row."""
    options = list(row.options)
    top = torch.softmax(s.float(), 0).topk(min(3, len(options)))
    return {
        "source": row.source,
        "id": row.id,
        "state": " / ".join(p.text for p in row.parts)[:400],
        "question": row.question,
        "options": options if len(options) <= 10 else [*options[:10], "…"],
        "gold": f"[{row.gold + 1}] {options[row.gold]}",
        "scored_top3": [
            f"[{k + 1}] {options[k]} ({v:.3f})"
            for v, k in zip(
                top.values.tolist(), top.indices.tolist(), strict=True
            )
        ],
        "generation": text,
        "valid": valid,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def macro_average(
    per_source: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    """The mean over datasets of each metric every dataset has ("ALL").

    Row counts ("rows", "gen_rows") are not averaged; NaNs are skipped.

    Raises:
      ValueError: If there are no datasets (no rows were selected).
    """
    rows = list(per_source.values())
    if not rows:
        raise ValueError(
            "no rows to evaluate: check rows.source, rows.part, rows.datasets "
            "and rows_file"
        )
    keys = [
        k
        for k in rows[0]
        if k not in ("rows", "gen_rows") and all(k in m for m in rows)
    ]
    out: dict[str, float] = {}
    for k in keys:
        values = [float(m[k]) for m in rows if not math.isnan(float(m[k]))]
        out[k] = _mean(values) if values else float("nan")
    return out


def summarize(results: Sequence[RowResult]) -> Metrics:
    """Per-dataset metrics and rows; "ALL" averages the datasets.

    Optional metrics (legal mass, text mode) appear where any row has them.
    """
    by_source: dict[str, list[RowResult]] = collections.defaultdict(list)
    for r in results:
        by_source[r.source].append(r)
    metrics: Metrics = {}
    for name, rs in sorted(by_source.items()):
        m: dict[str, float] = {
            "accuracy": _mean([float(r.correct) for r in rs]),
            "log_p_gold": _mean([r.log_p_gold for r in rs]),
        }
        masses = [
            r.root_legal_mass for r in rs if r.root_legal_mass is not None
        ]
        if masses:
            m["root_legal_mass"] = _mean(masses)
        gen = [r for r in rs if r.gen_valid is not None]
        if gen:
            m["gen_valid"] = _mean([float(bool(r.gen_valid)) for r in gen])
            m["gen_correct"] = _mean([float(bool(r.gen_correct)) for r in gen])
        m["rows"] = len(rs)
        if gen:
            m["gen_rows"] = len(gen)
        metrics[name] = m
    metrics["ALL"] = macro_average(metrics)
    return metrics


def label_shares(
    model: base.DecisionModel,
    rows: Sequence[schema.Example],
    shuffles: int = 3,
    max_labels: int = 5,
) -> dict[str, dict[str, dict[str, float]]]:
    """Gold vs predicted share of each label, under shuffled option orders.

    Only datasets with one fixed label set of at most `max_labels` options.
    A model that prefers a label (not a slot) shows predicted >> gold for it
    whatever its position; shuffling rules out position bias.
    """
    fixed = sampling.fixed_label_sources(rows, max_labels)
    jobs: list[tuple[schema.Example, schema.Example, list[int]]] = []
    for rs in fixed.values():
        for r in rs:
            for k in range(shuffles):
                order = list(range(len(r.options)))
                random.Random(f"labels:{r.id}:{k}").shuffle(order)
                jobs.append((r, r.reordered(order), order))
    scores = model.score([shown for _, shown, _ in jobs])
    gold: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    predicted: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for (r, _, order), s in zip(jobs, scores, strict=True):
        gold[r.source][r.answer] += 1
        predicted[r.source][r.options[order[int(s.argmax())]]] += 1
    return {
        source: {
            label: {
                "gold": gold[source][label] / sum(gold[source].values()),
                "predicted": predicted[source][label]
                / sum(predicted[source].values()),
            }
            for label in rs[0].options
        }
        for source, rs in sorted(fixed.items())
    }


def evaluate_distributed(
    model: base.DecisionModel,
    rows: Sequence[schema.Example],
    accelerator: Any,
    gen_ids: set[str] | None = None,
) -> Metrics:
    """`summarize(score_rows(...))` across the processes of a training run.

    Under FSDP every process must run the same forwards, so each scores all
    rows; otherwise the rows are split over the processes and gathered.
    """
    from accelerate.utils import DistributedType
    from accelerate.utils import gather_object

    was_training = model.network.training
    model.eval()
    if accelerator.distributed_type == DistributedType.FSDP:
        results = score_rows(model, rows, gen_ids)
    else:
        mine = list(
            rows[accelerator.process_index :: accelerator.num_processes]
        )
        results = gather_object(score_rows(model, mine, gen_ids))
    model.train(was_training)
    return summarize(results)


def save(
    out: pathlib.Path,
    tag: str,
    metrics: Metrics,
    examples: Sequence[dict[str, Any]] = (),
) -> None:
    """Writes <out>/eval_<tag>.json (and the text-mode examples, if any)."""
    (out / f"eval_{tag}.json").write_text(json.dumps(metrics, indent=2))
    if examples:
        with open(out / f"eval_{tag}_examples.jsonl", "w") as f:
            for e in examples:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


def save_rows(
    out: pathlib.Path, tag: str, results: Sequence[RowResult]
) -> None:
    """Writes <out>/eval_<tag>_rows.json (id, dataset, correctness)."""
    rows = [
        {"id": r.id, "source": r.source, "correct": r.correct} for r in results
    ]
    (out / f"eval_{tag}_rows.json").write_text(json.dumps(rows))


def log_summary(tag: str, metrics: Metrics) -> None:
    """Logs the mean metrics, then accuracy per dataset."""
    logger.info(
        "eval {}: {}",
        tag,
        " ".join(f"{k} {v:.4f}" for k, v in metrics["ALL"].items()),
    )
    logger.info(
        "eval {} accuracy per dataset: {}",
        tag,
        " ".join(
            f"{name} {m['accuracy']:.3f}"
            for name, m in metrics.items()
            if name != "ALL" and "accuracy" in m
        ),
    )


def main() -> None:
    """Evaluates a model on one part of a split."""
    cfg, _ = config.from_command_line(config.EvaluateConfig, __doc__)
    ids = sampling.read_ids(cfg.rows_file)
    rows = sampling.select(
        cfg.rows, None if ids is None else lambda r: r.id in ids
    )
    if not rows:
        raise SystemExit(
            f"no rows selected from {cfg.rows.source} ({cfg.rows.split}, part "
            f"{cfg.rows.part}, datasets {cfg.rows.datasets}, rows_file "
            f"{cfg.rows_file})"
        )
    gen_ids = {
        r.id for r in sampling.per_dataset(rows, cfg.gen_per, cfg.rows.seed)
    }
    model = models.load(cfg.model)
    examples: list[dict[str, Any]] = []
    results = score_rows(
        model, rows, gen_ids if cfg.gen_per else None, examples
    )
    metrics = summarize(results)
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(out, cfg.tag, metrics, examples)
    if cfg.save_rows:
        save_rows(out, cfg.tag, results)
    log_summary(cfg.tag, metrics)
    if cfg.label_shares:
        shares = label_shares(model, rows)
        path = out / f"eval_{cfg.tag}_labels.json"
        path.write_text(json.dumps(shares, indent=2))


if __name__ == "__main__":
    main()
