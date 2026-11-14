"""Compares citation baselines for an encoder on rows with gold evidence.

Usage:
  python -m analysis.citation at2=runs/at2_ultrabert_380m rows.part=test

Methods: AT2 (theta from <at2>/theta.pt), average attention (uniform
theta) and ContextCite by deletion, all on the AT2 run's encoder. Each gives
per-sentence weights for the model's own answer, scored by:
- span metrics against the gold spans (ap, top1, precision / recall @ 3);
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sentences and the actual logit q(answer);
- drop@k: how much logit q(answer) falls when the k highest-weighted
  sentences are deleted;
- seconds per row.
Writes <at2>/attribution_<split>[_<part>].json and logs a table.
"""

import collections
from collections.abc import Callable
from collections.abc import Sequence
import json
import pathlib
from typing import Any

from loguru import logger
import numpy as np
import torch

from analysis import at2
from analysis import config as analysis_config
from decision_models import config
from decision_models.citation import context_cite
from decision_models.citation import evaluate as cite_evaluate
from decision_models.citation.evaluate import METRICS
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base
from decision_models.models import encoder

Method = Callable[[schema.Example], context_cite.Attribution]


def methods(
    model: encoder.MaskedEncoder, theta: torch.Tensor
) -> dict[str, Method]:
    """Method name -> row -> attribution."""
    uniform = torch.full_like(theta, 1 / len(theta))
    return {
        "at2": lambda r: at2.attribute(model, r, theta),
        "average_attention": lambda r: at2.attribute(model, r, uniform),
        "contextcite": lambda r: context_cite.attribute(model, r),
    }


def evaluate_row(
    model: encoder.MaskedEncoder,
    row: schema.Example,
    fns: dict[str, Method],
) -> dict[str, Any]:
    """Every method's weights and metrics on one row."""
    results: dict[str, context_cite.Attribution] = {}
    seconds: dict[str, float] = {}
    for name, fn in fns.items():
        results[name], seconds[name] = base.timed(
            model.device, lambda fn=fn: fn(row)
        )
    reference = results["contextcite"]

    def targets(keep: np.ndarray) -> np.ndarray:
        return context_cite.ablation_targets(
            model, row, reference.sentences, keep, reference.answer
        )

    scores = cite_evaluate.weight_metrics(
        row,
        reference.sentences,
        {name: r.weights for name, r in results.items()},
        targets,
    )
    return {
        "id": row.id,
        "source": row.source,
        "sources": len(reference.sentences),
        "answer": reference.answer,
        "gold": row.gold,
        "answers_agree": len({r.answer for r in results.values()}) == 1,
        "methods": {
            name: {
                **scores[name],
                "seconds": seconds[name],
                "w": r.weights.tolist(),
            }
            for name, r in results.items()
        },
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of each metric per dataset and method; "ALL" averages datasets."""
    values: dict[tuple[str, str], dict[str, list[float]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    for row in rows:
        for name, m in row["methods"].items():
            for k in METRICS:
                if k in m and m[k] == m[k]:  # Skips NaN.
                    values[(row["source"], name)][k].append(m[k])
    out: dict[str, dict[str, Any]] = collections.defaultdict(dict)
    for (source, name), ms in values.items():
        out[source][name] = {k: float(np.mean(v)) for k, v in ms.items()}
        out[source][name]["rows"] = len(ms["seconds"])
    names = {n for d in out.values() for n in d}
    out["ALL"] = {
        n: {
            k: float(
                np.mean(
                    [out[s][n][k] for s in out if s != "ALL" and k in out[s][n]]
                )
            )
            for k in METRICS
        }
        for n in names
    }
    return dict(out)


def table(summary: dict[str, Any]) -> str:
    """The summary as a fixed-width table."""
    lines = [
        f"{'dataset':12s} {'method':18s} "
        + " ".join(f"{k:>8s}" for k in METRICS)
    ]
    for source in sorted(summary, key=lambda s: (s == "ALL", s)):
        for name in ("at2", "average_attention", "contextcite"):
            if name in summary[source]:
                m = summary[source][name]
                cells = " ".join(
                    f"{m.get(k, float('nan')):8.3f}" for k in METRICS
                )
                lines.append(f"{source:12s} {name:18s} {cells}")
    return "\n".join(lines)


def main() -> None:
    """Scores every method on the rows and writes the JSON."""
    cfg, _ = config.from_command_line(analysis_config.CitationConfig, __doc__)
    at2_dir = pathlib.Path(cfg.at2)
    at2_cfg = config.load(analysis_config.AT2Config, at2_dir / "config.yaml")
    part = "" if cfg.rows.part == "all" else f"_{cfg.rows.part}"
    name = f"attribution_{cfg.rows.split}{part}"
    logger.add(at2_dir / f"{name}.log")
    model = encoder.MaskedEncoder.from_run(
        at2_cfg.encoder.run, attention=at2_cfg.encoder.attention
    )
    theta = torch.load(at2_dir / "theta.pt").to(model.device)
    rows = sampling.select(cfg.rows, sentences_lib.has_evidence)
    logger.info(
        "{} rows: {}",
        len(rows),
        {s: len(rs) for s, rs in sampling.by_source(rows).items()},
    )
    fns = methods(model, theta)
    results: list[dict[str, Any]] = []
    for i, row in enumerate(rows, 1):
        results.append(evaluate_row(model, row, fns))
        if i % 25 == 0 or i == len(rows):
            logger.info("{}/{} rows", i, len(rows))
    summary = summarize(results)
    (at2_dir / f"{name}.json").write_text(
        json.dumps({"summary": summary, "rows": results})
    )
    logger.info("\n{}", table(summary))


if __name__ == "__main__":
    main()
