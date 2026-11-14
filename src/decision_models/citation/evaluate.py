"""Measures ContextCite on rows with gold evidence spans.

Usage:
  decision-models-cite-eval model.kind=decoder \
      model.path=Athekunal/llm-jev-qwen3-0.6b \
      rows.per=50 out=runs/attribution/qwen3_0.6b

Each row's sentence weights are scored by:
- spans: precision, recall, f_beta of the top 3, average precision, top-1
  (`evidence.metrics.span_metrics`);
- drop@k: how much logit q(answer) falls when the k highest-weighted
  sentences are removed, ContextCite's top-k removal test;
- lds: Spearman correlation, over held-out random ablations, between the
  summed weights of the kept sentences and the actual logit q(answer);
- seconds per row.
Writes <out>/rows.jsonl and summary.json.
"""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import json
import pathlib
from typing import Any

from loguru import logger
import numpy as np

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.citation import context_cite
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.data import storage
from decision_models.evidence import metrics
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base

TOP_K = (1, 3, 5)
METRICS = (
    "ap",
    "top1",
    "precision",
    "recall",
    "lds",
    "drop@1",
    "drop@3",
    "drop@5",
    "seconds",
)


def weight_metrics(
    row: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
    weights: Mapping[str, np.ndarray],
    targets: Callable[[np.ndarray], np.ndarray],
    lds_masks: int = 32,
    seed: int = 1_000_000,
) -> dict[str, dict[str, float]]:
    """Span metrics, LDS and drop@k of each named sentence weighting.

    Every weighting shares the held-out ablations, and one `targets` call
    scores them together with every weighting's top-k removals.

    Args:
      row: The row, with gold spans.
      sentences: Its sentences.
      weights: Method name -> one weight per sentence.
      targets: keep masks [n, S] -> logit q(answer) [n] under them.
      lds_masks: Held-out random ablations for the LDS.
      seed: Seeds them (unseen by any surrogate).
    """
    from scipy.stats import spearmanr

    n = len(sentences)
    held = sentences_lib.ablation_masks(lds_masks, n, seed=seed)  # [L, S]
    ks = [k for k in TOP_K if k < n]
    keep: list[np.ndarray] = [np.ones(n, bool), *held]
    where: dict[tuple[str, int], int] = {}
    for name, w in weights.items():
        order = np.argsort(-w, kind="stable")
        for k in ks:
            mask = np.ones(n, bool)
            mask[order[:k]] = False
            where[name, k] = len(keep)
            keep.append(mask)
    y = targets(np.stack(keep))
    out: dict[str, dict[str, float]] = {}
    for name, w in weights.items():
        lds = (
            spearmanr(held @ w, y[1 : 1 + len(held)]).statistic
            if w.std() > 0
            else 0.0
        )
        out[name] = {
            **(metrics.span_metrics(row, sentences, w) or {}),
            "lds": float(lds),
            **{f"drop@{k}": float(y[0] - y[where[name, k]]) for k in ks},
        }
    return out


def evaluate_row(
    model: base.DecisionModel,
    row: schema.Example,
    cfg: config.CiteConfig,
    lds_masks: int = 32,
    seed: int = 1_000_000,
) -> dict[str, Any]:
    """One row's ContextCite weights and metrics."""
    result, seconds = base.timed(
        model.device, lambda: cite.cite(model, row, cfg)
    )
    exact = cite.scorer(model, cfg, exact=True)

    def targets(keep: np.ndarray) -> np.ndarray:
        return context_cite.ablation_targets(
            model, row, result.sentences, keep, result.answer, exact
        )

    scores = weight_metrics(
        row, result.sentences, {"w": result.weights}, targets, lds_masks, seed
    )["w"]
    return {
        "id": row.id,
        "source": row.source,
        "sources": len(result.sentences),
        "answer": result.answer,
        "gold": row.gold,
        "q": result.q,
        "seconds": seconds,
        **scores,
        "w": result.weights.tolist(),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean of each metric over all rows ("all") and per dataset."""
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        for group in ("all", r["source"]):
            values[group]["rows"].append(1)
            for key in METRICS:
                if not np.isnan(r.get(key, np.nan)):
                    values[group][key].append(r[key])
    return {
        group: {
            k: float(np.sum(v) if k == "rows" else np.mean(v))
            for k, v in m.items()
        }
        for group, m in values.items()
    }


def table(summary: dict[str, dict[str, float]]) -> str:
    """A plain-text table of the means, one line per group."""
    lines = [f"{'group':<14}{'rows':>6}" + "".join(f"{c:>10}" for c in METRICS)]
    for group in ["all", *sorted(x for x in summary if x != "all")]:
        m = summary[group]
        cells = "".join(
            f"{m[c]:>10.3f}" if c in m else f"{'-':>10}" for c in METRICS
        )
        lines.append(f"{group:<14}{int(m['rows']):>6}{cells}")
    return "\n".join(lines)


def load_rows(cfg: config.CiteEvalConfig) -> list[schema.Example]:
    """Rows whose gold spans cover a sentence, `cfg.rows.per` per dataset."""
    rows = storage.load_examples(cfg.rows.source, cfg.rows.split)
    if cfg.generated_spans:
        import datasets

        generated = {
            r["id"]: r["spans"]
            for r in datasets.load_dataset(
                cfg.rows.source, "spans", split="train"
            )
        }
        rows = [
            dataclasses.replace(
                r, spans=tuple(schema.Span(**s) for s in generated[r.id])
            )
            for r in rows
            if generated.get(r.id)
        ]
    return sampling.select_rows(
        rows, cfg.rows, lambda r: sentences_lib.has_evidence(r, cover=True)
    )


def main() -> None:
    """Attributes the rows and writes the per-row and summary metrics."""
    cfg, _ = config.from_command_line(config.CiteEvalConfig, __doc__)
    model = models.load(cfg.model)
    cite.prepare(model, cfg.citation)
    rows = load_rows(cfg)
    if rows and cfg.citation.fuse:
        cite.cite(model, rows[0], cfg.citation)  # Compiles outside the timings.
    out = pathlib.Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    done: list[dict[str, Any]] = []
    with open(out / "rows.jsonl", "w") as f:
        for i, row in enumerate(rows):
            r = evaluate_row(model, row, cfg.citation, cfg.lds_masks)
            f.write(json.dumps(r) + "\n")
            f.flush()
            done.append(r)
            logger.info(
                "{}/{} {} ({} sentences): ap {:.2f} lds {:.2f} {:.2f}s",
                i + 1,
                len(rows),
                row.id,
                r["sources"],
                r.get("ap", float("nan")),
                r["lds"],
                r["seconds"],
            )
    summary = summarize(done)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("ContextCite\n{}", table(summary))


if __name__ == "__main__":
    main()
