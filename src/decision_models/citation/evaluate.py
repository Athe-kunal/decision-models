"""Measures ContextCite on rows with gold evidence spans.

Usage:
  decision-models-cite-eval model.kind=decoder \
      model.path=Athekunal/llm-jev-qwen3-0.6b \
      per=50 out=runs/attribution/qwen3_0.6b

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
from collections.abc import Sequence
import dataclasses
import json
import pathlib
import time
from typing import Any

from loguru import logger
import numpy as np
import torch

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.citation import context_cite
from decision_models.citation import masked
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.data import storage
from decision_models.evidence import metrics
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base
from decision_models.models import decoder

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


def ablated_scores(
    model: base.DecisionModel,
    example: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
    keep: np.ndarray,
    answer: int,
    cfg: config.CiteConfig,
) -> np.ndarray:
    """Logit q(answer) for every keep row [n, S], as ContextCite scores them."""
    if isinstance(model, decoder.TrieDecoder) and cfg.removal == "full":
        ids, owner, _ = masked.token_owners(model.tokenizer, example, sentences)
        scores = masked.score_keep(
            model.unwrapped,
            model.tokenizer,
            ids,
            owner,
            keep,
            len(example.options),
            model.device,
            cfg.budget,
        )
        return np.array([context_cite.logit_q(s, answer) for s in scores])
    return context_cite.ablation_targets(
        model, example, sentences, keep, answer
    )


def _sync(model: base.DecisionModel) -> None:
    if model.device.type == "cuda":
        torch.cuda.synchronize()


def evaluate_row(
    model: base.DecisionModel,
    row: schema.Example,
    cfg: config.CiteConfig,
    lds_masks: int = 32,
    seed: int = 1_000_000,
) -> dict[str, Any]:
    """One row's ContextCite weights and metrics."""
    _sync(model)
    start = time.time()
    result = cite.cite(model, row, cfg)
    _sync(model)
    seconds = time.time() - start
    n, answer, w = len(result.sentences), result.answer, result.weights
    order = np.argsort(-w, kind="stable")
    ks = [k for k in TOP_K if k < n]
    held = sentences_lib.ablation_masks(lds_masks, n, seed=seed)  # Unseen.
    keep = np.ones((1 + len(held) + len(ks), n), bool)
    keep[1 : 1 + len(held)] = held
    for i, k in enumerate(ks):
        keep[1 + len(held) + i, order[:k]] = False
    y = ablated_scores(model, row, result.sentences, keep, answer, cfg)
    from scipy.stats import spearmanr

    lds = (
        spearmanr(held @ w, y[1 : 1 + len(held)]).statistic
        if w.std() > 0
        else 0.0
    )
    return {
        "id": row.id,
        "source": row.source,
        "sources": n,
        "answer": answer,
        "gold": row.gold,
        "q": result.q,
        "seconds": seconds,
        **(metrics.span_metrics(row, result.sentences, w) or {}),
        "lds": float(lds),
        **{
            f"drop@{k}": float(y[0] - y[1 + len(held) + i])
            for i, k in enumerate(ks)
        },
        "w": w.tolist(),
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
    """`per` seeded rows per dataset whose gold spans cover >= 1 sentence."""
    rows = storage.load_examples(cfg.source, cfg.split)
    if cfg.generated_spans:
        import datasets

        generated = {
            r["id"]: r["spans"]
            for r in datasets.load_dataset(cfg.source, "spans", split="train")
        }
        rows = [
            dataclasses.replace(
                r, spans=tuple(schema.Span(**s) for s in generated[r.id])
            )
            for r in rows
            if generated.get(r.id)
        ]
    keep = []
    for r in rows:
        if not r.spans or (cfg.datasets and r.source not in cfg.datasets):
            continue
        sents = sentences_lib.split(r)
        if len(sents) >= 2 and sentences_lib.gold_indices(r, sents):
            keep.append(r)
    return sampling.per_dataset(keep, cfg.per, seed=0)


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
    done = []
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
