"""Decisions for a split, with ContextCite evidence when `citation` is set.

Usage:
  decision-models-predict model.kind=decoder \
      model.path=Athekunal/llm-jev-qwen3-0.6b \
      rows.source=Athekunal/llm-jev-reason-long limit=100 \
      citation={} \
      out=runs/predict/eval.jsonl

Without `citation`, rows are scored in batches. With it (`citation={}`),
every row runs ContextCite (`citation.cite`: one-pass full masking with
fused kernels for decoders, packed deletion for encoders) and the decision
is the full row's.
Writes one JSON line per row: id, source, answer (0-based), option, q, gold,
correct, and with `citation` the top-k sentences ("evidence") and seconds.
"""

import json
import pathlib
from typing import Any

from loguru import logger

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.models import base


def record(row: schema.Example, answer: int, q: float) -> dict[str, Any]:
    """One output line, without evidence."""
    return {
        "id": row.id,
        "source": row.source,
        "answer": answer,
        "option": row.options[answer],
        "q": q,
        "gold": row.gold,
        "correct": answer == row.gold,
    }


def main() -> None:
    """Writes the decisions (and with `citation`, the evidence) for a split."""
    cfg, _ = config.from_command_line(config.PredictConfig, __doc__)
    rows = sampling.select(cfg.rows)[: cfg.limit]
    model = models.load(cfg.model)
    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        if cfg.citation is None:
            for row, s in zip(rows, model.score(rows), strict=True):
                f.write(json.dumps(record(row, *base.decide(s))) + "\n")
        else:
            cite.prepare(model, cfg.citation)
            for i, row in enumerate(rows):
                result, seconds = base.timed(
                    model.device,
                    lambda row=row: cite.cite(model, row, cfg.citation),
                )
                line = {
                    **record(row, result.answer, result.q),
                    "seconds": seconds,
                    "evidence": cite.top_sentences(
                        row, result, cfg.citation.top_k
                    ),
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                if (i + 1) % 50 == 0 or i + 1 == len(rows):
                    logger.info("{}/{} rows", i + 1, len(rows))
    logger.info("wrote {} decisions to {}", len(rows), out)


if __name__ == "__main__":
    main()
