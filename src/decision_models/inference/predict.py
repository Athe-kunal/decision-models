"""Decisions for a split, with ContextCite evidence behind `cite=true`.

Usage:
  decision-models-predict model.kind=decoder \
      model.path=Athekunal/llm-jev-qwen3-0.6b \
      source=Athekunal/llm-jev-reason-long limit=100 cite=true \
      out=runs/predict/eval.jsonl

Without `cite`, rows are scored in batches. With it, every row runs
ContextCite (`citation.cite`: one-pass full masking with fused kernels for
decoders, packed deletion for encoders) and the decision is the full row's.
Writes one JSON line per row: id, source, answer (0-based), option, q, gold,
correct, and with `cite` the top-k sentences ("evidence") and seconds.
"""

import json
import pathlib
import time
from typing import Any

from loguru import logger
import torch

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.data import schema
from decision_models.data import storage


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
    """Writes the decisions (and with `cite`, the evidence) for a split."""
    cfg, _ = config.from_command_line(config.PredictConfig, __doc__)
    rows = [
        r
        for r in storage.load_examples(cfg.source, cfg.split)
        if not cfg.datasets or r.source in cfg.datasets
    ][: cfg.limit]
    model = models.load(cfg.model)
    out = pathlib.Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        if not cfg.cite:
            for row, s in zip(rows, model.score(rows), strict=True):
                answer = int(s.argmax())
                q = torch.softmax(s, 0)[answer].item()
                f.write(json.dumps(record(row, answer, q)) + "\n")
        else:
            cite.prepare(model, cfg.citation)
            for i, row in enumerate(rows):
                start = time.time()
                result = cite.cite(model, row, cfg.citation)
                line = {
                    **record(row, result.answer, result.q),
                    "seconds": time.time() - start,
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
