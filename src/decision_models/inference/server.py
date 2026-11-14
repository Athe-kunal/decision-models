"""A local web server to try a decision model on rows of a split.

Usage (needs `uv sync --extra serve`; binds 127.0.0.1 by default):
  decision-models-serve model.kind=encoder \
      model.path=runs/ultrabert_380m_lr1.5e-5 \
      evidence=runs/evidence_ultrabert_380m
  decision-models-serve model.kind=decoder \
      model.path=Athekunal/llm-jev-qwen3-0.6b

Open http://127.0.0.1:8767/ to sample a row, see the decision, the option
probabilities and the evidence. The page reads this JSON API:
  GET /api/info                   the loaded model and what it can cite
  GET /api/datasets               datasets, their rows, rows with gold spans
  GET /api/sample?source=&spans=  a random row id (spans=1: gold evidence only)
  GET /api/row?id=&shuffle=&seed= the decision: options (in the order the
                                  model saw), q, answer, sentences (gold
                                  marked) and seconds; shuffle=1 scores a
                                  seeded random option order
  GET /api/attribute?id=          ContextCite sentence weights and, with an
                                  evidence head, its token scores
"""

import dataclasses
import functools
import pathlib
import random
import threading
from typing import Any

from loguru import logger
import torch

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.data import sampling
from decision_models.data import schema
from decision_models.evidence import head as head_lib
from decision_models.evidence import metrics
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base
from decision_models.models import encoder

PAGE = pathlib.Path(__file__).with_name("page.html")


def make_app(
    model: base.DecisionModel,
    rows: list[schema.Example],
    cfg: config.ServeConfig,
    evidence: head_lib.Trained | None = None,
) -> Any:
    """The FastAPI app over a loaded model and its rows.

    Raises:
      ValueError: If an evidence head is given with a non-encoder model.
    """
    import fastapi
    from fastapi import responses

    if evidence is not None and not isinstance(model, encoder.MaskedEncoder):
        raise ValueError("the evidence head needs an encoder model")
    page = PAGE.read_text()
    by_id = {r.id: r for r in rows}
    groups = sampling.by_source(rows)
    ids = {source: [r.id for r in rs] for source, rs in groups.items()}
    span_ids = {
        source: [r.id for r in rs if r.has_spans]
        for source, rs in groups.items()
    }
    datasets_info = [
        {"id": s, "rows": len(ids[s]), "withSpans": len(span_ids[s])}
        for s in sorted(groups)
    ]
    methods = (["contextcite"] if cfg.citation else []) + (
        ["evidence"] if evidence else []
    )
    info = {
        "model": cfg.model.path,
        "kind": cfg.model.kind,
        "adapter": cfg.model.adapter,
        "data": f"{cfg.rows.source} ({cfg.rows.split})",
        "methods": methods,
        "removal": cfg.citation.removal if cfg.citation else None,
    }
    if cfg.citation:
        cite.prepare(model, cfg.citation)
    lock = threading.Lock()  # One model, one request at a time.
    app = fastapi.FastAPI()

    def get(row_id: str) -> schema.Example:
        if row_id not in by_id:
            raise fastapi.HTTPException(404, f"unknown row {row_id}")
        return by_id[row_id]

    @functools.cache
    def sentences(row_id: str) -> list[sentences_lib.Sentence]:
        return sentences_lib.split(by_id[row_id])

    @app.get("/", response_class=responses.HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/info")
    def get_info() -> dict[str, Any]:
        return info

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        return datasets_info

    @app.get("/api/sample")
    def sample(source: str = "", spans: int = 0) -> dict[str, str]:
        pools = span_ids if spans else ids
        pool = [
            i for s, p in pools.items() if not source or s == source for i in p
        ]
        if not pool:
            raise fastapi.HTTPException(404, "no rows match")
        return {"id": random.choice(pool)}

    @app.get("/api/row")
    def row(id: str, shuffle: int = 0, seed: int = 0) -> dict[str, Any]:  # noqa: A002
        r = get(id)
        order = list(range(len(r.options)))
        if shuffle:
            random.Random(f"{id}:{seed}").shuffle(order)
        shown = r.reordered(order)
        with lock:
            s, seconds = base.timed(
                model.device, lambda: model.score([shown])[0]
            )
        sents = sentences(id)
        gold = sentences_lib.gold_indices(r, sents)
        return {
            "id": id,
            "source": r.source,
            "question": r.question,
            "options": list(shown.options),
            "order": order,
            "shuffled": bool(shuffle),
            "gold": shown.gold,
            "answer": int(s.argmax()),
            "q": torch.softmax(s.float(), 0).tolist(),
            "parts": [p.name for p in r.parts],
            "partTexts": [p.text for p in r.parts],
            "hasSpans": r.has_spans,
            "spans": [dataclasses.asdict(x) for x in r.spans],
            "tokens": model.num_tokens(r),
            "sentences": [
                {
                    **dataclasses.asdict(x),
                    "gold": j in gold,
                    "text": r.parts[x.part].text[x.start : x.end],
                }
                for j, x in enumerate(sents)
            ],
            "seconds": seconds,
        }

    @app.get("/api/attribute")
    def attribute(id: str) -> dict[str, Any]:  # noqa: A002
        if not methods:
            raise fastapi.HTTPException(400, "no citation method loaded")
        r = get(id)
        out: dict[str, Any] = {}
        if cfg.citation:
            with lock:
                result, seconds = base.timed(
                    model.device, lambda: cite.cite(model, r, cfg.citation)
                )
            w = result.weights.tolist()
            top = cite.top_sentences(r, result, max(3, cfg.citation.top_k))
            out["contextcite"] = {
                "w": w,
                "top": [x["index"] for x in top[:3]],
                "evidence": top[: cfg.citation.top_k],
                "answer": result.answer,
                "seconds": seconds,
                "metrics": metrics.span_metrics(r, result.sentences, w)
                if r.has_spans and len(w) >= 2
                else None,
            }
        if evidence is not None:
            with lock:
                pred, seconds = base.timed(
                    model.device,
                    lambda: head_lib.predict(model, evidence.head, r),  # type: ignore[arg-type]
                )
            out["evidence"] = {
                "answer": pred.answer,
                "seconds": seconds,
                "threshold": evidence.threshold,
                "tokens": [
                    {"part": p, "start": a, "end": b, "p": float(v)}
                    for (p, a, b), v in zip(
                        pred.tokens.where, pred.probabilities, strict=True
                    )
                ],
            }
        return out

    @app.exception_handler(Exception)
    def failed(_: Any, exc: Exception) -> Any:
        logger.exception("request failed")
        return responses.JSONResponse(
            {"detail": f"{type(exc).__name__}: {exc}"}, status_code=500
        )

    return app


def main() -> None:
    """Loads the model and rows, then serves until interrupted."""
    import uvicorn

    cfg, _ = config.from_command_line(config.ServeConfig, __doc__)
    model = models.load(cfg.model)
    evidence = None
    if cfg.evidence:
        if not isinstance(model, encoder.MaskedEncoder):
            raise ValueError("evidence= needs model.kind=encoder")
        evidence = head_lib.load(cfg.evidence, model)
    rows = sampling.select(cfg.rows)
    logger.info(
        "serving {} rows on http://{}:{}/", len(rows), cfg.host, cfg.port
    )
    uvicorn.run(
        make_app(model, rows, cfg, evidence),
        host=cfg.host,
        port=cfg.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
