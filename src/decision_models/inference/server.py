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

import collections
import dataclasses
import json
import pathlib
import random
import threading
import time
from typing import Any

from loguru import logger
import torch

from decision_models import config
from decision_models import models
from decision_models.citation import cite
from decision_models.data import schema
from decision_models.data import storage
from decision_models.evidence import head as head_lib
from decision_models.evidence import metrics
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base
from decision_models.models import encoder

PAGE = pathlib.Path(__file__).with_name("page.html")


@dataclasses.dataclass
class Evidence:
    """A loaded evidence head and its word threshold."""

    head: head_lib.EvidenceHead
    threshold: float


def load_evidence(
    run: str | pathlib.Path, model: base.DecisionModel
) -> Evidence:
    """The evidence head of a `decision-models-train-evidence` run.

    Raises:
      ValueError: If the model is not an encoder.
    """
    if not isinstance(model, encoder.MaskedEncoder):
        raise ValueError("the evidence head needs an encoder model")
    run = pathlib.Path(run)
    cfg_file = next(run.glob("config*.yaml"))
    cfg: config.EvidenceConfig = config.resolve(
        config.load(config.EvidenceConfig, cfg_file)
    )  # type: ignore[assignment]
    hidden = model.unwrapped.model.config.hidden_size
    head = head_lib.EvidenceHead(hidden, cfg.head.width, cfg.head.dropout)
    head.load_state_dict(torch.load(run / "head.pt", map_location="cpu"))
    threshold_files = sorted(run.glob("threshold*.json"))
    threshold = (
        json.loads(threshold_files[0].read_text())["threshold"]
        if threshold_files
        else 0.5
    )
    return Evidence(head.to(model.device).eval(), threshold)


def _timed(model: base.DecisionModel, fn: Any) -> tuple[Any, float]:
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def shuffled(
    example: schema.Example, seed: str
) -> tuple[schema.Example, list[int]]:
    """The example with its options in a seeded order, and that order."""
    order = list(range(len(example.options)))
    random.Random(seed).shuffle(order)
    return (
        dataclasses.replace(
            example,
            options=tuple(example.options[i] for i in order),
            gold=order.index(example.gold),
        ),
        order,
    )


def make_app(
    model: base.DecisionModel,
    rows: list[schema.Example],
    cfg: config.ServeConfig,
    evidence: Evidence | None = None,
) -> Any:
    """The FastAPI app over a loaded model and its rows."""
    import fastapi
    from fastapi import responses

    by_id = {r.id: r for r in rows}
    by_source: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        by_source[r.source].append(r.id)
    methods = (["contextcite"] if cfg.cite else []) + (
        ["evidence"] if evidence else []
    )
    info = {
        "model": cfg.model.path,
        "kind": cfg.model.kind,
        "adapter": cfg.model.adapter,
        "data": f"{cfg.source} ({cfg.split})",
        "methods": methods,
        "removal": cfg.citation.removal if cfg.cite else None,
    }
    if cfg.cite:
        cite.prepare(model, cfg.citation)
    lock = threading.Lock()  # One model, one request at a time.
    app = fastapi.FastAPI()

    def get(row_id: str) -> schema.Example:
        if row_id not in by_id:
            raise fastapi.HTTPException(404, f"unknown row {row_id}")
        return by_id[row_id]

    @app.get("/", response_class=responses.HTMLResponse)
    def page() -> str:
        return PAGE.read_text()

    @app.get("/api/info")
    def get_info() -> dict[str, Any]:
        return info

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        return [
            {
                "id": s,
                "rows": len(ids),
                "withSpans": sum(by_id[i].has_spans for i in ids),
            }
            for s, ids in sorted(by_source.items())
        ]

    @app.get("/api/sample")
    def sample(source: str = "", spans: int = 0) -> dict[str, str]:
        pool = [
            i
            for s, ids in by_source.items()
            if not source or s == source
            for i in ids
            if not spans or by_id[i].has_spans
        ]
        if not pool:
            raise fastapi.HTTPException(404, "no rows match")
        return {"id": random.choice(pool)}

    @app.get("/api/row")
    def row(id: str, shuffle: int = 0, seed: int = 0) -> dict[str, Any]:  # noqa: A002
        r = get(id)
        shown, order = (
            shuffled(r, f"{id}:{seed}")
            if shuffle
            else (r, list(range(len(r.options))))
        )
        with lock:
            s, seconds = _timed(model, lambda: model.score([shown])[0])
        sents = sentences_lib.split(r)
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
        if cfg.cite:
            with lock:
                result, seconds = _timed(
                    model, lambda: cite.cite(model, r, cfg.citation)
                )
            w = result.weights.tolist()
            out["contextcite"] = {
                "w": w,
                "top": sorted(
                    (j for j, v in enumerate(w) if v > 0), key=lambda j: -w[j]
                )[:3],
                "evidence": cite.top_sentences(r, result, cfg.citation.top_k),
                "answer": result.answer,
                "seconds": seconds,
                "metrics": metrics.span_metrics(r, result.sentences, w)
                if r.has_spans and len(w) >= 2
                else None,
            }
        if evidence is not None:
            with lock:
                pred, seconds = _timed(
                    model,
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
    evidence = load_evidence(cfg.evidence, model) if cfg.evidence else None
    rows = storage.load_examples(cfg.source, cfg.split)
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
