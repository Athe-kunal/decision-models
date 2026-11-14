"""Decisions and ContextCite through an SGLang server (its /v1/score API).

Usage (a server from scripts/sglang_serve.sh, or: make sglang-serve):
  decision-models-sglang tokenizer=runs/qwen3_4b/merged \
      datasets=[contractnli,musique] per=5

Prompts are tokenised here and sent as token ids, so the server scores the
merged model (inference.merge) exactly as trained. SGLang's /v1/decisions
endpoint is not used: it renders its own prompt. ContextCite here deletes
the dropped sentences: the radix cache reuses each ablated prompt's prefix,
and the trie's deeper nodes are sent after the prompts so they hit the cache.
"""

from collections.abc import Sequence
import concurrent.futures
import json
import pathlib
import random
import time
from typing import Any
import urllib.request

from loguru import logger
import numpy as np
import torch

from decision_models import config
from decision_models.citation import context_cite
from decision_models.data import schema
from decision_models.data import storage
from decision_models.evidence import metrics
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import decoder


def post(url: str, payload: dict[str, Any], timeout: float = 600) -> Any:
    """POSTs `payload` as JSON to `url`; returns the JSON reply."""
    request = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read())


def score_prompts(
    url: str,
    tokenizer: Any,
    prompts: Sequence[list[int]],
    ks: Sequence[int],
    chunk: int = 32,
    workers: int = 8,
) -> list[torch.Tensor]:
    """S [K] per prompt, through the server.

    prompts[i] ends at "Best answer: [" and has ks[i] options. Items are sent
    one trie depth at a time so the server's prefix cache reuses prompts.
    """
    depths: dict[int, list[tuple[int, tuple[int, ...], list[int]]]] = {}
    for i, (p, k) in enumerate(zip(prompts, ks, strict=True)):
        for node in decoder.candidates(tokenizer, k)[1]:
            depths.setdefault(len(node), []).append((i, node, p + list(node)))
    labels = sorted(
        {
            t
            for k in set(ks)
            for legal in decoder.candidates(tokenizer, k)[1].values()
            for t in legal
        }
    )

    def request(items: list[list[int]]) -> list[list[float]]:
        reply = post(
            f"{url}/v1/score",
            {
                "query": [],
                "items": items,
                "label_token_ids": labels,
                "return_token_logprobs": True,
            },
        )
        return reply["token_logprobs"]

    owners: list[tuple[int, tuple[int, ...]]] = []
    scores: list[list[float]] = []
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        for _, level in sorted(depths.items()):
            parts = [
                [x[2] for x in level[c : c + chunk]]
                for c in range(0, len(level), chunk)
            ]
            scores += [row for part in pool.map(request, parts) for row in part]
            owners += [(i, node) for i, node, _ in level]
    width = labels[-1] + 1
    logp: list[dict[tuple[int, ...], torch.Tensor]] = [{} for _ in prompts]
    for (i, node), row in zip(owners, scores, strict=True):
        lp = torch.full((width,), -torch.inf, dtype=torch.float64)
        lp[labels] = torch.tensor(row, dtype=torch.float64)
        logp[i][node] = lp
    out = []
    for lp, k in zip(logp, ks, strict=True):
        suffixes, nodes = decoder.candidates(tokenizer, k)
        out.append(decoder.read_trie(lp, suffixes, nodes))
    return out


def predict(
    url: str, tokenizer: Any, example: schema.Example
) -> tuple[int, torch.Tensor]:
    """The predicted option and the option scores s [K]."""
    prompt = decoder.prompt_ids(example, tokenizer)
    s = score_prompts(url, tokenizer, [prompt], [len(example.options)])[0]
    return int(s.argmax()), s


def attribute(
    url: str,
    tokenizer: Any,
    example: schema.Example,
    num_ablations: int | None = None,
    keep_prob: float = 0.5,
    alpha: float = 0.01,
    seed: int = 0,
) -> context_cite.Attribution:
    """`context_cite.attribute` through the server (sentences deleted)."""
    answer, s = predict(url, tokenizer, example)
    q = torch.softmax(s.float(), 0)[answer].item()
    sentences = sentences_lib.split(example)
    if not sentences:
        return context_cite.Attribution(answer, q, [], np.zeros(0))
    n = num_ablations or context_cite.ablation_count(len(sentences))
    masks = sentences_lib.ablation_masks(n, len(sentences), keep_prob, seed)
    prompts = [
        decoder.prompt_ids(
            sentences_lib.ablate(example, sentences, list(keep)), tokenizer
        )
        for keep in masks
    ]
    scored = score_prompts(url, tokenizer, prompts, [len(example.options)] * n)
    y = np.array([context_cite.logit_q(x, answer) for x in scored])
    weights, bias = context_cite.fit_surrogate(masks, y, alpha)
    return context_cite.Attribution(answer, q, sentences, weights, bias)


def main() -> None:
    """Attributes `per` seeded rows per dataset and logs the results."""
    from transformers import AutoTokenizer

    cfg, _ = config.from_command_line(config.SGLangConfig, __doc__)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    by_source: dict[str, list[schema.Example]] = {}
    for r in storage.load_examples(cfg.source, cfg.split):
        if (not cfg.datasets or r.source in cfg.datasets) and len(
            sentences_lib.split(r)
        ) >= 2:
            by_source.setdefault(r.source, []).append(r)
    out = open(cfg.out, "w") if cfg.out else None  # noqa: SIM115
    for source, rows in sorted(by_source.items()):
        for row in random.Random(0).sample(rows, min(cfg.per, len(rows))):
            start = time.time()
            result = attribute(cfg.url, tokenizer, row, cfg.num_ablations)
            seconds = time.time() - start
            logger.info(
                "{} {}: {} sentences, {:.2f} s, q {:.3f}, {}",
                source,
                row.id,
                len(result.sentences),
                seconds,
                result.q,
                metrics.span_metrics(row, result.sentences, result.weights),
            )
            if out:
                line = {
                    "id": row.id,
                    "seconds": seconds,
                    "answer": result.answer,
                    "q": result.q,
                    "w": result.weights.tolist(),
                }
                out.write(json.dumps(line) + "\n")
    if out:
        out.close()
        logger.info("wrote {}", pathlib.Path(cfg.out))


if __name__ == "__main__":
    main()
