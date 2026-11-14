"""ContextCite for any decision model, with the fastest backend that fits.

Decoders with removal "reader" or "full" use one-pass packing (`masked`)
after `prepare`; encoders, and decoders with removal "delete", delete the
dropped sentences (packed on FA2 varlen for encoders).
"""

from typing import Any

from decision_models import config
from decision_models.citation import context_cite
from decision_models.citation import masked
from decision_models.data import schema
from decision_models.models import base
from decision_models.models import decoder
from decision_models.models import flash_attention


def packed(model: base.DecisionModel, cfg: config.CiteConfig) -> bool:
    """Whether ContextCite runs one-pass packed masking for this model."""
    return isinstance(model, decoder.TrieDecoder) and cfg.removal != "delete"


def prepare(model: base.DecisionModel, cfg: config.CiteConfig) -> None:
    """Switches a decoder to packed attention (and fused kernels) once."""
    if packed(model, cfg):
        flash_attention.use_decoder(model.unwrapped)
        if cfg.fuse:
            decoder.fuse(model.unwrapped)


def scorer(
    model: base.DecisionModel,
    cfg: config.CiteConfig,
    exact: bool = False,
) -> context_cite.KeepScorer:
    """The keep-mask scorer ContextCite uses for `model` under `cfg`.

    With `exact`, packed decoders use full masking (equal to deletion) even
    when `cfg.removal` is "reader", e.g. to measure what a weighting predicts.
    """
    if packed(model, cfg):
        removal = "full" if exact else cfg.removal
        return masked.packed_scorer(model, removal, cfg.budget)  # type: ignore[arg-type]
    return context_cite.deletion_scorer(model)


def cite(
    model: base.DecisionModel, example: schema.Example, cfg: config.CiteConfig
) -> context_cite.Attribution:
    """ContextCite on one row; call `prepare(model, cfg)` first."""
    return context_cite.attribute_with(
        scorer(model, cfg), example, cfg.num_ablations
    )


def top_sentences(
    example: schema.Example, result: context_cite.Attribution, k: int = 3
) -> list[dict[str, Any]]:
    """The k highest-weighted sentences with w > 0, strongest first."""
    ranked = sorted(
        (i for i, w in enumerate(result.weights) if w > 0),
        key=lambda i: -result.weights[i],
    )[:k]
    return [
        {
            "index": i,
            "part": result.sentences[i].part,
            "start": result.sentences[i].start,
            "end": result.sentences[i].end,
            "w": float(result.weights[i]),
            "text": example.parts[result.sentences[i].part].text[
                result.sentences[i].start : result.sentences[i].end
            ],
        }
        for i in ranked
    ]
