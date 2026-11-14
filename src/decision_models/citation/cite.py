"""ContextCite for any decision model, with the fastest backend that fits.

Decoders use one-pass packing (`masked`, removal "reader" or "full") after
`prepare`; encoders, and decoders with removal "delete", use
`context_cite.attribute` (packed on FA2 varlen for encoders).
"""

from typing import Any

from decision_models import config
from decision_models.citation import context_cite
from decision_models.citation import masked
from decision_models.data import schema
from decision_models.models import base
from decision_models.models import decoder
from decision_models.models import flash_attention

PACKED = ("reader", "full")


def _packed(model: base.DecisionModel, cfg: config.CiteConfig) -> bool:
    return isinstance(model, decoder.TrieDecoder) and cfg.removal in PACKED


def prepare(model: base.DecisionModel, cfg: config.CiteConfig) -> None:
    """Switches a decoder to packed attention (and fused kernels) once.

    Raises:
      ValueError: On an unknown removal.
    """
    if cfg.removal not in (*PACKED, "delete"):
        raise ValueError(
            f"removal is reader, full or delete, not {cfg.removal!r}"
        )
    if _packed(model, cfg):
        flash_attention.use_decoder(model.unwrapped)
        if cfg.fuse:
            decoder.fuse(model.unwrapped)


def cite(
    model: base.DecisionModel, example: schema.Example, cfg: config.CiteConfig
) -> context_cite.Attribution:
    """ContextCite on one row; call `prepare(model, cfg)` first."""
    if _packed(model, cfg):
        return masked.attribute(
            model,  # type: ignore[arg-type]
            example,
            num_ablations=cfg.num_ablations,
            removal=cfg.removal,
            budget=cfg.budget,
        )
    return context_cite.attribute(model, example, cfg.num_ablations)


def top_sentences(
    example: schema.Example, result: context_cite.Attribution, k: int = 3
) -> list[dict[str, Any]]:
    """The k highest-weighted sentences with w > 0, strongest first."""
    ranked = sorted(
        (i for i, w in enumerate(result.weights) if w > 0),
        key=lambda i: -result.weights[i],
    )[:k]
    out = []
    for i in ranked:
        s = result.sentences[i]
        out.append(
            {
                "part": s.part,
                "start": s.start,
                "end": s.end,
                "w": float(result.weights[i]),
                "text": example.parts[s.part].text[s.start : s.end],
            }
        )
    return out
