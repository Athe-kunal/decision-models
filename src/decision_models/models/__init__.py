"""Decision models: a shared interface and its decoder and encoder families."""

import torch

from decision_models import config
from decision_models.models import base


def load(ref: config.ModelRef, device: str | None = None) -> base.DecisionModel:
    """A decision model for inference, from a `ModelRef`.

    Args:
      ref: Which model to load.
      device: Where to put it (default: base.default_device()).
    """
    device = device or base.default_device()
    if ref.kind == "decoder":
        from decision_models.models import decoder

        return decoder.TrieDecoder.for_inference(ref.path, ref.adapter, device)
    from decision_models.models import encoder

    return encoder.MaskedEncoder.from_run(
        ref.path, device, getattr(torch, ref.dtype), ref.attention
    )
