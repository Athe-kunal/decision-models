"""Encoder recipes: how option scores s [K] become a training loss.

Every recipe normalises a per-option value over the K options and takes
-log p_gold, so the options always compete and the decision is argmax s:

- "listwise": p_k = softmax(s)_k.
- "sigmoid_norm": p_k = sigmoid(s_k) / sum_j sigmoid(s_j), the form of
  DeepSeek-V3's expert gate.
- "sqrt_softplus": p_k = h(s_k) / sum_j h(s_j), h(s) = sqrt(softplus(s)),
  unbounded above, so an accepted option keeps a slowly fading gradient.

The last two need s = logit(yes) - logit(no), a log-odds near 0 where the
sigmoid curves. Raw MLM logits are in the hundreds, on a flat tail: very
negative, sigmoid(s) ~ e^s and the recipe collapses to a softmax; very
positive, sigmoid(s) ~ 1 and the gradient vanishes.
"""

from collections.abc import Callable

import torch
from torch import nn

from decision_models import config


def log_sqrt_softplus(s: torch.Tensor) -> torch.Tensor:
    """Log sqrt(softplus(s)), stable: softplus(s) ~ e^s for very negative s."""
    softplus = nn.functional.softplus(s)
    return 0.5 * torch.where(s < -20, s, softplus.clamp_min(1e-30).log())


# Recipe -> log g(s), whose normalisation over the options gives p [K].
LOG_WEIGHTS: dict[config.Loss, Callable[[torch.Tensor], torch.Tensor]] = {
    "listwise": lambda s: s,
    "sigmoid_norm": nn.functional.logsigmoid,
    "sqrt_softplus": log_sqrt_softplus,
}
NEEDS_NO_TOKEN: frozenset[config.Loss] = frozenset(
    {"sigmoid_norm", "sqrt_softplus"}
)


def probabilities(s: torch.Tensor, kind: config.Loss) -> torch.Tensor:
    """The recipe's distribution p [K] over the options."""
    return torch.softmax(LOG_WEIGHTS[kind](s), 0)


def loss(
    s: torch.Tensor, gold: int, kind: config.Loss = "listwise"
) -> torch.Tensor:
    """-log p_gold under recipe `kind`, from scores s [K]. Returns []."""
    log_g = LOG_WEIGHTS[kind](s)
    return log_g.logsumexp(0) - log_g[gold]
