"""ContextCite by deletion, for any decision model.

Cohen-Wang et al., ContextCite (NeurIPS 2024), following
github.com/MadryLab/context-cite (MIT). The attributed answer is the model's
prediction on the full row. Random ablations keep about half of the
sentences, each ablated row is re-scored, and a sparse linear surrogate of
logit q(answer) gives one weight per sentence: how much keeping that
sentence adds to logit q(answer).
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import math

import numpy as np
import torch

from decision_models.data import schema
from decision_models.evidence import sentences as sentences_lib
from decision_models.models import base


@dataclasses.dataclass(frozen=True)
class Attribution:
    """Per-sentence weights for a model's answer.

    Attributes:
      answer: The attributed option (release order).
      q: softmax(s)[answer] on the full row.
      sentences: The row's sentences.
      weights: One weight per sentence; higher = more support.
      bias: The surrogate's intercept (ContextCite only).
    """

    answer: int
    q: float
    sentences: list[sentences_lib.Sentence]
    weights: np.ndarray
    bias: float = 0.0


def ablation_count(num_sentences: int, low: int = 32, high: int = 256) -> int:
    """About two ablations per sentence, a power of two in [low, high]."""
    return min(high, max(low, 1 << math.ceil(math.log2(2 * num_sentences))))


def logit_q(s: torch.Tensor, answer: int) -> float:
    """Logit q(answer) = log q - log(1 - q) from the option scores s [K].

    Computed as s_answer - logsumexp(other s), exact when q rounds to 1.
    """
    s = s.double()
    others = torch.cat([s[:answer], s[answer + 1 :]])
    return (s[answer] - others.logsumexp(0)).item()


# Scores rows of an example under keep masks [n, S] (True = sentence kept):
# one s [K] per mask, options in release order.
KeepScorer = Callable[
    [schema.Example, Sequence[sentences_lib.Sentence], np.ndarray],
    list[torch.Tensor],
]


def deletion_scorer(model: base.DecisionModel) -> KeepScorer:
    """Scores each mask by deleting the dropped sentences (the reference)."""

    def score(
        example: schema.Example,
        sentences: Sequence[sentences_lib.Sentence],
        keep: np.ndarray,
    ) -> list[torch.Tensor]:
        ablated = [
            sentences_lib.ablate(example, sentences, list(k)) for k in keep
        ]
        return model.score(ablated)

    return score


def ablation_targets(
    model: base.DecisionModel,
    example: schema.Example,
    sentences: Sequence[sentences_lib.Sentence],
    masks: np.ndarray,
    answer: int,
    scorer: KeepScorer | None = None,
) -> np.ndarray:
    """[n] logit q(answer | ablated row) for every mask (True = kept).

    Options stay in release order, so `answer` is the same option under
    every mask. `scorer` defaults to deletion.
    """
    scorer = scorer or deletion_scorer(model)
    return np.array(
        [logit_q(s, answer) for s in scorer(example, sentences, masks)]
    )


def fit_surrogate(
    masks: np.ndarray, y: np.ndarray, alpha: float = 0.01
) -> tuple[np.ndarray, float]:
    """Sparse linear surrogate y ~ b + w . mask: one weight per sentence.

    Lasso on standardized masks, with w and b mapped back to 0/1 masks.
    """
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    lasso = Lasso(alpha=alpha, random_state=0, fit_intercept=True)
    make_pipeline(scaler, lasso).fit(masks.astype(np.float32), y)
    weight = lasso.coef_ / scaler.scale_
    bias = lasso.intercept_ - (scaler.mean_ / scaler.scale_) @ lasso.coef_.T
    return weight, float(bias)


def attribute_with(
    scorer: KeepScorer,
    example: schema.Example,
    num_ablations: int | None = None,
    keep_prob: float = 0.5,
    alpha: float = 0.01,
    seed: int = 0,
) -> Attribution:
    """ContextCite on one row, scoring masks with `scorer`.

    The full row and every ablation are scored in one `scorer` call; the
    full row's decision is the attributed answer.

    Args:
      scorer: Scores the rows under keep masks (e.g. `deletion_scorer`).
      example: The row.
      num_ablations: Ablations (None: `ablation_count` of the sentences).
      keep_prob: Probability that an ablation keeps a sentence.
      alpha: Lasso regularisation.
      seed: Seeds the ablation masks.
    """
    sentences = sentences_lib.split(example)
    keep = np.ones((1, len(sentences)), bool)  # Row 0: the full example.
    if sentences:
        n = num_ablations or ablation_count(len(sentences))
        masks = sentences_lib.ablation_masks(n, len(sentences), keep_prob, seed)
        keep = np.vstack([keep, masks])
    scores = scorer(example, sentences, keep)
    answer, q = base.decide(scores[0])
    if not sentences:
        return Attribution(answer, q, [], np.zeros(0))
    y = np.array([logit_q(s, answer) for s in scores[1:]])
    weights, bias = fit_surrogate(keep[1:], y, alpha)
    return Attribution(answer, q, sentences, weights, bias)


def attribute(
    model: base.DecisionModel,
    example: schema.Example,
    num_ablations: int | None = None,
    keep_prob: float = 0.5,
    alpha: float = 0.01,
    seed: int = 0,
) -> Attribution:
    """ContextCite on one row by deleting sentences (any decision model)."""
    return attribute_with(
        deletion_scorer(model), example, num_ablations, keep_prob, alpha, seed
    )
