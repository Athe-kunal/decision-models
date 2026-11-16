"""An evidence head on a frozen encoder: which tokens support an answer.

The encoder reads the row once; its final hidden states are bidirectional,
so every document token has already seen the question and the options. The
head scores token t for answer k from [h_t ; h_k ; h_t * h_k], h_k being
the hidden state at option k's mask. The encoder stays frozen, so its
decisions are unchanged and one forward gives the decision and the evidence.

Unlike AT2 or ContextCite, which explain what the model used, the head is
trained on evidence spans (gold, or generated), so it predicts what an
annotator would mark.
"""

from collections.abc import Sequence
import dataclasses
import json
import pathlib

import torch
from torch import nn

from decision_models import config
from decision_models.data import schema
from decision_models.models import encoder


class EvidenceHead(nn.Module):
    """MLP over [h_t ; h_answer ; h_t * h_answer] -> one logit per token."""

    def __init__(
        self, hidden: int, width: int = 512, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(
        self, tokens: torch.Tensor, answer: torch.Tensor
    ) -> torch.Tensor:
        """Tokens [T, d], answer [d] -> logits [T]."""
        a = answer.expand_as(tokens)
        return self.mlp(torch.cat([tokens, a, tokens * a], -1))[..., 0]


@dataclasses.dataclass(frozen=True)
class DocumentTokens:
    """The document tokens of a release-order prompt.

    Attributes:
      index: Token positions in the prompt.
      where: (part, start, end) of each token in its part's text.
      labels: [T] float, 1 where the token overlaps an evidence span.
    """

    index: list[int]
    where: list[tuple[int, int, int]]
    labels: torch.Tensor


def document_tokens(
    model: encoder.MaskedEncoder,
    example: schema.Example,
    spans: Sequence[schema.Span] = (),
) -> DocumentTokens:
    """The document tokens of `example`, labelled against `spans`."""
    tokens = encoder.document_tokens(example, model.tokenizer, model.fmt)
    labels = torch.tensor(
        [float(schema.overlaps_any(spans, *where)) for _, where in tokens]
    )
    return DocumentTokens(
        [t for t, _ in tokens], [w for _, w in tokens], labels
    )


@dataclasses.dataclass(frozen=True)
class Prediction:
    """The head's evidence for one answer.

    Attributes:
      answer: The option the evidence is for (release order).
      q: softmax(s)[answer].
      tokens: The document tokens.
      probabilities: [T] evidence probability per document token.
    """

    answer: int
    q: float
    tokens: DocumentTokens
    probabilities: torch.Tensor


def build(
    model: encoder.MaskedEncoder, cfg: config.EvidenceHeadConfig
) -> EvidenceHead:
    """A fresh head for `model`'s hidden size, on its device."""
    hidden = model.unwrapped.model.config.hidden_size
    return EvidenceHead(hidden, cfg.width, cfg.dropout).to(model.device)


@dataclasses.dataclass(frozen=True)
class Trained:
    """A trained head and the word threshold tuned for it."""

    head: EvidenceHead
    threshold: float


# Evaluation tags of a run (train.py writes config<tag>.yaml and
# threshold<tag>.json), in the order `load` prefers them: first the recipe's
# (reported on test, tuned on dev), the one scripts/release_encoders.py ships.
TAGS = ("_test", "", "_dev")


def load(
    run: str | pathlib.Path,
    model: encoder.MaskedEncoder,
    tag: str | None = None,
) -> Trained:
    """The head of a `decision-models-train-evidence` run, in eval mode.

    Args:
      run: The run directory.
      model: The encoder the head reads.
      tag: Which evaluation's threshold<tag>.json to use ("_test", "" or
        "_dev"; default: the first of TAGS the run has). 0.5 if none.
    """
    run = pathlib.Path(run)
    tags = [tag] if tag is not None else TAGS
    found = next(
        (t for t in tags if (run / f"threshold{t}.json").exists()), None
    )
    configs = [run / f"config{t}.yaml" for t in (found, *TAGS) if t is not None]
    cfg_file = next(f for f in configs if f.exists())
    cfg = config.load(config.EvidenceConfig, cfg_file)
    head = build(model, cfg.head)
    head.load_state_dict(torch.load(run / "head.pt", map_location=model.device))
    threshold = (
        json.loads((run / f"threshold{found}.json").read_text())["threshold"]
        if found is not None
        else 0.5
    )
    return Trained(head.eval(), threshold)


@torch.no_grad()
def predict(
    model: encoder.MaskedEncoder,
    head: EvidenceHead,
    example: schema.Example,
    answer: int | None = None,
) -> Prediction:
    """Evidence probabilities for `answer` (default: the predicted option)."""
    ex, hidden, s = model.forward_row(example)
    answer = int(s.argmax()) if answer is None else answer
    doc = document_tokens(model, example)
    if doc.index:
        logits = head(hidden[doc.index], hidden[ex.masks[answer]])
        p = torch.sigmoid(logits).float().cpu()
    else:
        p = torch.zeros(0)
    q = torch.softmax(s, 0)[answer].item()
    return Prediction(answer, q, doc, p)
