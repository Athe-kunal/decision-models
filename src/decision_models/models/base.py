"""The interface every decision model implements.

A decision model reads an `Example` and gives one score s_k per option; its
decision is argmax_k s_k. Training, evaluation, the benchmarks and
ContextCite only use this interface, so they work for both model families:

- `TrieDecoder` (models.decoder): a causal LM; option k is the token path
  "k]" through a candidate trie after "Best answer: [".
- `MaskedEncoder` (models.encoder): a masked LM with one mask token in front
  of every option; s_k is read from the logits at option k's mask.
"""

import abc
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
import dataclasses
import pathlib
import random
import time
from typing import Any
from typing import TypeVar

import torch
from torch import nn

from decision_models.data import schema

T = TypeVar("T")


def default_device() -> str:
    """The default device: "cuda" when a GPU is visible, else "cpu"."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def timed(device: torch.device, fn: Callable[[], T]) -> tuple[T, float]:
    """(fn(), seconds), with the GPU synchronised around the call."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def decide(s: torch.Tensor) -> tuple[int, float]:
    """(argmax s, softmax(s) at it) from option scores s [K]."""
    answer = int(s.argmax())
    return answer, torch.softmax(s.float(), 0)[answer].item()


@dataclasses.dataclass(frozen=True)
class LossOutput:
    """The training loss of one row.

    Attributes:
      loss: Scalar to back-propagate.
      logs: Detached scalars to log (summed over rows, then averaged).
    """

    loss: torch.Tensor
    logs: dict[str, torch.Tensor]


class DecisionModel(abc.ABC):
    """Scores the options of an `Example` with a neural network.

    Attributes:
      network: The trainable module. The trainer replaces it by the module
        Accelerate returns (DDP or FSDP), so losses must call it, not a
        submodule of it.
      tokenizer: The model's tokenizer.
      device: Where inputs are placed.
    """

    def __init__(
        self, network: nn.Module, tokenizer: Any, device: str | torch.device
    ) -> None:
        self.network = network
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @property
    def checkpoint_name(self) -> str:
        """Folder name, inside the run directory, of the saved weights."""
        return "model"

    @property
    def unwrapped(self) -> nn.Module:
        """The network without its DDP / compile wrappers."""
        from accelerate.utils import extract_model_from_parallel

        return extract_model_from_parallel(self.network)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """The parameters the optimizer updates."""
        return (p for p in self.network.parameters() if p.requires_grad)

    @abc.abstractmethod
    def num_tokens(self, example: schema.Example) -> int:
        """Length of the encoded row, to drop rows over a token budget."""

    @abc.abstractmethod
    def score(self, examples: Sequence[schema.Example]) -> list[torch.Tensor]:
        """Option scores s [K] (fp32, CPU or device) per example.

        Options are in release order, so argmax s is an index into
        `example.options`. Runs without gradients.
        """

    @abc.abstractmethod
    def loss(self, example: schema.Example, rng: random.Random) -> LossOutput:
        """The training loss of one row; `rng` shuffles its options."""

    @abc.abstractmethod
    def save(self, path: pathlib.Path, accelerator: Any) -> None:
        """Saves the trained weights; every process must call it."""

    def score_with_mass(
        self, examples: Sequence[schema.Example]
    ) -> list[tuple[torch.Tensor, float | None]]:
        """(s [K], root legal mass) per example; None where it has no mass."""
        return [(s, None) for s in self.score(examples)]

    def generate_answer(
        self, example: schema.Example
    ) -> tuple[str, int | None] | None:
        """Unguided text answer and the option it names; None if unsupported."""
        del example
        return None

    def predict(self, example: schema.Example) -> int:
        """The index of the chosen option."""
        return int(self.score([example])[0].argmax())

    def train(self, mode: bool = True) -> None:
        """Sets the network's training mode."""
        self.network.train(mode)

    def eval(self) -> None:
        """Sets the network to evaluation mode."""
        self.network.eval()
