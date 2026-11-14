"""Typed configs for the analyses (not part of the installed package).

They extend `decision_models.config` and load the same way: a YAML under
configs/analysis/ plus dotlist overrides.
"""

import dataclasses

from omegaconf import MISSING

from decision_models.config import DEFAULT_DATA
from decision_models.config import SEED
from decision_models.config import FrozenEncoderConfig


@dataclasses.dataclass
class AT2DataConfig:
    """Rows and ablations for AT2's training data (stage 1).

    Attributes:
      source: Data folder or Hub id.
      split: Never the eval split: it is for evaluation.
      datasets: Only these datasets (None: all).
      rows: Rows, balanced over the datasets.
      min_sources: Ablating needs at least two sentences.
      masks: Ablations per row.
      seed: Picks rows and ablations.
    """

    source: str = DEFAULT_DATA
    split: str = "train"
    datasets: list[str] | None = None
    rows: int = 2000
    min_sources: int = 2
    masks: int = 32
    seed: int = 0


@dataclasses.dataclass
class AT2FitConfig:
    """Fitting theta (stage 2), as the reference's estimator trainer."""

    steps: int = 1000
    batch_size: int = 512
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    val_frac: float = 0.1
    seed: int = 0


@dataclasses.dataclass
class AT2Config:
    """Everything one AT2 run reads."""

    encoder: FrozenEncoderConfig = dataclasses.field(
        default_factory=FrozenEncoderConfig
    )
    data: AT2DataConfig = dataclasses.field(default_factory=AT2DataConfig)
    train: AT2FitConfig = dataclasses.field(default_factory=AT2FitConfig)
    out: str = MISSING


@dataclasses.dataclass
class CitationConfig:
    """Compares AT2, average attention and ContextCite on gold evidence.

    Attributes:
      at2: AT2 run directory (its config names the encoder).
      source: Data folder or Hub id.
      split: Split with gold spans.
      per: Rows with gold evidence per dataset (0: all).
      part: Restrict to this part ("dev" / "test") of the held-out split.
      seed: Picks the rows.
    """

    at2: str = MISSING
    source: str = DEFAULT_DATA
    split: str = "eval"
    per: int = 50
    part: str | None = None
    seed: int = 0


@dataclasses.dataclass
class CouplingConfig:
    """How strongly encoder option probabilities depend on the other options.

    Attributes:
      runs: Encoder training run directories.
      source: Data folder or Hub id (None: each run's data.source).
      split: The split to read.
      part: Part of the split ("dev", "test", "all").
      eval_per: Rows per dataset of the dev sample.
      shuffles: Extra random option orders per row.
      seed: Seeds the sample and the shuffles.
      tag: Writes <run>/<tag>.json.
    """

    runs: list[str] = MISSING
    source: str | None = None
    split: str = "eval"
    part: str = "dev"
    eval_per: int = 100
    shuffles: int = 3
    seed: int = SEED
    tag: str = "coupling_dev"


@dataclasses.dataclass
class LatencyConfig:
    """Time and memory per decision: decoders against encoders.

    Attributes:
      decoders: Hub ids or paths (merged weights).
      encoders: Encoder training run directories.
      source: Data folder or Hub id.
      split: The split to read.
      eval_per: Rows per dataset.
      warmup: Untimed warm-up rows.
      out: Output JSON.
      merge: Keep the other models already in `out`.
    """

    decoders: list[str] = dataclasses.field(default_factory=list)
    encoders: list[str] = dataclasses.field(default_factory=list)
    source: str = DEFAULT_DATA
    split: str = "eval"
    eval_per: int = 100
    warmup: int = 20
    out: str = MISSING
    merge: bool = False
