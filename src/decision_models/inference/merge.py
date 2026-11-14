"""Merges a LoRA adapter into its base model and saves it (for SGLang).

Usage (or: make sglang-merge):
  decision-models-merge model=Qwen/Qwen3-4B adapter=runs/qwen3_4b/adapter \
      out=runs/qwen3_4b/merged

SGLang's score API cannot select a LoRA adapter, so it serves merged weights.
"""

import pathlib

from loguru import logger

from decision_models import config
from decision_models.models import decoder


def main() -> None:
    """Writes the merged model and tokenizer to `out`."""
    cfg, _ = config.from_command_line(config.MergeConfig, __doc__)
    model = decoder.TrieDecoder.for_inference(cfg.model, cfg.adapter, "cpu")
    out = pathlib.Path(cfg.out)
    model.network.save_pretrained(out)
    model.tokenizer.save_pretrained(out)
    logger.info("merged {} + {} -> {}", cfg.model, cfg.adapter, out)


if __name__ == "__main__":
    main()
