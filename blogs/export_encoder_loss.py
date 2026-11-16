"""Exports one row's mask logits for the blog's step-by-step encoder loss.

Scores one short eval row with UltraBERT-380M before training and with each
released recipe's run, and records logit(" yes") and logit(" no") at every
option's mask. The blog computes s, g(s), p, the loss and its gradient from
these, with the formulas of decision_models.losses.option.

Usage (one GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src .venv/bin/python blogs/export_encoder_loss.py
"""

import argparse
import dataclasses
import json
import pathlib

import torch

from decision_models import config
from decision_models.data import storage
from decision_models.models import encoder

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "blogs" / "data" / "encoder_loss.json"
SOURCE = "data/splits/reason_long_nimble"
# blog recipe key -> released run
RUNS = {
    "ub_list": "runs/ultrabert_380m_lr1.5e-5",
    "ub_yesno": "runs/ultrabert_380m_lr1.5e-5_yesno_seed1",
    "ub_sig": "runs/ultrabert_380m_lr1.5e-5_sigmoidnorm",
    "ub_sqrt": "runs/ultrabert_380m_lr1.5e-5_sqrtsoftplus",
}
SKIP_SOURCES = {"reclor"}  # not redistributed


def yes_no(model: encoder.MaskedEncoder, row) -> tuple[list[float], list[float]]:
    """logit(" yes") and logit(" no") at each option's mask, release order."""
    yes, no = encoder.verbalizer_ids(
        model.tokenizer, dataclasses.replace(model.fmt, no_token=" no")
    )
    ex, hidden, _ = model.forward_row(row)
    with torch.no_grad():
        logits = encoder.mask_logits(model.unwrapped.model, hidden[ex.masks])
    return logits[:, yes].tolist(), logits[:, no].tolist()


def candidates(rows, tokenizer, fmt, max_tokens: int, k_range: tuple[int, int]):
    for r in rows:
        if r.source.lower() in SKIP_SOURCES or not k_range[0] <= len(r.options) <= k_range[1]:
            continue
        if len(encoder.encode(r, tokenizer, fmt).input_ids) <= max_tokens:
            yield r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", help="row id (default: first short row the base misses and every run gets)")
    p.add_argument("--max-tokens", type=int, default=260)
    args = p.parse_args()

    trained = {k: encoder.MaskedEncoder.from_run(ROOT / v, attention="sdpa") for k, v in RUNS.items()}
    sig = trained["ub_sig"]
    base_cfg = dataclasses.replace(encoder.run_config(ROOT / RUNS["ub_sig"]).model, attention="sdpa")
    base = encoder.MaskedEncoder.build(base_cfg, sig.fmt, sig.device, torch.float32, "sigmoid_norm")
    base.eval()

    rows = storage.load_examples(str(ROOT / SOURCE), split="eval")
    if args.id:
        row = next(r for r in rows if r.id == args.id)
    else:
        for row in candidates(rows, sig.tokenizer, sig.fmt, args.max_tokens, (3, 5)):
            by, bn = yes_no(base, row)
            base_pick = max(range(len(by)), key=lambda k: by[k] - bn[k])
            ok = all(int(m.score([row])[0].argmax()) == row.gold for m in trained.values())
            if base_pick != row.gold and ok:
                break
        else:
            raise SystemExit("no row found")

    text, _ = encoder.build_prompt(row, list(range(len(row.options))), sig.tokenizer, sig.fmt)
    models = {"base": yes_no(base, row)} | {k: yes_no(m, row) for k, m in trained.items()}
    OUT.write_text(json.dumps({
        "row": {"id": row.id, "source": row.source, "options": list(row.options), "gold": row.gold,
                "prompt": text, "tokens": len(sig.encode(row).input_ids)},
        "models": {k: {"yes": y, "no": n} for k, (y, n) in models.items()},
    }, indent=1, ensure_ascii=False))
    print(f"{row.source} {row.id} ({len(row.options)} options) -> {OUT}")


if __name__ == "__main__":
    main()
