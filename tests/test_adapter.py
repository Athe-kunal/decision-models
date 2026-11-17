import dataclasses
import json
import random

import omegaconf
import pytest
import torch

from decision_models import config
from decision_models.data import adapter
from decision_models.data import schema
from decision_models.data import storage
from decision_models.models import decoder
from decision_models.models import encoder
from decision_models.training import trainer


def _records() -> list[adapter.Record]:
    return [
        adapter.Record(
            documents="The sky is blue today. Grass grows in the field.",
            question="Is the sky blue?",
            options=adapter.YES_NO,
            answer=True,
            evidence=["The sky is blue today."],
        ),
        adapter.Record(
            documents=["Cats sleep a lot.", "Dogs bark."],
            question="Which animal barks?",
            options=["Cats", "Dogs", "Fish"],
            answer="dogs",
        ),
        adapter.Record(
            documents="",
            question="What is 2 + 2?",
            options=["3", "4"],
            answer=1,
            id="arith-1",
        ),
    ]


def test_records_become_examples() -> None:
    yes_no, multi, bare = adapter.to_examples(_records(), "mine")
    assert yes_no.options == ("Yes", "No") and yes_no.gold == 0
    assert yes_no.parts == (schema.Part("Document", _records()[0].documents),)
    (span,) = yes_no.spans
    assert (
        yes_no.parts[0].text[span.start : span.end] == "The sky is blue today."
    )
    assert [p.name for p in multi.parts] == ["Document 1", "Document 2"]
    assert multi.gold == 1
    assert bare.parts == () and bare.id == "mine:arith-1"


@pytest.mark.parametrize(
    "options, answer",
    [
        (adapter.YES_NO, "maybe"),
        (["a", "b"], 2),
        (["a", "b"], True),
        (["only"], 0),
        ("yes or no", "yes"),
    ],
)
def test_bad_records_are_rejected(options, answer) -> None:
    rec = adapter.Record("doc", "q?", options, answer)
    with pytest.raises(ValueError):
        adapter.to_examples([rec], "mine")


def test_yes_no_answer_accepts_text_and_bool() -> None:
    for answer, gold in (("no", 1), ("Yes", 0), (False, 1), (0, 0)):
        assert adapter.Record("d", "q", adapter.YES_NO, answer).gold == gold


def test_duplicate_ids_are_rejected() -> None:
    rec = adapter.Record("d", "q", ["a", "b"], 0, id="x")
    with pytest.raises(ValueError, match="duplicate"):
        adapter.to_examples([rec, rec], "mine")


def test_save_round_trips_through_storage(tmp_path) -> None:
    counts = adapter.save(
        {"train": _records(), "eval": _records()[:1]}, tmp_path, "mine"
    )
    assert counts == {"train": 3, "eval": 1}
    rows = storage.load_examples(str(tmp_path), "train")
    assert rows == adapter.to_examples(_records(), "mine")


def test_encoder_collator_matches_training(
    masked_encoder, encoder_format
) -> None:
    from_model = adapter.Collator(masked_encoder, source="mine")
    from_tok = adapter.Collator.for_encoder(
        masked_encoder.tokenizer, encoder_format, source="mine"
    )
    assert from_model.kind == from_tok.kind == "encoder"
    examples = adapter.to_examples(_records(), "mine")
    for record, example in zip(_records(), examples, strict=True):
        out = from_model.collate_one(record, random.Random(3))
        want = masked_encoder.encode(example, random.Random(3))
        assert out.encoding == want
        assert from_tok.collate_one(record, random.Random(3)).encoding == want
        assert out.text.count(masked_encoder.tokenizer.mask_token) == len(
            example.options
        )
        assert example.options[out.encoding.order[out.gold]] == example.answer
    # The collated rows train: a finite loss per row.
    for example in examples:
        assert masked_encoder.loss(example, random.Random(0)).loss.isfinite()


def test_decoder_collator_matches_training(trie_decoder) -> None:
    collate = adapter.Collator(trie_decoder, source="mine")
    assert collate.kind == "decoder"
    examples = adapter.to_examples(_records(), "mine")
    for record, example in zip(_records(), examples, strict=True):
        out = collate.collate_one(record, random.Random(5))
        assert out.encoding == decoder.encode(
            example, trie_decoder.tokenizer, random.Random(5)
        )
        assert out.text.endswith(decoder.ANSWER_PREFIX)
        # The shown text is the one that was tokenized.
        prompt = trie_decoder.tokenizer(out.text, add_special_tokens=False)
        assert prompt["input_ids"] == out.encoding.prompt


def test_unshuffled_collation_keeps_release_order(masked_encoder) -> None:
    out = adapter.Collator(masked_encoder)(_records())
    assert [c.encoding.order for c in out] == [[0, 1], [0, 1, 2], [0, 1]]
    assert isinstance(out[0].encoding, encoder.MaskEncoding)


@pytest.mark.parametrize("kind", ["encoder", "decoder"])
def test_adapter_output_trains(
    kind, tmp_path, encoder_dir, decoder_dir, encoder_format
) -> None:
    records = [
        dataclasses.replace(r, id=f"{i}:{j}")
        for i in range(2)
        for j, r in enumerate(_records())
    ]
    adapter.save({"train": records, "eval": records}, tmp_path / "data", "mine")
    out = tmp_path / "run"
    common = {
        "data.source": str(tmp_path / "data"),
        "data.eval_per": 2,
        "data.max_tokens": 100000,
        "optim.batch_size": 2,
        "run.out": str(out),
        "run.max_steps": 1,
        "run.skip_base_eval": True,
        "run.mixed_precision": "'no'",
        "model.gradient_checkpointing": False,
    }
    if kind == "encoder":
        schema_cls = config.EncoderTrainConfig
        common |= {
            "model.id": str(encoder_dir),
            "model.attention": "sdpa",
            "format.yes_token": encoder_format.yes_token,
            "format.no_token": encoder_format.no_token,
            "optim.loss": "sigmoid_norm",
        }
    else:
        schema_cls = config.DecoderTrainConfig
        common |= {"model.id": str(decoder_dir), "model.lora_rank": 4}
    raw = config.merge(
        schema_cls, None, [f"{k}={v}" for k, v in common.items()]
    )
    cfg = omegaconf.OmegaConf.to_object(raw)
    if kind == "encoder":
        build = lambda device: encoder.MaskedEncoder.build(  # noqa: E731
            cfg.model, cfg.format, device, torch.float32, cfg.optim.loss
        )
    else:
        build = lambda device: decoder.TrieDecoder.for_training(cfg, device)  # noqa: E731
    trainer.launch(cfg, raw, build)
    metrics = json.loads((out / "eval_final.json").read_text())
    assert set(metrics) == {"mine", "ALL"}
