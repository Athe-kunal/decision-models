import pytest

from decision_models import config
from decision_models.citation import evaluate as cite_evaluate
from decision_models.evaluation import evaluate


def test_decoder_legal_mass_and_text_mode(
    monkeypatch, trie_decoder, example
) -> None:
    # stop_strings generation needs a real tokenizer; stub the text reply.
    monkeypatch.setattr(trie_decoder, "generate_answer", lambda row: ("1]", 0))
    examples: list = []
    results = evaluate.score_rows(
        trie_decoder, [example], {example.id}, examples
    )
    (r,) = results
    assert r.root_legal_mass is not None
    assert 0.0 <= r.root_legal_mass <= 1.0
    assert r.gen_valid and r.gen_correct
    assert examples and examples[0]["id"] == example.id
    metrics = evaluate.summarize(results)
    assert {"root_legal_mass", "gen_valid", "gen_correct"} <= set(
        metrics["ALL"]
    )


def test_rows_are_scored_in_one_call(monkeypatch, trie_decoder, example):
    # Decoders length-sort the rows they get into padded batches, and bf16
    # scores depend on the padding: scoring in chunks changed test results.
    calls: list[int] = []
    score = trie_decoder.score_with_mass

    def spy(rows):
        calls.append(len(rows))
        return score(rows)

    monkeypatch.setattr(trie_decoder, "score_with_mass", spy)
    evaluate.score_rows(trie_decoder, [example] * 300)
    assert calls == [300]


def test_encoder_has_no_decoder_metrics(masked_encoder, example) -> None:
    metrics = evaluate.summarize(evaluate.score_rows(masked_encoder, [example]))
    assert set(metrics["ALL"]) == {"accuracy", "log_p_gold"}


def test_label_shares(masked_encoder, example) -> None:
    shares = evaluate.label_shares(masked_encoder, [example], shuffles=2)
    labels = shares["toy"]
    assert set(labels) == set(example.options)
    assert sum(v["gold"] for v in labels.values()) == pytest.approx(1.0)
    assert sum(v["predicted"] for v in labels.values()) == pytest.approx(1.0)


def test_context_cite_evaluation_row(
    offline_split, masked_encoder, example
) -> None:
    cfg = config.CiteConfig(removal="delete", num_ablations=8)
    row = cite_evaluate.evaluate_row(masked_encoder, example, cfg, lds_masks=8)
    assert row["sources"] == 3 and len(row["w"]) == 3
    assert {"ap", "top1", "lds", "drop@1", "seconds"} <= set(row)
    summary = cite_evaluate.summarize([row])
    assert summary["all"]["rows"] == 1 and "toy" in summary
    assert "all" in cite_evaluate.table(summary)
