import numpy as np
import pytest
import torch

from decision_models.citation import context_cite
from decision_models.data import schema
from decision_models.evidence import head as head_lib
from decision_models.evidence import metrics
from decision_models.evidence import sentences

nltk_punkt = pytest.importorskip("nltk")


@pytest.fixture(scope="module")
def sents(example):
    try:
        return sentences.split(example)
    except LookupError:  # pragma: no cover - offline without punkt_tab
        pytest.skip("nltk punkt_tab is not available offline")


def test_split_and_ablate_round_trip(example, sents) -> None:
    assert len(sents) == 3
    assert sentences.ablate(example, sents, [True] * 3).parts == example.parts
    dropped = (
        sentences.ablate(example, sents, [True, False, True]).parts[0].text
    )
    assert "Grass" not in dropped and "Cats" in dropped


def test_has_content() -> None:
    assert sentences.has_content("The sky is blue.")
    assert not sentences.has_content("[ ] ____ :")


def test_span_metrics(example, sents) -> None:
    perfect = metrics.span_metrics(example, sents, [3.0, 1.0, 0.5])
    assert perfect is not None
    assert perfect["ap"] == 1.0 and perfect["top1"] == 1.0
    worst = metrics.span_metrics(example, sents, [0.0, 1.0, 2.0])
    assert worst is not None
    assert worst["ap"] == pytest.approx(1 / 3) and worst["top1"] == 0.0
    no_gold = schema.Example("x", "s", example.parts, "q?", ("a", "b"), 0)
    assert metrics.span_metrics(no_gold, sents, [1, 2, 3]) is None


def test_word_scores_take_the_max_token() -> None:
    words = metrics.word_scores(
        "ab cd", [(0, 1, 0.1), (1, 2, 0.7), (3, 5, 0.2)]
    )
    assert words == [(0, 2, 0.7), (3, 5, 0.2)]


def test_ablation_masks_are_seeded() -> None:
    a = sentences.ablation_masks(4, 6, seed=3)
    np.testing.assert_array_equal(a, sentences.ablation_masks(4, 6, seed=3))
    assert a.shape == (4, 6) and a.dtype == bool


def test_context_cite_on_an_encoder(masked_encoder, example, sents) -> None:
    result = context_cite.attribute(masked_encoder, example, num_ablations=16)
    assert result.answer == masked_encoder.predict(example)
    assert len(result.weights) == len(sents) == len(result.sentences)


def test_logit_q_is_stable() -> None:
    s = torch.tensor([50.0, 0.0, -1.0])
    assert context_cite.logit_q(s, 0) == pytest.approx(
        50 - np.log(1 + np.exp(-1)), rel=1e-6
    )


def test_evidence_head_predicts_every_document_token(
    masked_encoder, evidence_head, example
) -> None:
    pred = head_lib.predict(masked_encoder, evidence_head, example)
    labelled = head_lib.document_tokens(masked_encoder, example, example.spans)
    assert (
        len(pred.probabilities)
        == len(pred.tokens.index)
        == len(labelled.labels)
    )
    assert 0 < labelled.labels.sum() < len(labelled.labels)
    assert ((pred.probabilities >= 0) & (pred.probabilities <= 1)).all()


def test_held_out_masks_avoid_fit_masks() -> None:
    fit = sentences.ablation_masks(32, 5, seed=0)
    held = sentences.held_out_masks(32, 5, fit, seed=1_000_000)
    seen = {m.tobytes() for m in fit}
    assert len(held) == len({m.tobytes() for m in held})  # distinct
    assert not any(m.tobytes() in seen for m in held)
    # Long rows: the same masks as an unconstrained draw.
    long_fit = sentences.ablation_masks(64, 20, seed=0)
    np.testing.assert_array_equal(
        sentences.held_out_masks(32, 20, long_fit, seed=1_000_000),
        sentences.ablation_masks(32, 20, seed=1_000_000),
    )


def test_spans_resume_restores_a_cut_newline(tmp_path) -> None:
    from decision_models.evidence import spans

    out = tmp_path / "spans.jsonl"
    out.write_text('{"id": "a"}\n{"id": "b"}')  # killed before the newline
    assert spans.labelled_ids(out) == {"a", "b"}
    with open(out, "a") as f:
        f.write('{"id": "c"}\n')
    assert spans.labelled_ids(out) == {"a", "b", "c"}
    out.write_text('{"id": "a"}\n{"id": "b')  # killed mid-record
    assert spans.labelled_ids(out) == {"a"}
    assert out.read_text() == '{"id": "a"}\n'


def test_head_load_prefers_the_dev_tuned_threshold(
    tmp_path, masked_encoder, evidence_head
) -> None:
    from decision_models import config

    torch.save(evidence_head.state_dict(), tmp_path / "head.pt")
    for tag, value in (("", 0.2), ("_test", 0.7)):
        config.save(
            config.merge(
                config.EvidenceConfig,
                None,
                ["out=x", "encoder.run=r", "head.width=8"],
            ),
            tmp_path / f"config{tag}.yaml",
        )
        (tmp_path / f"threshold{tag}.json").write_text(
            f'{{"threshold": {value}}}'
        )
    assert head_lib.load(tmp_path, masked_encoder).threshold == 0.7
    assert head_lib.load(tmp_path, masked_encoder, tag="").threshold == 0.2
