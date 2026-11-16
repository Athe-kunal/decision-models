import pytest

from decision_models import config
from decision_models.data import prompts
from decision_models.data import sampling
from decision_models.data import schema


def test_row_round_trip(example: schema.Example) -> None:
    assert schema.Example.from_row(example.to_row()) == example


def test_invalid_gold_is_rejected() -> None:
    with pytest.raises(ValueError, match="gold"):
        schema.Example("x", "s", (), "q?", ("a", "b"), gold=2)


def test_render_example_locates_parts(example: schema.Example) -> None:
    text, where = prompts.render_example(example, [2, 0, 1])
    ((start, end),) = where
    assert text[start:end] == example.parts[0].text
    assert (
        text.index("[1] Green") < text.index("[2] Blue") < text.index("[3] Red")
    )


def _rows(n_sources: int = 3, per: int = 10) -> list[schema.Example]:
    return [
        schema.Example(f"{s}:{i}", f"src{s}", (), "q?", ("a", "b"), 0)
        for s in range(n_sources)
        for i in range(per)
    ]


def test_dev_and_test_partition_the_split() -> None:
    rows = _rows()
    dev = sampling.select_part(rows, "dev")
    test = sampling.select_part(rows, "test")
    assert len(dev) == 3 * min(10, sampling.DEV_PER)
    assert {r.id for r in dev} | {r.id for r in test} == {r.id for r in rows}
    assert not {r.id for r in dev} & {r.id for r in test}


def test_balanced_round_robin() -> None:
    rows = _rows(3, 10)
    picked = sampling.balanced(rows, 7, seed=0)
    counts = {
        s: sum(r.source == s for r in picked) for s in ("src0", "src1", "src2")
    }
    assert sorted(counts.values()) == [2, 2, 3]


def _old_style_pick(rows, part, per, seed):
    """The original selection: restrict to the part's ids in row order, then draw."""
    keep = {r.id for r in sampling.select_part(rows, part)}
    return sampling.per_dataset([r for r in rows if r.id in keep], per, seed)


def test_part_then_per_draws_from_rows_in_their_order() -> None:
    # Regression: the dev sample's shuffled order must not change the draw.
    rows = _rows(3, 150)
    cfg = config.RowsConfig(part="dev", per=7, seed=3)
    picked = sampling.select_rows(rows, cfg)
    assert [r.id for r in picked] == [
        r.id for r in _old_style_pick(rows, "dev", 7, 3)
    ]


def test_part_keeps_row_order() -> None:
    rows = _rows(3, 150)
    picked = sampling.select_rows(rows, config.RowsConfig(part="dev"))
    position = {r.id: i for i, r in enumerate(rows)}
    assert [position[r.id] for r in picked] == sorted(
        position[r.id] for r in picked
    )


def test_rows_with_evidence_all_rows_grouped_by_dataset(monkeypatch) -> None:
    # Regression: every row (per=0) comes grouped by dataset, as before.
    from decision_models.evidence import sentences

    rows = [
        schema.Example(
            f"{s}:{i}",
            s,
            (schema.Part("P", "One two. Three four."),),
            "q?",
            ("a", "b"),
            0,
            spans=(schema.Span(0, 0, 3),),
        )
        for i in range(4)
        for s in ("zeta", "alpha")
    ]
    monkeypatch.setattr(sentences, "has_evidence", lambda r, cover=False: True)
    picked = sentences.rows_with_evidence(rows, per=0)
    assert [r.source for r in picked] == ["alpha"] * 4 + ["zeta"] * 4
    assert [r.id for r in picked[:4]] == [f"alpha:{i}" for i in range(4)]


def test_trainer_eval_rows_are_dev_rows_only() -> None:
    rows = _rows(3, 300)
    dev = sampling.select_part(rows, "dev")
    assert sampling.dev_rows(rows, sampling.DEV_PER, 7) == dev
    picked = sampling.dev_rows(rows, 40, seed=1)
    assert len(picked) == 3 * 40
    assert {r.id for r in picked} <= {r.id for r in dev}
    with pytest.raises(ValueError, match="dev part"):
        sampling.dev_rows(rows, sampling.DEV_PER + 1)
