import random

import pytest

from decision_models.data import adapter
from decision_models.data import storage

RECORDS = [
    adapter.Record(
        "The sky is blue.", "Is the sky blue?", adapter.YES_NO, "yes"
    ),
    adapter.Record("Dogs bark.", "Which animal barks?", ["Cats", "Dogs"], 1),
    adapter.Record("", "What is 2 + 2?", ["3", "4"], "4"),
]


def test_records_become_examples() -> None:
    yes_no, multi, bare = (adapter.to_example(r, "mine") for r in RECORDS)
    assert yes_no.options == ("Yes", "No") and yes_no.gold == 0
    assert multi.gold == 1 and multi.parts[0].text == "Dogs bark."
    assert bare.parts == () and bare.gold == 1


@pytest.mark.parametrize(
    "options, answer", [(adapter.YES_NO, "maybe"), (["a", "b"], 2), (["a"], 0)]
)
def test_bad_records_are_rejected(options, answer) -> None:
    with pytest.raises(ValueError):
        adapter.to_example(adapter.Record("d", "q", options, answer), "mine")


def test_save_round_trips(tmp_path) -> None:
    adapter.save({"train": RECORDS}, tmp_path, "mine")
    assert storage.load_examples(str(tmp_path), "train") == [
        adapter.to_example(r, "mine") for r in RECORDS
    ]


@pytest.mark.parametrize("model", ["masked_encoder", "trie_decoder"])
def test_collate_matches_training_encoding(model, request) -> None:
    model = request.getfixturevalue(model)
    got = adapter.collate(RECORDS, model, random.Random(3), "mine")
    rng = random.Random(3)
    want = [model.encode(adapter.to_example(r, "mine"), rng) for r in RECORDS]
    assert got == want
