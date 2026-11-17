import dataclasses
import random
import types

import pytest

from decision_models.data import adapter
from decision_models.data import schema
from decision_models.data import storage

RECORDS = [
    adapter.Record(
        "The sky is blue.", "Is the sky blue?", adapter.YES_NO, "yes"
    ),
    adapter.Record("Dogs bark.", "Which animal barks?", ["Cats", "Dogs"], 1),
    adapter.Record("", "What is 2 + 2?", ["3", "4"], "4"),
]


@dataclasses.dataclass(frozen=True)
class WithEvidence(adapter.Record):
    """A record extended with one evidence quote."""

    quote: str = ""

    def to_example(self, source: str) -> schema.Example:
        example = super().to_example(source)
        start = self.documents.index(self.quote)
        span = schema.Span(0, start, start + len(self.quote))
        return dataclasses.replace(example, spans=(span,))


@dataclasses.dataclass(frozen=True)
class TrueFalse(adapter.Record):
    """A record with one more named option set."""

    option_sets = types.MappingProxyType(
        {**adapter.Record.option_sets, "true/false": ("True", "False")}
    )


def test_records_become_examples() -> None:
    yes_no, multi, bare = (adapter.to_example(r, "mine") for r in RECORDS)
    assert yes_no.options == ("Yes", "No") and yes_no.gold == 0
    assert multi.gold == 1 and multi.parts[0].text == "Dogs bark."
    assert bare.parts == () and bare.gold == 1


@pytest.mark.parametrize(
    "options, answer",
    [
        (adapter.YES_NO, "maybe"),
        (adapter.YES_NO, True),
        (["a", "b"], 2),
        (["a"], 0),
        ("true/false", 0),
    ],
)
def test_bad_records_are_rejected(options, answer) -> None:
    with pytest.raises(ValueError):
        adapter.to_example(adapter.Record("d", "q", options, answer), "mine")


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_JEV_DATA", str(tmp_path / "home"))
    return tmp_path / "home"


def test_save_by_name_goes_to_the_cache(data_home) -> None:
    folder = adapter.save({"train": RECORDS}, "mine")
    assert folder == data_home / "splits" / "mine"
    assert storage.load_examples("mine", "train") == [
        adapter.to_example(r, "mine") for r in RECORDS
    ]
    with pytest.raises(FileNotFoundError):
        storage.load_dataset("mine", "eval")


def test_save_to_a_folder(tmp_path, data_home) -> None:
    folder = adapter.save({"train": RECORDS}, "mine", out=tmp_path / "x")
    assert folder == tmp_path / "x" and not data_home.exists()
    assert len(storage.load_examples(str(folder), "train")) == len(RECORDS)


@pytest.mark.parametrize("model_name", ["masked_encoder", "trie_decoder"])
def test_collate_matches_training_encoding(model_name, request) -> None:
    model = request.getfixturevalue(model_name)
    got = adapter.collate(RECORDS, model, random.Random(3), "mine")
    rng = random.Random(3)
    want = [model.encode(adapter.to_example(r, "mine"), rng) for r in RECORDS]
    assert got == want


def test_subclasses_extend_records(data_home) -> None:
    evidence = WithEvidence("A. B.", "Q?", ["x", "y"], 0, quote="B.")
    true_false = TrueFalse("d", "q", "true/false", "false")
    example = adapter.to_example(RECORDS[0], "mine")
    adapter.save({"train": [evidence, true_false, example]}, "mine")
    got = storage.load_examples("mine", "train")
    assert got[0].spans == (schema.Span(0, 3, 5),)
    assert got[1].options == ("True", "False") and got[1].gold == 1
    assert got[2] == example
