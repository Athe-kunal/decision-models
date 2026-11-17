import dataclasses
import types

import pytest

from decision_models.data import adapter
from decision_models.data import storage

RECORDS = [
    adapter.Record("The sky is blue.", "Is it blue?", adapter.YES_NO, "yes"),
    adapter.Record("Dogs bark.", "Which barks?", ["Cats", "Dogs"], 1),
    adapter.Record("", "What is 2 + 2?", ["3", "4"], "4"),
]


@dataclasses.dataclass(frozen=True)
class TrueFalse(adapter.Record):
    option_sets = types.MappingProxyType({"true/false": ("True", "False")})


def test_records_become_examples() -> None:
    yes_no, multi, bare = (r.to_example("mine") for r in RECORDS)
    assert yes_no.options == ("Yes", "No") and yes_no.gold == 0
    assert multi.gold == 1 and multi.parts[0].text == "Dogs bark."
    assert bare.parts == () and bare.gold == 1
    custom = TrueFalse("d", "q", "true/false", "false").to_example("mine")
    assert custom.options == ("True", "False") and custom.gold == 1


@pytest.mark.parametrize("answer", ["maybe", True])
def test_bad_answers_are_rejected(answer) -> None:
    with pytest.raises(ValueError):
        adapter.Record("d", "q", adapter.YES_NO, answer).to_example("mine")


def test_save_by_name_goes_to_the_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_JEV_DATA", str(tmp_path))
    example = RECORDS[0].to_example("other")
    folder = adapter.save({"train": [*RECORDS, example]}, "mine")
    assert folder == tmp_path / "splits" / "mine"
    assert storage.load_examples("mine", "train") == [
        *(r.to_example("mine") for r in RECORDS),
        example,
    ]
    with pytest.raises(FileNotFoundError):
        storage.load_dataset("mine", "eval")
