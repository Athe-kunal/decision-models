import dataclasses
import types

import pytest

from decision_models.data import adapter
from decision_models.data import storage

RECORDS = [
    adapter.Record("The sky is blue.", "Is it blue?", adapter.YES_NO, " YES "),
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


@pytest.mark.parametrize(
    "options, answer",
    [("yes/no", "maybe"), ("yes/no", True), (["A", "a"], "a")],
)
def test_bad_answers_are_rejected(options, answer) -> None:
    with pytest.raises(ValueError):
        adapter.Record("d", "q", options, answer).to_example("mine")


def test_save_drops_repeats_and_leaks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_JEV_DATA", str(tmp_path))
    leak = dataclasses.replace(RECORDS[1].to_example("mine"), id="mine:x")
    new = adapter.Record("d", "q", ["a", "b"], 0)
    folder = adapter.save(
        {"train": [*RECORDS, RECORDS[0]], "eval": [RECORDS[1], leak, new]},
        "mine",
    )
    assert folder == tmp_path / "splits" / "mine"
    assert storage.load_examples("mine", "train") == [
        r.to_example("mine") for r in RECORDS
    ]
    assert storage.load_examples("mine", "eval") == [new.to_example("mine")]
    with pytest.raises(ValueError, match="no rows"):
        adapter.save({"eval": []}, "mine")
