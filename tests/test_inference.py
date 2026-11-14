import json

import pytest

from decision_models import config
from decision_models.evidence import head as head_lib
from decision_models.evidence import sentences as sentences_lib
from decision_models.inference import predict
from decision_models.inference import server
from decision_models.models import encoder

fastapi_testclient = pytest.importorskip("fastapi.testclient")


def _split(example):
    text = example.parts[0].text
    out, start = [], 0
    for piece in text.split(". "):
        piece = piece if piece.endswith(".") else piece + "."
        begin = text.index(piece, start)
        out.append(
            sentences_lib.Sentence(
                0, begin, begin + len(piece), text[start:begin]
            )
        )
        start = begin + len(piece)
    return out


@pytest.fixture
def client(monkeypatch, encoder_dir, encoder_format, example, tmp_path):
    monkeypatch.setattr(sentences_lib, "split", _split)
    model_cfg = config.EncoderModelConfig(id=str(encoder_dir), attention="sdpa")
    model = encoder.MaskedEncoder.build(model_cfg, encoder_format, "cpu")
    hidden = model.unwrapped.model.config.hidden_size
    evidence = server.Evidence(
        head_lib.EvidenceHead(hidden, width=8).eval(), 0.5
    )
    cfg = config.ServeConfig(
        model=config.ModelRef(kind="encoder", path="toy"),
        citation=config.CiteConfig(removal="delete", num_ablations=8),
    )
    app = server.make_app(model, [example], cfg, evidence)
    return fastapi_testclient.TestClient(app)


def test_page_and_info(client) -> None:
    assert "Decision Model Explorer" in client.get("/").text
    info = client.get("/api/info").json()
    assert info["methods"] == ["contextcite", "evidence"]
    assert client.get("/api/datasets").json() == [
        {"id": "toy", "rows": 1, "withSpans": 1}
    ]


def test_sample_row_and_attribute(client, example) -> None:
    row_id = client.get("/api/sample", params={"spans": 1}).json()["id"]
    assert row_id == example.id
    row = client.get("/api/row", params={"id": row_id, "shuffle": 1}).json()
    assert sorted(row["options"]) == sorted(example.options)
    assert row["options"][row["gold"]] == example.answer
    assert abs(sum(row["q"]) - 1) < 1e-4
    assert [s["gold"] for s in row["sentences"]] == [True, False, False]
    cited = client.get("/api/attribute", params={"id": row_id}).json()
    assert len(cited["contextcite"]["w"]) == 3
    assert cited["evidence"]["tokens"]
    assert client.get("/api/row", params={"id": "nope"}).status_code == 404


def test_predict_record(example) -> None:
    line = predict.record(example, 0, 0.9)
    assert line["correct"] and line["option"] == "Blue"
    json.dumps(line)
