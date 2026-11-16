import json

import pytest

from decision_models import config
from decision_models.evidence import head as head_lib
from decision_models.inference import predict
from decision_models.inference import server

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def client(offline_split, masked_encoder, evidence_head, example):
    cfg = config.ServeConfig(
        model=config.ModelRef(kind="encoder", path="toy"),
        citation=config.CiteConfig(removal="delete", num_ablations=8),
    )
    evidence = head_lib.Trained(evidence_head, 0.5)
    app = server.make_app(masked_encoder, [example], cfg, evidence)
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


def test_encoder_with_an_adapter_is_rejected() -> None:
    from decision_models import models

    ref = config.ModelRef(kind="encoder", path="runs/x", adapter="runs/a")
    with pytest.raises(ValueError, match="no adapter"):
        models.load(ref, "cpu")


def test_merged_model_with_an_adapter_is_rejected(tmp_path) -> None:
    from decision_models.models import decoder

    (tmp_path / decoder.MERGED_MARKER).write_text("{}")
    with pytest.raises(ValueError, match="already has an adapter"):
        decoder.TrieDecoder.for_inference(str(tmp_path), "runs/a", "cpu")
