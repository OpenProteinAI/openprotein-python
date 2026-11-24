import hashlib
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from openprotein import auth_store
from openprotein.base import APISession
from openprotein.errors import HTTPError
from openprotein.jobs import JobsAPI
from openprotein.predictor.api import decode_predict

API = "https://api.example/"
ARTIFACT = "https://storage.example/signed"
PREDICTION_DATA = (
    Path(__file__).resolve().parents[1] / "data" / "model_result_contracts"
)


def response(status, url, body, content_type="application/json"):
    result = requests.Response()
    result.status_code = status
    result.url = url
    result.headers["Content-Type"] = content_type
    result._content = body
    result._content_consumed = True
    return result


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )
    login = MagicMock(status_code=200, ok=True, headers={})
    login.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 900,
    }
    with patch.object(APISession, "post", return_value=login):
        session = APISession("user@example.com", "password", backend=API)
    session._expires_at = time.time() + 10_000
    return JobsAPI(session)


@pytest.mark.parametrize(
    "status,detail", [(404, "output not available"), (502, "Bad Gateway")]
)
@pytest.mark.parametrize("paged", [False, True], ids=["item", "page"])
def test_affinity_http_errors_propagate(jobs, monkeypatch, status, detail, paged):
    path = "v1/data/results/job/affinity" + ("" if paged else "/0")

    def request(session, method, url, **kwargs):
        assert method == "GET"
        assert url == API + path
        return response(status, url, json.dumps({"detail": detail}).encode())

    monkeypatch.setattr(requests.Session, "request", request)
    with pytest.raises(HTTPError) as caught:
        if paged:
            list(jobs._iter_job_outputs("job", "affinity"))
        else:
            jobs._get_job_output("job", 0, "affinity")
    assert caught.value.status_code == status
    assert caught.value.response.json() == {"detail": detail}


def test_affinity_json_ref_decodes_for_item_and_page(jobs, monkeypatch):
    expected = {"affinity_pred_value": 1.2, "affinity_probability_binary": 0.8}
    payload = json.dumps(expected).encode()
    entry = {
        "position": 0,
        "unit_id": "unit",
        "state": "SUCCESS",
        "value": {
            "ref": {
                "uri": "s3://content/affinity",
                "byte_length": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "codec": "json-utf8-v1",
                "download_url": ARTIFACT,
            }
        },
    }
    downloads = []

    def request(session, method, url, **kwargs):
        assert method == "GET"
        if url == ARTIFACT:
            downloads.append(url)
            assert session.auth is None
            assert "Authorization" not in session.headers
            assert "Authorization" not in kwargs.get("headers", {})
            return response(200, url, payload)
        if url == API + "v1/data/results/job/affinity/0":
            body = entry
        else:
            assert url == API + "v1/data/results/job/affinity"
            body = {"values": [entry], "next_cursor": None}
        return response(200, url, json.dumps(body).encode())

    monkeypatch.setattr(requests.Session, "request", request)
    assert jobs._get_job_output("job", 0, "affinity").value == expected
    assert [output.value for output in jobs._iter_job_outputs("job", "affinity")] == [
        expected
    ]
    assert downloads == [ARTIFACT, ARTIFACT]


def test_prediction_inline_values_preserve_quoted_property(jobs, monkeypatch):
    expected = json.loads((PREDICTION_DATA / "quoted_predictions.json").read_text())
    assert expected == [{'p"q': {"mean": 1.5, "variance": 2.5}}]
    entry = {"position": 0, "unit_id": "unit", "value": {"inline": expected[0]}}

    def request(session, method, url, **kwargs):
        assert method == "GET"
        if url == API + "v1/data/results/job/predictions/0":
            body = entry
        else:
            assert url == API + "v1/data/results/job/predictions"
            body = {"values": [entry], "next_cursor": None}
        return response(200, url, json.dumps(body).encode())

    monkeypatch.setattr(requests.Session, "request", request)
    item = jobs._get_job_output("job", 0, "predictions")
    page = list(jobs._iter_job_outputs("job", "predictions"))
    assert (item.position, item.value) == (0, expected[0])
    assert [(output.position, output.value) for output in page] == [(0, expected[0])]


def test_decode_predict_batched_quoted_columns():
    expected = json.loads((PREDICTION_DATA / "quoted_predictions.json").read_text())
    mean, variance = decode_predict(
        (PREDICTION_DATA / "quoted_predictions.csv").read_bytes(), batched=True
    )
    assert mean.shape == (1, 1)
    assert variance.shape == (1, 1)
    np.testing.assert_array_equal(mean, [[expected[0]['p"q']["mean"]]])
    np.testing.assert_array_equal(variance, [[expected[0]['p"q']["variance"]]])
