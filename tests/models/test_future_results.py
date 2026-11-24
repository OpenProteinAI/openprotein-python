"""Fetching a models API job's results through Future."""

import hashlib
import io
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.errors import HTTPError
from openprotein.jobs import Job
from openprotein.jobs.futures import Future

MODEL_ID = "esm2_t33_650M_UR50D"

METADATA: dict[str, Any] = {
    "model_id": MODEL_ID,
    "model_family": "esm2",
    "name": None,
    "version": None,
    "is_system": True,
    "base": True,
    "tags": None,
    "protein_types": None,
    "description": {"summary": "ESM-2 650M protein language model."},
    "dimension": 1280,
    "query": None,
    "multichain": None,
    "methods": [
        {
            "path": "batch/embeddings",
            "outputs": ["embeddings", "annotations"],
            "description": "Per-residue embeddings.",
        },
    ],
    "max_sequence_length": 4094,
}

JOB: dict[str, Any] = {
    "job_id": "e0b1c3d4-0000-4000-8000-000000000001",
    "job_type": f"/models/{MODEL_ID}/batch/embeddings",
    "status": "SUCCESS",
    "created_date": "2026-09-05T08:00:00Z",
    "args": {"sequences": ["MKTAYIAK", "MEEP"]},
}

UNITS: list[dict[str, Any]] = [
    {"position": 0, "unit_id": "f0", "state": "SUCCESS"},
    {"position": 1, "unit_id": "f1", "state": "SUCCESS"},
]


def _vector_bytes(values: list[float]) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.array(values))
    return buf.getvalue()


def _get_side_effect(units: list[dict], metadata: dict):
    def handler(path: str, **kwargs):
        response = MagicMock()
        if path.endswith("/units"):
            response.json.return_value = {"units": units, "next_position": None}
        elif path == f"v1/models/{MODEL_ID}":
            response.json.return_value = metadata
        else:
            raise AssertionError(f"unexpected GET {path}")
        return response

    return handler


@pytest.fixture
def future(mock_session) -> Future:
    mock_session.get.side_effect = _get_side_effect(UNITS, METADATA)
    return Future(mock_session, Job.model_validate(JOB))


class TestArgs:
    def test_the_method_is_read_from_the_job_type(self, mock_session):
        job = Job.model_validate(JOB | {"args": {"provider_id": "other"}})
        future = Future(mock_session, job)

        assert (future._model_id, future._path) == (MODEL_ID, "batch/embeddings")
        mock_session.get.assert_not_called()

    def test_a_job_without_its_method_is_refreshed_once(self, mock_session):
        bare = JOB | {"job_type": "/models"}
        future = Future(mock_session, Job.model_validate(bare))
        mock_session.get.side_effect = None
        mock_session.get.return_value.json.return_value = JOB

        assert future._model_id == MODEL_ID
        assert future._path == "batch/embeddings"
        mock_session.get.assert_called_once_with(f"v1/jobs/{JOB['job_id']}")

    def test_missing_args_are_fetched_from_the_jobs_api_and_cached(self, mock_session):
        job_without_args = dict(JOB)
        job_without_args["args"] = None
        future = Future(mock_session, Job.model_validate(job_without_args))

        def handler(path, **kwargs):
            response = MagicMock()
            if path == f"v1/jobs/{JOB['job_id']}/args":
                response.json.return_value = JOB["args"]
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler

        assert future.args == JOB["args"]
        assert future.args == JOB["args"]

        args_calls = [
            c for c in mock_session.get.call_args_list if c.args[0].endswith("/args")
        ]
        assert len(args_calls) == 1


class TestUnits:
    def test_units_are_cached_after_one_call(self, future, mock_session):
        first = future.units
        second = future.units

        assert first == second
        assert [u.position for u in first] == [0, 1]
        units_calls = [
            c for c in mock_session.get.call_args_list if c.args[0].endswith("/units")
        ]
        assert len(units_calls) == 1


class TestDescriptor:
    def test_metadata_is_fetched_once_across_calls(self, future, mock_session):
        assert future.outputs == ["embeddings", "annotations"]
        assert future._default_output == "embeddings"
        assert future._descriptor.path == "batch/embeddings"

        model_calls = [
            c
            for c in mock_session.get.call_args_list
            if c.args[0] == f"v1/models/{MODEL_ID}"
        ]
        assert len(model_calls) == 1


def _model_listing_calls(mock_session) -> list:
    return [
        c for c in mock_session.get.call_args_list if c.args[0].startswith("v1/models")
    ]


def _not_found() -> HTTPError:
    response = MagicMock(status_code=404)
    response.json.return_value = {"detail": "no such output"}
    return HTTPError(response)


def _unauthorized() -> HTTPError:
    response = MagicMock(status_code=401)
    response.json.return_value = {"detail": "not authorized"}
    return HTTPError(response)


class TestGetItem:
    def test_without_output_resolves_the_default_from_the_bare_route(
        self, future, mock_session
    ):
        job_id = JOB["job_id"]

        def handler(path, **kwargs):
            response = MagicMock(status_code=200)
            if path == f"v1/data/results/{job_id}":
                assert kwargs == {"params": {"limit": 1}}
                response.json.return_value = {
                    "values": [],
                    "next_cursor": None,
                    "output": "annotations",
                }
            elif path.startswith(f"v1/data/results/{job_id}/annotations/"):
                assert kwargs == {}
                response.json.return_value = {
                    "position": int(path.rsplit("/", 1)[1]),
                    "unit_id": "u",
                    "value": {"inline": "some text"},
                }
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler

        assert future.get_item(0) == "some text"
        assert future.get_item(1) == "some text"

        paths = [c.args[0] for c in mock_session.get.call_args_list]
        assert paths.count(f"v1/data/results/{job_id}") == 1
        assert _model_listing_calls(mock_session) == []

    def test_with_output_reads_the_named_route_without_the_model_listing(
        self, future, mock_session
    ):
        job_id = JOB["job_id"]

        def handler(path, **kwargs):
            if path == f"v1/data/results/{job_id}/annotations/1":
                response = MagicMock()
                response.json.return_value = {
                    "position": 1,
                    "unit_id": "u",
                    "value": {"inline": "some text"},
                }
                return response
            raise AssertionError(f"unexpected GET {path}")

        mock_session.get.side_effect = handler

        assert future.get_item(1, output="annotations") == "some text"
        assert _model_listing_calls(mock_session) == []

    def test_an_output_the_method_lacks_is_the_servers_404(self, future, mock_session):
        mock_session.get.side_effect = _not_found()

        with pytest.raises(HTTPError) as error:
            future.get_item(0, output="not-a-real-output")

        assert error.value.status_code == 404
        mock_session.get.assert_called_once_with(
            f"v1/data/results/{JOB['job_id']}/not-a-real-output/0"
        )

    def test_missing_default_output_is_not_guessed(self, future, mock_session):
        response = MagicMock(status_code=200)
        response.json.return_value = {"values": [], "next_cursor": None}
        mock_session.get.side_effect = None
        mock_session.get.return_value = response

        with pytest.raises(ValueError, match="do not name their output"):
            future.get_item(0)

        mock_session.get.assert_called_once_with(
            f"v1/data/results/{JOB['job_id']}", params={"limit": 1}
        )

    def test_default_resolution_propagates_authorization_failures(
        self, future, mock_session
    ):
        mock_session.get.side_effect = _unauthorized()

        with pytest.raises(HTTPError) as error:
            future.get_item(0)

        assert error.value.status_code == 401
        mock_session.get.assert_called_once_with(
            f"v1/data/results/{JOB['job_id']}", params={"limit": 1}
        )


class TestStream:
    def test_stream_decodes_canonical_references_in_position_order(
        self, future, mock_session, monkeypatch
    ):
        job_id = JOB["job_id"]
        payloads = [_vector_bytes([1.0]), _vector_bytes([2.0])]
        from openprotein.jobs._storage import DirectArtifactStorage

        monkeypatch.setattr(
            DirectArtifactStorage,
            "iter_bytes",
            lambda self, ref: iter([payloads[int(ref["uri"].rsplit("/", 1)[1])]]),
        )

        def handler(path, **kwargs):
            response = MagicMock(status_code=200)
            if path == f"v1/data/results/{job_id}":
                assert kwargs == {"params": {"limit": 1}}
                response.json.return_value = {
                    "output": "embeddings",
                    "values": [{"value": {"unknown": 1}}],
                    "next_cursor": "ignored",
                }
            elif path == f"v1/data/results/{job_id}/embeddings":
                assert kwargs == {}
                response.json.return_value = {
                    "values": [
                        {
                            "position": position,
                            "unit_id": f"u{position}",
                            "value": {
                                "ref": {
                                    "uri": f"s3://content/{position}",
                                    "byte_length": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                    "codec": "ndarray-v1",
                                }
                            },
                        }
                        for position, payload in enumerate(payloads)
                    ],
                    "next_cursor": None,
                }
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler

        results = list(future.stream())

        np.testing.assert_array_equal(results[0], [1.0])
        np.testing.assert_array_equal(results[1], [2.0])
        assert _model_listing_calls(mock_session) == []

    def test_an_output_the_method_lacks_is_the_servers_404(self, future, mock_session):
        mock_session.get.side_effect = _not_found()

        with pytest.raises(HTTPError):
            list(future.stream(output="not-a-real-output"))

        mock_session.get.assert_called_once_with(
            f"v1/data/results/{JOB['job_id']}/not-a-real-output"
        )

    def test_malformed_canonical_value_raises(self, future, mock_session):
        job_id = JOB["job_id"]

        def handler(path, **kwargs):
            response = MagicMock(status_code=200)
            if path == f"v1/data/results/{job_id}":
                response.json.return_value = {
                    "values": [],
                    "next_cursor": None,
                    "output": "embeddings",
                }
            elif path == f"v1/data/results/{job_id}/embeddings":
                response.json.return_value = {
                    "values": [
                        {"position": 0, "unit_id": "u0", "value": {"unknown": 1}}
                    ],
                    "next_cursor": None,
                }
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler

        with pytest.raises(ValueError, match="inline or ref"):
            list(future.stream())


class TestGet:
    def test_get_resolves_the_default_then_pages_its_named_route(
        self, future, mock_session
    ):
        job_id = JOB["job_id"]
        page = {
            "values": [
                {"position": 0, "unit_id": "u0", "value": {"inline": 1.0}},
                {"position": 1, "unit_id": "u1", "value": {"inline": 2.0}},
            ],
            "next_cursor": None,
        }

        def handler(path, **kwargs):
            response = MagicMock(status_code=200)
            if path == f"v1/data/results/{job_id}":
                assert kwargs == {"params": {"limit": 1}}
                response.json.return_value = {"output": "embeddings", "values": []}
            elif path == f"v1/data/results/{job_id}/embeddings":
                assert kwargs == {}
                response.json.return_value = page
            elif path == f"v1/data/results/{job_id}/annotations":
                response.json.return_value = page
            else:
                raise AssertionError(f"unexpected GET {path}")
            return response

        mock_session.get.side_effect = handler

        assert future.get(verbose=False) == [1.0, 2.0]
        assert future.get(output="annotations") == [1.0, 2.0]
        assert [c.args[0] for c in mock_session.get.call_args_list] == [
            f"v1/data/results/{job_id}",
            f"v1/data/results/{job_id}/embeddings",
            f"v1/data/results/{job_id}/annotations",
        ]


def test_stream_reads_a_named_output_without_resolving_the_default(
    future, mock_session
):
    job_id = JOB["job_id"]

    def handler(path, **kwargs):
        assert path == f"v1/data/results/{job_id}/embeddings"
        response = MagicMock(status_code=200)
        response.json.return_value = {
            "values": [{"position": 0, "unit_id": "u0", "value": {"inline": [1.0]}}],
            "next_cursor": None,
        }
        return response

    mock_session.get.side_effect = handler

    assert list(future.stream("embeddings")) == [[1.0]]


@pytest.mark.parametrize("argument", ["limit", "headers"])
def test_stream_takes_only_its_own_arguments(future, argument):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        future.stream(**{argument: 1})


@pytest.mark.parametrize("argument", ["limit", "headers"])
def test_get_item_takes_only_its_own_arguments(future, argument):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        future.get_item(0, **{argument: 1})
