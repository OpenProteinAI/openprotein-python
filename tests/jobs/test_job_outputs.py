import hashlib
import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch

import pytest
import requests

import openprotein.jobs as jobs_package
from openprotein.errors import APIError, JobFailedException, JobOutputsNotReady
from openprotein.jobs import Future, JobStatus
from openprotein.jobs import _api as jobs_wire
from openprotein.jobs._storage import DirectArtifactStorage, verified_artifact
from openprotein.jobs.jobs import JobsAPI, _JobOutput
from openprotein.jobs.schemas import _UnitRecord


@pytest.mark.parametrize(
    "name", ["get_job_output", "iter_job_outputs", "iter_job_units"]
)
def test_jobs_api_has_no_public_output_reads(name):
    """Job output reads are on the future; the jobs API has no public form of them."""
    assert not hasattr(JobsAPI, name)


def test_the_jobs_package_has_no_job_output_record():
    """The decoded output record is not part of the jobs package."""
    assert not hasattr(jobs_package, "JobOutput")
    assert "JobOutput" not in jobs_package.__all__


def test_a_future_has_no_iter_outputs():
    """A named output is read with `stream`."""
    assert not hasattr(Future, "iter_outputs")


def _jobs(session, storage):
    jobs = JobsAPI(session)
    jobs._storage = storage
    return jobs


def _page(entries, next_cursor=None):
    response = MagicMock(status_code=200)
    response.json.return_value = {"values": entries, "next_cursor": next_cursor}
    return response


def _entry(position, unit_id, value):
    return {"position": position, "unit_id": unit_id, "value": value}


def _reference(payload, codec=None):
    return {
        "uri": "s3://private/object",
        "byte_length": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "s3_checksum": "checksum",
        "codec": codec,
    }


class MemoryStorage:
    def __init__(self, payload):
        self.payload = payload
        self.reads = 0

    def iter_bytes(self, reference):
        self.reads += 1
        for start in range(0, len(self.payload), 7):
            yield self.payload[start : start + 7]


def test_output_page_forwards_cursor_and_limit():
    session = MagicMock()
    session.get.return_value = _page([])
    jobs_wire.job_output_page(session, "job", "embedding", cursor="opaque", limit=50)
    session.get.assert_called_once_with(
        "v1/data/results/job/embedding", params={"cursor": "opaque", "limit": 50}
    )


def test_inline_outputs_are_lazy_ordered_and_do_not_touch_storage():
    session = MagicMock()
    session.get.side_effect = [
        _page([_entry(0, "same", {"inline": 1})], "next"),
        _page([_entry(1, "same", {"inline": 2})]),
    ]
    storage = MemoryStorage(b"")
    iterator = _jobs(session, storage)._iter_job_outputs("job", "score")
    session.get.assert_not_called()
    assert list(iterator) == [_JobOutput(0, "same", 1), _JobOutput(1, "same", 2)]
    assert storage.reads == 0
    assert session.get.call_args_list[1].kwargs["params"] == {"cursor": "next"}


@pytest.mark.parametrize("inline", ["nan", {"mean": 1, "variance": -1}, [1, 2]])
def test_inline_values_come_back_as_stored(inline):
    session = MagicMock()
    session.get.return_value = _page([_entry(0, "unit", {"inline": inline})])
    (output,) = JobsAPI(session)._iter_job_outputs("job", "score")
    assert output.value == inline


def test_artifact_reads_directly_and_verifies_full_content():
    payload = b">query\nACGT\n"
    session = MagicMock()
    session.get.return_value = _page([_entry(3, "u3", {"ref": _reference(payload)})])
    storage = MemoryStorage(payload)
    (entry,) = _jobs(session, storage)._iter_job_outputs("job", "msa")
    assert entry.value == payload
    assert storage.reads == 1
    session.get.assert_called_once_with("v1/data/results/job/msa")


@pytest.mark.parametrize(
    "change,match",
    [
        ({"sha256": "0" * 64}, "digest"),
        ({"byte_length": 1}, "size"),
        ({"byte_length": 100}, "size"),
        ({"byte_length": True}, "byte_length"),
        ({"sha256": "x" * 64}, "sha256"),
    ],
)
def test_rejects_invalid_or_corrupt_reference(change, match):
    reference = _reference(b"actual content") | change
    with (
        pytest.raises(ValueError, match=match),
        verified_artifact(MemoryStorage(b"actual content"), reference),
    ):
        pytest.fail("invalid bytes became visible")


def test_incomplete_output_fails_without_polling():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=409)
    with pytest.raises(JobOutputsNotReady) as raised:
        list(JobsAPI(session)._iter_job_outputs("job", "embeddings"))
    assert isinstance(raised.value, APIError)
    assert str(raised.value) == "output 'embeddings' of job job is not complete yet"
    session.get.assert_called_once_with("v1/data/results/job/embeddings")


def _unit_record(position, state="SUCCESS"):
    return {
        "position": position,
        "unit_id": f"u{position}",
        "state": state,
        "retries": 0,
        "waits_on": ["upstream"] if position else [],
        "cached": position == 0,
    }


def test_iter_job_units_yields_unit_records_paginated_by_position():
    session = MagicMock()
    first, second = MagicMock(), MagicMock()
    first.json.return_value = {
        "units": [_unit_record(0), _unit_record(1)],
        "next_position": 5,
    }
    second.json.return_value = {
        "units": [_unit_record(5)],
        "next_position": None,
    }
    session.get.side_effect = [first, second]
    units = list(JobsAPI(session)._iter_job_units("job"))
    assert [unit.position for unit in units] == [0, 1, 5]
    assert all(isinstance(unit, _UnitRecord) for unit in units)
    assert units[0].status is JobStatus.SUCCESS
    assert not hasattr(units[0], "waits_on")
    session.get.assert_any_call("v1/jobs/job/units", params={"after_position": 5})


def test_storage_reads_the_ref_uri_and_closes_stream():
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(b"contents")}
    reference = _reference(b"contents")
    assert (
        b"".join(DirectArtifactStorage(client, chunk_size=3).iter_bytes(reference))
        == b"contents"
    )
    client.get_object.assert_called_once_with(Bucket="private", Key="object")
    assert client.get_object.return_value["Body"].closed


def test_signed_download_does_not_use_platform_session_or_headers():
    assert "Authorization" not in DirectArtifactStorage().http.headers
    http = MagicMock()
    response = http.get.return_value
    response.__enter__.return_value = response
    response.iter_content.return_value = iter([b"data"])
    storage = DirectArtifactStorage(http=http)
    reference = {"download_url": "https://storage.example/signed"}
    assert b"".join(storage.iter_bytes(reference)) == b"data"
    http.get.assert_called_once_with(
        "https://storage.example/signed", stream=True, timeout=180
    )


def test_ref_download_url_is_used_with_no_aws_client():
    payload = b">query\nACGT\n"
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = iter(
        [payload[start : start + 4] for start in range(0, len(payload), 4)]
    )
    reference = _reference(payload) | {"download_url": "https://storage.example/signed"}
    del reference["uri"]
    session = MagicMock()
    session.get.return_value = _page([_entry(0, "u0", {"ref": reference})])
    with patch.object(requests.Session, "get", return_value=response) as get:
        (entry,) = JobsAPI(session)._iter_job_outputs("job", "msa")
    assert entry.value == payload
    get.assert_called_once_with(
        "https://storage.example/signed", stream=True, timeout=180
    )


class SlowStorage:
    """Serves each reference's payload, slower for earlier positions."""

    def __init__(self, delays):
        self.delays = delays
        self.running = 0
        self.most = 0
        self.lock = threading.Lock()

    def iter_bytes(self, reference):
        with self.lock:
            self.running += 1
            self.most = max(self.most, self.running)
        time.sleep(self.delays[reference["uri"]])
        with self.lock:
            self.running -= 1
        yield reference["uri"].encode()


def test_ref_outputs_download_concurrently_and_arrive_in_position_order():
    count = 40
    uris = [f"s3://private/{position}" for position in range(count)]
    entries = [
        _entry(
            position, f"u{position}", {"ref": _reference(uri.encode()) | {"uri": uri}}
        )
        for position, uri in enumerate(uris)
    ]
    session = MagicMock()
    session.get.side_effect = [_page(entries[:25], "next"), _page(entries[25:])]
    storage = SlowStorage(
        {uri: 0.02 * (count - i) / count for i, uri in enumerate(uris)}
    )
    outputs = list(_jobs(session, storage)._iter_job_outputs("job", "embeddings"))
    assert [output.position for output in outputs] == list(range(count))
    assert [output.value for output in outputs] == [uri.encode() for uri in uris]
    assert storage.most > 1


def test_signed_download_retries_object_storage_5xx():
    requests_seen = []

    class Flaky(BaseHTTPRequestHandler):
        def do_GET(self):
            requests_seen.append(self.path)
            status, body = (500, b"") if len(requests_seen) == 1 else (200, b"data")
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/signed"
        data = b"".join(DirectArtifactStorage().iter_bytes({"download_url": url}))
    finally:
        server.shutdown()
    assert data == b"data"
    assert requests_seen == ["/signed", "/signed"]


def _emitted_page(entries, next_cursor, complete=False):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "values": entries,
        "next_cursor": next_cursor,
        "complete": complete,
    }
    return response


def test_an_emitted_output_read_ends_at_the_items_registered_now():
    session = MagicMock()
    session.get.side_effect = [
        _emitted_page(
            [_entry(0, "u", {"inline": 0}), _entry(1, "u", {"inline": 1})], "c1"
        ),
        _emitted_page([], "c1"),
    ]
    outputs = list(JobsAPI(session)._iter_job_outputs("job", "losses"))
    assert [output.value for output in outputs] == [0, 1]
    assert session.get.call_count == 2


def test_following_yields_each_item_before_waiting_until_complete():
    session = MagicMock()
    session.get.side_effect = [
        _emitted_page([_entry(0, "u", {"inline": 0})], "c1"),
        _emitted_page([], "c1"),
        _emitted_page([_entry(1, "u", {"inline": 1})], "c2"),
        _emitted_page([], None, complete=True),
    ]
    with patch("openprotein.jobs.jobs.time.sleep") as sleep:
        stream = JobsAPI(session)._iter_job_outputs(
            "job", "losses", follow=True, interval=7
        )
        assert next(stream).value == 0
        sleep.assert_not_called()
        assert [output.value for output in stream] == [1]
    sleep.assert_called_once_with(7)
    cursors = [
        call.kwargs.get("params", {}).get("cursor")
        for call in session.get.call_args_list
    ]
    assert cursors == [None, "c1", "c1", "c2"]


def test_a_failed_emitted_item_raises_with_its_error():
    session = MagicMock()
    session.get.return_value = _emitted_page(
        [_entry(0, "u", None) | {"error": "digest mismatch"}], None, complete=True
    )
    with pytest.raises(JobFailedException) as raised:
        list(JobsAPI(session)._iter_job_outputs("job", "embeddings"))
    assert raised.value.job_id == "job"
    assert (
        raised.value.failure_message == "output at position 0 failed: digest mismatch"
    )


@pytest.mark.parametrize("position", [-1, "0", True])
def test_an_invalid_output_position_is_a_value_error(position):
    with pytest.raises(ValueError, match="nonnegative integer"):
        JobsAPI(MagicMock())._decode_job_output(
            "job", {"position": position, "unit_id": "u", "value": {"inline": 1}}
        )


def test_a_final_output_page_without_successes_keeps_reading():
    session = MagicMock()
    session.get.side_effect = [
        _page([], "100"),
        _page([_entry(100, "u100", {"inline": 1})]),
    ]
    outputs = list(JobsAPI(session)._iter_job_outputs("job", "score"))
    assert [output.position for output in outputs] == [100]
