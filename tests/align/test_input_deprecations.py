"""The legacy align input reads are deprecated and keep working."""

import time
import warnings
from unittest.mock import MagicMock, patch

import pytest
import requests

from openprotein import auth_store
from openprotein.align import api
from openprotein.align.future import AlignFuture
from openprotein.align.schemas import AlignType
from openprotein.base import APISession
from openprotein.jobs import Job
from tests.conf import BACKEND

MSA_TEXT = (
    "`MSAFuture.wait()` for the generated MSA, or "
    "`session.data.MSA(id=...).to_string()`"
)
PROMPT_TEXT = (
    "`Prompt.get()` for the replicates of a sampled prompt, indexed by replicate"
)
API_SEED_TEXT = (
    "`openprotein.align.api.get_seed` for an MSA job's seed; the input of other "
    "align jobs has no read of its own, so keep your own copy of the submitted "
    "sequences"
)
FUTURE_SEED_TEXT = "`MSAFuture.get_seed()` for the seed"
INPUT_TYPES = (AlignType.MSA, AlignType.PROMPT, AlignType.INPUT)
SERVER_WARNING = "GET /align/inputs is deprecated"
JOB = Job.model_validate(
    {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": "/align/align",
        "created_date": "2023-01-01T00:00:00",
    }
)


@pytest.fixture
def session(tmp_path, monkeypatch):
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
        session = APISession("user@x.com", "pw", backend=BACKEND)
    session._expires_at = time.time() + 10_000
    return session


def _inputs_response() -> MagicMock:
    response = MagicMock(
        status_code=200,
        ok=True,
        headers={
            "Content-Type": "text/x-fasta",
            "Warning": f'299 - "{SERVER_WARNING}"',
        },
    )
    response.iter_lines.return_value = iter([">a", "MKT", ">b", "MKV"])
    return response


def _future(session) -> AlignFuture:
    future = AlignFuture()
    future.session = session
    future.job = JOB
    return future


CALLS = {
    "api.get_align_job_inputs": (
        "openprotein.align.api.get_align_job_inputs",
        lambda s, t: api.get_align_job_inputs(s, "job123", t),
    ),
    "api.get_input": (
        "openprotein.align.api.get_input",
        lambda s, t: list(api.get_input(s, "job123", t)),
    ),
    "AlignFuture.get_input": (
        "openprotein.align.future.AlignFuture.get_input",
        lambda s, t: list(_future(s).get_input(t)),
    ),
}


@pytest.mark.parametrize("input_type", INPUT_TYPES, ids=lambda t: t.name)
@pytest.mark.parametrize("name", CALLS)
def test_a_legacy_input_read_warns_once_with_the_replacement_for_its_input_type(
    session, name, input_type
):
    """Each legacy input read emits one versioned warning naming its mode's replacement."""
    qualified, call = CALLS[name]
    replacement = {
        AlignType.MSA: MSA_TEXT,
        AlignType.PROMPT: PROMPT_TEXT,
        AlignType.INPUT: (
            FUTURE_SEED_TEXT if name == "AlignFuture.get_input" else API_SEED_TEXT
        ),
    }[input_type]
    with (
        patch.object(requests.Session, "request", return_value=_inputs_response()),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        call(session, input_type)
    messages = [str(w.message) for w in caught if w.category is FutureWarning]
    assert messages == [
        (
            f"`{qualified}` is deprecated since 0.18.0 and will be removed in "
            f"0.19.0. Use {replacement}."
        )
    ]


def test_the_legacy_input_reads_keep_their_return_values(session):
    """The deprecated reads still return the response and the parsed pairs."""
    with (
        patch.object(requests.Session, "request", return_value=_inputs_response()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore", FutureWarning)
        pairs = list(api.get_input(session, "job123", AlignType.MSA))
        response = api.get_align_job_inputs(session, "job123", AlignType.MSA)
    assert pairs == [("a", "MKT"), ("b", "MKV")]
    assert response.ok


def test_get_seed_is_not_deprecated(session):
    """`get_seed` emits no deprecation warning and surfaces server warnings."""
    response = MagicMock(
        status_code=200, ok=True, headers={"Warning": f'299 - "{SERVER_WARNING}"'}
    )
    response.json.return_value = {"sequence": "MKT"}
    with (
        patch.object(requests.Session, "request", return_value=response),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        api.get_seed(session, "job123")
    assert [str(w.message) for w in caught] == [SERVER_WARNING]
