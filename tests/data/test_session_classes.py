"""Every handle a session hands out is an instance of that session's ``session.data`` class."""

from unittest.mock import MagicMock

import pytest

from openprotein.data import DataAPI
from openprotein.jobs import Future, Job

MSA_METADATA = {
    "msa_id": "m1",
    "created_at": "2026-07-20T00:00:00Z",
    "updated_at": "2026-07-20T00:00:00Z",
}
SAMPLED = {
    "id": "s1",
    "job_id": "s1",
    "name": "sampled",
    "status": "PENDING",
    "created_date": "2026-09-05T08:00:00Z",
}


@pytest.fixture
def session() -> MagicMock:
    session = MagicMock()
    session.data = DataAPI(session)
    return session


def _answer(session: MagicMock, verb: str, payload: dict) -> None:
    response = getattr(session, verb).return_value
    response.status_code = 200
    response.json.return_value = payload


def test_a_sampled_prompt_is_a_session_data_prompt(session):
    _answer(session, "get", MSA_METADATA)
    msa = session.data.MSA(id="m1")
    _answer(session, "post", SAMPLED)
    prompt = msa.sample_prompt()
    assert type(prompt) is session.data.Prompt


def test_a_reloaded_sampling_job_is_a_session_data_prompt(session):
    _answer(session, "get", {"id": "s1", "status": "SUCCESS", "job_id": "s1"})
    job = Job.create(
        {
            "job_id": "s1",
            "job_type": "/prompt/sample_msa",
            "status": "SUCCESS",
            "created_date": "2026-09-05T08:00:00Z",
        }
    )
    assert type(Future.create(session, job=job)) is session.data.Prompt


def test_an_assay_given_by_id_to_a_model_is_a_session_data_assay(session):
    from openprotein.models import _inputs

    _answer(
        session,
        "get",
        {
            "assay_name": "n",
            "assay_description": "d",
            "assay_id": "a1",
            "original_filename": "f.csv",
            "created_date": "2023-01-01T00:00:00",
            "num_rows": 1,
            "num_entries": 1,
            "measurement_names": ["y"],
        },
    )
    assert type(_inputs.as_dataset(session, "a1")) is session.data.Assay
