"""Sampling a stored MSA into a prompt, and the prompt it returns."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from openprotein.align.schemas import MSASamplingMethod
from openprotein.data.msa import MSA, MSAMetadata
from openprotein.data.prompts import Prompt, PromptMetadata
from openprotein.errors import HTTPError, MissingParameterError
from openprotein.jobs import JobStatus

MSA_ID = "e0b1c3d4-0000-4000-8000-000000000001"

PROMPT_ID = "e0b1c3d4-0000-4000-8000-000000000002"

JOB: dict[str, Any] = {
    "job_id": PROMPT_ID,
    "job_type": "/prompt/sample_msa",
    "status": "PENDING",
    "created_date": "2026-09-05T08:00:00Z",
}

URL = "v1/data/prompts/sample_msa"


def _respond(mock: MagicMock, payload: object, status_code: int = 200) -> None:
    mock.return_value.status_code = status_code
    mock.return_value.json.return_value = payload


def _mock_submission(mock_session, payload: dict[str, Any] = JOB) -> None:
    _respond(mock_session.post, payload, status_code=202)


def test_sample_msa_posts_one_replicate_and_a_seed_when_options_are_omitted(
    mock_session,
):
    _mock_submission(mock_session)

    _stored_msa(mock_session).sample_prompt()

    call = mock_session.post.call_args
    assert call.args == (URL,)
    body = call.kwargs["json"]
    assert body["msa_id"] == MSA_ID
    assert isinstance(body["seed"], int)
    assert body["replicates"] == 1
    assert set(body) == {"msa_id", "seed", "replicates"}
    assert "headers" not in call.kwargs
    assert "params" not in call.kwargs


def test_sample_msa_draws_a_different_seed_each_call(mock_session):
    _mock_submission(mock_session)

    msa = _stored_msa(mock_session)
    msa.sample_prompt()
    first_seed = mock_session.post.call_args.kwargs["json"]["seed"]
    msa.sample_prompt()
    second_seed = mock_session.post.call_args.kwargs["json"]["seed"]

    assert first_seed != second_seed


def test_sample_msa_posts_the_given_options(mock_session):
    _mock_submission(mock_session)

    _stored_msa(mock_session).sample_prompt(
        replicates=3,
        seed=11,
        method=MSASamplingMethod.TOP,
        homology_level=0.5,
        max_similarity=0.9,
        min_similarity=0.1,
        max_msa_sequences=32,
        force_include_first=True,
    )

    call = mock_session.post.call_args
    assert call.args == (URL,)
    assert call.kwargs["json"] == {
        "msa_id": MSA_ID,
        "seed": 11,
        "replicates": 3,
        "method": "TOP",
        "homology_level": 0.5,
        "max_similarity": 0.9,
        "min_similarity": 0.1,
        "max_msa_sequences": 32,
        "force_include_first": True,
    }


def test_sample_msa_accepts_a_token_budget_alone(mock_session):
    _mock_submission(mock_session)

    _stored_msa(mock_session).sample_prompt(max_msa_tokens=4096)

    body = mock_session.post.call_args.kwargs["json"]
    assert body["max_msa_tokens"] == 4096
    assert "max_msa_sequences" not in body


@pytest.mark.parametrize("job_type", ["/prompt/sample_msa", "/align/prompt"])
@pytest.mark.parametrize("status", ["PENDING", "RUNNING", "SUCCESS", "FAILURE"])
def test_sample_msa_returns_the_prompt_its_job_samples(mock_session, status, job_type):
    _mock_submission(mock_session, JOB | {"status": status, "job_type": job_type})

    prompt = _stored_msa(mock_session).sample_prompt()

    assert isinstance(prompt, Prompt)
    assert (prompt.id, prompt.job_id) == (PROMPT_ID, PROMPT_ID)
    assert prompt.status.value == status
    assert prompt.created_at is not None and prompt.created_at.year == 2026
    mock_session.get.assert_not_called()


def _stored_msa(session) -> MSA:
    return MSA(
        session,
        MSAMetadata(
            msa_id=MSA_ID,
            source_format="a3m",
            num_sequences=2,
            num_match_columns=5,
            has_insertions=False,
            created_at="2026-09-05T08:00:00Z",
            updated_at="2026-09-05T08:00:00Z",
        ),
    )


def test_an_msa_samples_itself_into_a_prompt(mock_session):
    _mock_submission(mock_session)

    prompt = _stored_msa(mock_session).sample_prompt(
        replicates=2, seed=7, max_msa_tokens=4096, force_include_first=True
    )

    assert isinstance(prompt, Prompt) and prompt.id == PROMPT_ID
    call = mock_session.post.call_args
    assert call.args == (URL,)
    assert call.kwargs["json"] == {
        "msa_id": MSA_ID,
        "seed": 7,
        "replicates": 2,
        "max_msa_tokens": 4096,
        "force_include_first": True,
    }


def test_an_msa_sampling_takes_one_budget_or_the_other(mock_session):
    with pytest.raises(MissingParameterError, match="not both"):
        _stored_msa(mock_session).sample_prompt(
            max_msa_sequences=32, max_msa_tokens=4096
        )
    mock_session.post.assert_not_called()


def _response(status_code: int, content: bytes = b"", payload: dict | None = None):
    response = MagicMock()
    response.status_code = status_code
    response.content = content
    response.json.return_value = payload or {}
    return response


def _route(mock_session, routes: dict[str, MagicMock]) -> None:
    """Answer each GET by its path; an unlisted path is a 404."""

    def get(path, **kwargs):
        response = routes.get(path, _response(404, payload={"detail": "not found"}))
        if response.status_code >= 400:
            raise HTTPError(response)
        return response

    mock_session.get.side_effect = get


def _sampled(mock_session, status: str = "SUCCESS") -> Prompt:
    return Prompt(
        mock_session,
        PromptMetadata(id=PROMPT_ID, status=JobStatus(status), job_id=PROMPT_ID),
    )


@pytest.mark.parametrize("job_type", ["/prompt/sample_msa", "/align/prompt"])
def test_a_sampled_prompt_waits_on_its_job_then_reads_its_row(
    mock_session, monkeypatch, job_type
):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    stored = {"id": PROMPT_ID, "job_id": PROMPT_ID, "status": "SUCCESS", "name": "s"}
    _route(
        mock_session,
        {
            f"v1/jobs/{PROMPT_ID}": _response(
                200, payload=JOB | {"job_type": job_type, "status": "SUCCESS"}
            ),
            f"v1/data/prompts/{PROMPT_ID}": _response(200, payload=stored),
        },
    )
    prompt = _sampled(mock_session, "PENDING")

    assert prompt.wait() is prompt
    assert (prompt.status, prompt.name) == (JobStatus.SUCCESS, "s")
    assert not any(
        call.args[0].startswith("v1/prompt/")
        for call in mock_session.get.call_args_list
    )


def test_a_sampled_prompt_reads_only_the_data_routes(mock_session):
    _route(mock_session, {})

    with pytest.raises(HTTPError) as excinfo:
        list(_sampled(mock_session))

    assert excinfo.value.status_code == 404
    assert [call.args[0] for call in mock_session.get.call_args_list] == [
        f"v1/data/prompts/{PROMPT_ID}/download"
    ]


def test_a_models_job_still_resolves_to_the_generic_future():
    """Futures without a domain job schema declare no job annotation."""
    from openprotein.jobs.futures import EmbeddingFuture, SVDFitFuture

    for domain in (SVDFitFuture, EmbeddingFuture):
        assert domain.__annotations__.get("job") is None
