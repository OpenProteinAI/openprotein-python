"""L2 integration tests for the prompt domain."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from openprotein.jobs import JobStatus, JobType, JobsAPI
from openprotein.prompt.models import Prompt, Query
from openprotein.prompt.prompt import PromptAPI
from openprotein.prompt.schemas import PromptJob


@pytest.fixture
def prompt_api(mock_session: MagicMock):
    return PromptAPI(mock_session)


@pytest.fixture
def mock_prompt_job():
    return PromptJob(
        job_id="job-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.align_prompt,
        created_date=datetime.now(),
    )


def test_create_prompt(prompt_api: PromptAPI, mock_session: MagicMock, mock_prompt_job):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": "2023-01-01T00:00:00",
        "num_replicates": 1,
        "status": "SUCCESS",
        "job_id": "job-123",
        "job_type": JobType.align_prompt.value,
    }
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_prompt_job

    prompt = prompt_api.create_prompt(["ACGT"], name="Test Prompt")
    mock_session.post.assert_called_once()
    assert isinstance(prompt, Prompt)
    assert prompt.id == "prompt-123"


def test_get_prompt(prompt_api: PromptAPI, mock_session: MagicMock, mock_prompt_job):
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": "2023-01-01T00:00:00",
        "num_replicates": 1,
        "status": "SUCCESS",
        "job_id": "job-123",
        "job_type": JobType.align_prompt.value,
    }
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_prompt_job

    prompt = prompt_api.get_prompt("prompt-123")
    mock_session.get.assert_called_once_with("v1/prompt/prompt-123")
    assert isinstance(prompt, Prompt)


def test_create_query(prompt_api: PromptAPI, mock_session: MagicMock):
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "query-123",
        "created_date": "2023-01-01T00:00:00",
    }

    query = prompt_api.create_query("ACGT")
    mock_session.post.assert_called_once()
    assert isinstance(query, Query)
    assert query.id == "query-123"


class TestResolveQueryIds:
    """Text in the hyphenated id form names a query, in any case and as str or bytes."""

    UPPER = "C9BF9E57-1685-4C89-BAFB-FF5AF830BE8A"
    LOWER = UPPER.lower()

    @pytest.mark.parametrize("value", [UPPER, LOWER, UPPER.encode()])
    def test_an_id_is_sent_lower_cased_without_an_upload(self, mock_session, value):
        api = PromptAPI(mock_session)
        api.create_query = MagicMock()
        assert api._resolve_query(query=value) == self.LOWER
        api.create_query.assert_not_called()

    def test_a_sequence_is_uploaded(self, mock_session):
        api = PromptAPI(mock_session)
        api.create_query = MagicMock()
        api.create_query.return_value.id = "q-1"
        assert api._resolve_query(query="ACDE") == "q-1"
