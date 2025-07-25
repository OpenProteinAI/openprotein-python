from openprotein.jobs import JobsAPI
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.jobs import JobStatus, JobType
from openprotein.prompt import Prompt, PromptJob, PromptMetadata, Query, QueryMetadata


@pytest.fixture
def mock_session(prompt_job):
    """Fixture to create a mock APISession."""
    session = MagicMock()
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = prompt_job
    return session


@pytest.fixture
def prompt_metadata():
    """Fixture for PromptMetadata."""
    return PromptMetadata(
        id="prompt-123",
        name="Test Prompt",
        description="A test prompt.",
        created_date=datetime.now(),
        num_replicates=1,
        job_id="job-123",
        status=JobStatus.SUCCESS,
    )


@pytest.fixture
def prompt_job():
    """Fixture for PromptJob."""
    return PromptJob(
        job_id="job-123",
        job_type=JobType.align_prompt,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
    )


@pytest.fixture
def query_metadata():
    """Fixture for QueryMetadata."""
    return QueryMetadata(
        id="query-123",
        created_date=datetime.now(),
    )


def test_prompt_initialization_with_metadata(mock_session, prompt_metadata, prompt_job):
    """Test Prompt initialization using PromptMetadata."""
    mock_session.jobs.get_job.return_value = prompt_job
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    assert prompt.id == "prompt-123"
    assert prompt.name == "Test Prompt"
    assert prompt.job is not None


def test_prompt_initialization_with_job(mock_session, prompt_job, prompt_metadata):
    """Test Prompt initialization using a PromptJob."""
    with patch(
        "openprotein.prompt.api.get_prompt_metadata", return_value=prompt_metadata
    ):
        prompt = Prompt(session=mock_session, job=prompt_job)
        assert prompt.id == prompt_metadata.id
        assert prompt.metadata is not None


@patch("openprotein.prompt.api.get_prompt")
def test_prompt_get(mock_get_prompt, mock_session, prompt_metadata):
    """Test the get method of the Prompt class."""
    mock_get_prompt.return_value = "Prompt Content"
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    content = prompt.get()
    mock_get_prompt.assert_called_with(session=mock_session, prompt_id="prompt-123")
    assert content == "Prompt Content"


def test_query_initialization(mock_session, query_metadata):
    """Test Query initialization."""
    query = Query(session=mock_session, metadata=query_metadata)
    assert query.id == "query-123"


@patch("openprotein.prompt.api.get_query")
def test_query_get(mock_get_query, mock_session, query_metadata):
    """Test the get method of the Query class."""
    mock_get_query.return_value = "Query Content"
    query = Query(session=mock_session, metadata=query_metadata)
    content = query.get()
    mock_get_query.assert_called_with(session=mock_session, query_id="query-123")
    assert content == "Query Content"
