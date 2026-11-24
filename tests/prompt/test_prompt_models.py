from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.errors import InvalidParameterError
from openprotein.jobs import Future, Job, JobsAPI, JobStatus, JobType
from openprotein.molecules import Complex, Protein
from openprotein.prompt import Prompt, PromptJob, PromptMetadata, Query, QueryMetadata


@pytest.fixture
def mock_session(prompt_job):
    session = MagicMock()
    session.jobs = MagicMock(spec=JobsAPI)
    session.jobs.get_job.return_value = prompt_job
    return session


@pytest.fixture
def prompt_metadata():
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
    return PromptJob(
        job_id="job-123",
        job_type=JobType.align_prompt,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
    )


@pytest.fixture
def query_metadata():
    return QueryMetadata(
        id="query-123",
        created_date=datetime.now(),
    )


def test_prompt_initialization_with_metadata(mock_session, prompt_metadata, prompt_job):
    mock_session.jobs.get_job.return_value = prompt_job
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    assert prompt.id == "prompt-123"
    assert prompt.name == "Test Prompt"
    assert prompt.job is not None


def test_prompt_initialization_with_job(mock_session, prompt_job, prompt_metadata):
    with patch(
        "openprotein.prompt.api.get_prompt_metadata", return_value=prompt_metadata
    ):
        prompt = Prompt(session=mock_session, job=prompt_job)
        assert prompt.id == prompt_metadata.id
        assert prompt.metadata is not None


@patch("openprotein.prompt.api.get_prompt")
def test_prompt_get(mock_get_prompt, mock_session, prompt_metadata):
    mock_get_prompt.return_value = "Prompt Content"
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    content = prompt.get()
    mock_get_prompt.assert_called_with(session=mock_session, prompt_id="prompt-123")
    assert content == "Prompt Content"


def test_an_align_prompt_job_loads_as_a_prompt(mock_session, prompt_metadata):
    payload = {
        "job_id": "job-123",
        "job_type": "/align/prompt",
        "status": "SUCCESS",
        "created_date": datetime.now().isoformat(),
    }
    mock_session.get.return_value.json.return_value = payload
    mock_session.jobs.get_job.return_value = Job.create(payload)

    with patch(
        "openprotein.prompt.api.get_prompt_metadata", return_value=prompt_metadata
    ):
        future = Future.create(session=mock_session, job_id="job-123")

    assert isinstance(future, Prompt)
    assert isinstance(future.job, PromptJob)


@pytest.mark.parametrize("update", ["refresh", "wait_until_done"])
def test_the_metadata_reports_the_status_of_the_refreshed_job(
    mock_session, prompt_metadata, prompt_job, update
):
    mock_session.jobs.get_job.return_value = prompt_job.model_copy(
        update={"status": JobStatus.PENDING}
    )
    prompt = Prompt(
        session=mock_session,
        metadata=prompt_metadata.model_copy(update={"status": JobStatus.PENDING}),
    )
    mock_session.get.return_value.json.return_value = prompt_job.model_dump(mode="json")

    getattr(prompt, update)()

    assert prompt.status == JobStatus.SUCCESS
    assert prompt.metadata.status == JobStatus.SUCCESS
    assert repr(prompt).startswith("Prompt(id='prompt-123'")
    assert "SUCCESS" in repr(prompt)


def test_a_prompt_refresh_passes_its_timeout_to_the_poll(
    mock_session, prompt_metadata, prompt_job
):
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    mock_session.get.return_value.json.return_value = prompt_job.model_dump(mode="json")

    prompt._refresh_job(timeout=2)

    assert mock_session.get.call_args.kwargs == {"timeout": 2}


def test_query_initialization(mock_session, query_metadata):
    query = Query(session=mock_session, metadata=query_metadata)
    assert query.id == "query-123"


@patch("openprotein.prompt.api.get_query")
def test_query_get(mock_get_query, mock_session, query_metadata):
    mock_get_query.return_value = "Query Content"
    query = Query(session=mock_session, metadata=query_metadata)
    content = query.get()
    mock_get_query.assert_called_with(session=mock_session, query_id="query-123")
    assert content == "Query Content"


@patch("openprotein.prompt.api.get_prompt")
def test_prompt_get_as_complexes_wraps_proteins(
    mock_get_prompt, mock_session, prompt_metadata
):
    """Single-chain Protein entries are wrapped as Complex."""
    p = Protein(name="p", sequence=b"ACDE")
    multi = Complex({"A": Protein(sequence=b"ACDE"), "B": Protein(sequence=b"GHIK")})
    mock_get_prompt.return_value = [[p, multi]]
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    result = prompt.get_as_complexes()
    assert len(result) == 1 and len(result[0]) == 2
    assert all(isinstance(e, Complex) for e in result[0])
    assert result[0][1] is multi  # passthrough for already-Complex


@patch("openprotein.prompt.api.get_prompt")
def test_prompt_get_as_proteins_raises_on_multichain(
    mock_get_prompt, mock_session, prompt_metadata
):
    """get_as_proteins raises if any entry is multichain."""
    multi = Complex({"A": Protein(sequence=b"ACDE"), "B": Protein(sequence=b"GHIK")})
    mock_get_prompt.return_value = [[Protein(name="p", sequence=b"ACDE"), multi]]
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    with pytest.raises(InvalidParameterError, match="multichain"):
        prompt.get_as_proteins()


@patch("openprotein.prompt.api.get_prompt")
def test_prompt_get_as_proteins_passthrough_when_all_single(
    mock_get_prompt, mock_session, prompt_metadata
):
    p = Protein(name="p", sequence=b"ACDE")
    mock_get_prompt.return_value = [[p]]
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)
    result = prompt.get_as_proteins()
    assert result == [[p]]


@patch("openprotein.prompt.api._get_query_as_complex")
def test_query_get_as_complex_wraps_protein(
    mock_get_query_as_complex, mock_session, query_metadata
):
    """Delegates to api._get_query_as_complex."""
    c = Complex({"A": Protein(name="q", sequence=b"ACDE")})
    mock_get_query_as_complex.return_value = c
    query = Query(session=mock_session, metadata=query_metadata)
    result = query.get_as_complex()
    assert result is c
    mock_get_query_as_complex.assert_called_with(
        session=mock_session, query_id="query-123"
    )


@patch("openprotein.prompt.api._get_query_as_complex")
def test_query_get_as_complex_passthrough(
    mock_get_query_as_complex, mock_session, query_metadata
):
    c = Complex({"A": Protein(sequence=b"ACDE"), "B": Protein(sequence=b"GHIK")})
    mock_get_query_as_complex.return_value = c
    query = Query(session=mock_session, metadata=query_metadata)
    assert query.get_as_complex() is c


@patch("openprotein.prompt.api.get_query")
def test_query_get_as_protein_raises_on_multichain(
    mock_get_query, mock_session, query_metadata
):
    c = Complex({"A": Protein(sequence=b"ACDE"), "B": Protein(sequence=b"GHIK")})
    mock_get_query.return_value = c
    query = Query(session=mock_session, metadata=query_metadata)
    with pytest.raises(InvalidParameterError, match="multichain"):
        query.get_as_protein()


@patch("openprotein.prompt.api.get_query")
def test_query_get_as_protein_passthrough(mock_get_query, mock_session, query_metadata):
    p = Protein(name="q", sequence=b"ACDE")
    mock_get_query.return_value = p
    query = Query(session=mock_session, metadata=query_metadata)
    assert query.get_as_protein() is p


@pytest.fixture
def uploaded_prompt(mock_session, prompt_metadata):
    return Prompt(
        session=mock_session,
        metadata=prompt_metadata.model_copy(update={"job_id": None}),
    )


def test_a_prompt_does_not_rename(mock_session, prompt_metadata):
    prompt = Prompt(session=mock_session, metadata=prompt_metadata)

    with pytest.raises(NotImplementedError, match="set when it is created"):
        prompt.rename(name="other")
    mock_session.put.assert_not_called()


def test_an_uploaded_prompt_does_not_rename(mock_session, uploaded_prompt):
    with pytest.raises(NotImplementedError, match="set when it is created"):
        uploaded_prompt.rename(name="other")
    mock_session.put.assert_not_called()
