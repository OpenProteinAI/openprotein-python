from datetime import datetime
import pytest
from unittest.mock import MagicMock, patch

from openprotein.base import APISession
from openprotein.jobs import Job, JobStatus, JobType
from openprotein.align.msa import MSAFuture
from openprotein.align.schemas import MafftJob, MSASamplingMethod
from openprotein.prompt import Prompt


@pytest.fixture
def mock_session() -> MagicMock:
    """Fixture for a mocked APISession."""
    return MagicMock(spec=APISession)


@pytest.fixture
def mafft_job() -> MafftJob:
    """Fixture for a sample MafftJob object."""
    return MafftJob(
        job_id="job123",
        status=JobStatus.SUCCESS,
        job_type=JobType.mafft,
        created_date=datetime.fromisoformat("2023-01-01T00:00:00"),
    )


@pytest.fixture
def msa_future(mock_session: MagicMock, mafft_job: MafftJob) -> MSAFuture:
    """Fixture for an MSAFuture instance."""
    return MSAFuture(mock_session, mafft_job)


@patch("openprotein.align.api.get_msa")
def test_msa_future_get(mock_get_msa, msa_future: MSAFuture):
    """Test the get method of MSAFuture."""
    msa_future.get()
    mock_get_msa.assert_called_once_with(
        session=msa_future.session, job_id=msa_future.job.job_id
    )


@patch("openprotein.align.api.prompt_post")
@patch("openprotein.prompt.Prompt.create")
def test_msa_future_sample_prompt(
    mock_prompt_create, mock_prompt_post, msa_future: MSAFuture
):
    """Test the sample_prompt method of MSAFuture."""
    mock_job = MagicMock(spec=Job)
    mock_prompt_post.return_value = mock_job

    mock_prompt_future = MagicMock(spec=Prompt)
    mock_prompt_create.return_value = mock_prompt_future

    prompt = msa_future.sample_prompt(num_sequences=10, method=MSASamplingMethod.RANDOM)

    mock_prompt_post.assert_called_once_with(
        msa_future.session,
        msa_id=msa_future.msa_id,
        num_sequences=10,
        num_residues=None,
        method=MSASamplingMethod.RANDOM,
        homology_level=0.8,
        max_similarity=1.0,
        min_similarity=0.0,
        always_include_seed_sequence=False,
        num_ensemble_prompts=1,
        random_seed=None,
    )

    mock_prompt_create.assert_called_once_with(
        session=msa_future.session, job=mock_job, num_replicates=1
    )

    assert prompt == mock_prompt_future
