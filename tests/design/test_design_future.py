"""Test the Future class for the design domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.design.future import DesignFuture
from openprotein.design.schemas import Criteria, Design, DesignAlgorithm, DesignJob
from openprotein.jobs import JobsAPI, JobStatus, JobType


@pytest.fixture
def mock_design_metadata():
    """Fixture for a valid Design instance."""
    return Design(
        id="design-123",
        status=JobStatus.SUCCESS,
        progress_counter=100,
        created_date=datetime.now(),
        algorithm=DesignAlgorithm.genetic_algorithm,
        num_rows=10,
        num_steps=10,
        assay_id="a1",
        criteria=Criteria(root=[]),  # Correctly instantiate Criteria
        allowed_tokens={},
        pop_size=128,
        n_offsprings=256,
        crossover_prob=0.8,
        crossover_prob_pointwise=0.1,
        mutation_average_mutations_per_seq=2,
    )


@pytest.fixture
def mock_design_job():
    """Fixture for a valid DesignJob instance."""
    return DesignJob(
        job_id="design-123",
        status=JobStatus.SUCCESS,
        job_type=JobType.designer,
        created_date=datetime.now(),
    )


def test_design_future_init(mock_session, mock_design_metadata, mock_design_job):
    """Test DesignFuture initialization with metadata."""
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_design_job
    future = DesignFuture(session=mock_session, metadata=mock_design_metadata)
    assert future.id == "design-123"
    mock_session.jobs.get_job.assert_called_once()


def test_design_future_stream(mock_session, mock_design_metadata, mock_design_job):
    """Test the stream method of DesignFuture."""
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_design_job
    future = DesignFuture(session=mock_session, metadata=mock_design_metadata)

    with (
        patch("openprotein.design.api.designer_get_design_results") as mock_get_results,
        patch("openprotein.design.api.decode_design_results_stream") as mock_decode,
    ):
        # The stream method returns a generator, so we need to consume it
        list(future.stream(step=1))
        mock_get_results.assert_called_once_with(
            session=mock_session, design_id=future.id, step=1
        )
        mock_decode.assert_called_once()
