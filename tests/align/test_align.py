"""L2 integration tests for the align domain."""
import io
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from openprotein.align.align import AlignAPI
from openprotein.align.msa import MSAFuture
from openprotein.jobs import JobStatus, JobType


@pytest.fixture
def align_api(mock_session: MagicMock) -> AlignAPI:
    """Fixture for an AlignAPI instance."""
    return AlignAPI(mock_session)


def test_align_api_mafft(align_api: AlignAPI, mock_session: MagicMock):
    """Test the mafft method."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.mafft.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.mafft(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)


def test_align_api_clustalo(align_api: AlignAPI, mock_session: MagicMock):
    """Test the clustalo method."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.clustalo.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.clustalo(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)


def test_align_api_abnumber(align_api: AlignAPI, mock_session: MagicMock):
    """Test the abnumber method."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job123",
        "status": "SUCCESS",
        "job_type": JobType.abnumber.value,
        "created_date": "2023-01-01T00:00:00",
    }
    sequences = ["ACGT"]
    names = ["seq1"]
    future = align_api.abnumber(sequences, names)
    mock_session.post.assert_called_once()
    assert isinstance(future, MSAFuture)
