import io
from unittest.mock import MagicMock, patch

import pytest

from openprotein.align.api import (
    abnumber_post,
    clustalo_post,
    get_align_job_inputs,
    get_input,
    get_seed,
    mafft_post,
    msa_post,
    prompt_post,
)
from openprotein.align.schemas import AlignType, MSASamplingMethod
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError, MissingParameterError
from openprotein.jobs import Job, JobStatus, JobType


@pytest.fixture
def sample_job_dict() -> dict:
    """Fixture for a sample job dictionary response."""
    return {
        "job_id": "job123",
        "status": JobStatus.SUCCESS,
        "job_type": JobType.align_align,
        "created_date": "2023-01-01T00:00:00",
        "last_update": "2023-01-01T00:00:00",
    }


def test_get_align_job_inputs(mock_session: MagicMock):
    """Test get_align_job_inputs calls the correct endpoint and params."""
    mock_session.get.return_value = MagicMock()

    get_align_job_inputs(mock_session, "job1", AlignType.MSA)
    mock_session.get.assert_called_with(
        "v1/align/inputs",
        params={"job_id": "job1", "msa_type": "GENERATED"},
        stream=True,
    )

    get_align_job_inputs(mock_session, "job1", AlignType.PROMPT, prompt_index=1)
    mock_session.get.assert_called_with(
        "v1/align/inputs",
        params={"job_id": "job1", "msa_type": "PROMPT", "replicate": 1},
        stream=True,
    )


@patch("openprotein.align.api.get_align_job_inputs")
@patch("openprotein.csv.parse_stream")
def test_get_input(mock_csv_stream, mock_get_align_job_inputs, mock_session: MagicMock):
    """Test get_input correctly calls dependencies."""
    get_input(mock_session, "job1", AlignType.MSA)
    mock_get_align_job_inputs.assert_called_once_with(
        session=mock_session, job_id="job1", input_type=AlignType.MSA, prompt_index=None
    )
    mock_csv_stream.assert_called_once()


def test_msa_post_with_seed(mock_session: MagicMock, sample_job_dict: dict):
    """Test msa_post with a seed sequence."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    job = msa_post(mock_session, seed="ACGT")

    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/msa"
    assert kwargs["params"] == {"is_seed": True}
    assert "msa_file" in kwargs["files"]
    assert isinstance(job, Job)


def test_msa_post_with_file(mock_session: MagicMock, sample_job_dict: dict):
    """Test msa_post with a file."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    msa_file = io.BytesIO(b"test")

    job = msa_post(mock_session, msa_file=msa_file)
    mock_session.post.assert_called_once_with(
        "v1/align/msa", files={"msa_file": msa_file}, params={"is_seed": False}
    )
    assert isinstance(job, Job)


def test_msa_post_missing_params(mock_session: MagicMock):
    """Test msa_post raises error if both or neither param is given."""
    with pytest.raises(MissingParameterError):
        msa_post(mock_session)

    with pytest.raises(MissingParameterError):
        msa_post(mock_session, seed="ACGT", msa_file=io.BytesIO())


def test_mafft_post(mock_session: MagicMock, sample_job_dict: dict):
    """Test mafft_post."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    sequence_file = io.BytesIO(b"test")
    mafft_post(mock_session, sequence_file, auto=False, ep=0.1, op=1.5)
    mock_session.post.assert_called_once_with(
        "v1/align/mafft",
        files={"file": sequence_file},
        params={"auto": False, "ep": 0.1, "op": 1.5},
    )


def test_abnumber_post_invalid_scheme(mock_session: MagicMock):
    """Test abnumber_post raises error for invalid scheme."""
    with pytest.raises(
        Exception, match="Antibody numbering invalid_scheme not recognized"
    ):
        abnumber_post(mock_session, io.BytesIO(), scheme="invalid_scheme")  # type: ignore - testing runtime error


def test_prompt_post_valid(mock_session: MagicMock, sample_job_dict: dict):
    """Test prompt_post with valid parameters."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    prompt_post(mock_session, "msa123", num_sequences=10)

    args, kwargs = mock_session.post.call_args
    assert args[0] == "v1/align/prompt"
    assert kwargs["params"]["msa_id"] == "msa123"
    assert kwargs["params"]["max_msa_sequences"] == 10


def test_prompt_post_missing_both_sizing_params(
    mock_session: MagicMock, sample_job_dict: dict
):
    """Test prompt_post uses default if both num_sequences and num_residues are None."""
    mock_session.post.return_value.json.return_value = sample_job_dict
    prompt_post(mock_session, "msa123")
    assert mock_session.post.call_args.kwargs["params"]["max_msa_tokens"] == 12288


def test_prompt_post_invalid_params(mock_session: MagicMock):
    """Test prompt_post raises errors for invalid parameter combinations."""
    with pytest.raises(MissingParameterError):
        prompt_post(mock_session, "msa123", num_sequences=10, num_residues=100)

    with pytest.raises(InvalidParameterError):
        prompt_post(mock_session, "msa123", homology_level=2.0)
