from datetime import datetime
import warnings
import pytest
from unittest.mock import MagicMock, patch

from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.jobs import Job, JobStatus, JobType
from openprotein.align.msa import MSAFuture
from openprotein.align.schemas import (
    AbNumberJob,
    JobWarning,
    MafftJob,
    MSASamplingMethod,
)
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


_UNNUMBERED_WARNING = {
    "code": "UNNUMBERED_RESIDUES_DROPPED",
    "message": "Residues outside the region AbNumber numbered were dropped.",
    "metadata": {
        "residue_count": 103,
        "chain_count": 1,
        "chains": [
            {
                "chain_index": 0,
                "sequence_count": 3,
                "residue_count": 103,
                "max_prefix_length": 0,
                "max_suffix_length": 103,
                "sampled": False,
                "sequences": [{"name": "s0", "prefix_length": 0, "suffix_length": 103}],
            }
        ],
    },
}


@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_antibody_schema(mock_schema_get, msa_future: MSAFuture):
    """get_antibody_schema passes the job through to the API helper."""
    msa_future.get_antibody_schema()
    mock_schema_get.assert_called_once_with(
        session=msa_future.session, job_id=msa_future.job.job_id
    )


@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_warnings(mock_schema_get, msa_future: MSAFuture):
    """get_warnings parses the extras' warnings into models."""
    mock_schema_get.return_value = {"chains": [], "warnings": [_UNNUMBERED_WARNING]}

    (warning,) = msa_future.get_warnings()

    assert isinstance(warning, JobWarning)
    assert warning.code == "UNNUMBERED_RESIDUES_DROPPED"
    assert warning.metadata is not None
    assert warning.metadata["chains"][0]["max_suffix_length"] == 103


@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_warnings_empty(mock_schema_get, msa_future: MSAFuture):
    """A job with nothing to report has no warnings."""
    mock_schema_get.return_value = {"chains": [], "warnings": []}

    assert msa_future.get_warnings() == []


@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_warnings_absent_key(mock_schema_get, msa_future: MSAFuture):
    """Extras with no ``warnings`` key report nothing."""
    mock_schema_get.return_value = {"chains": []}

    assert msa_future.get_warnings() == []


@pytest.fixture
def abnumber_job() -> AbNumberJob:
    """Fixture for a sample AbNumberJob object."""
    return AbNumberJob(
        job_id="job456",
        status=JobStatus.SUCCESS,
        job_type=JobType.abnumber,
        created_date=datetime.fromisoformat("2023-01-01T00:00:00"),
    )


@pytest.fixture
def abnumber_future(mock_session: MagicMock, abnumber_job: AbNumberJob) -> MSAFuture:
    """Fixture for an MSAFuture over an AbNumber job."""
    return MSAFuture(mock_session, abnumber_job)


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_warns_on_reported_warning(
    mock_schema_get, mock_get_msa, abnumber_future: MSAFuture
):
    """Fetching the results surfaces what the job reported about them."""
    mock_schema_get.return_value = {
        "chains": [],
        "warnings": [
            {
                "code": "UNNUMBERED_RESIDUES_DROPPED",
                "message": "16 residues were dropped",
                "metadata": None,
            }
        ],
    }

    with pytest.warns(UserWarning, match="UNNUMBERED_RESIDUES_DROPPED"):
        abnumber_future.get()

    mock_get_msa.assert_called_once()


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_wait_warns_on_reported_warning(
    mock_schema_get, mock_get_msa, abnumber_future: MSAFuture
):
    """`wait` reaches the results through `get`, so it warns too."""
    mock_schema_get.return_value = {
        "warnings": [{"code": "UNNUMBERED_RESIDUES_RETAINED", "message": "kept 16"}]
    }

    with (
        patch.object(MSAFuture, "_wait_job", return_value=abnumber_future.job),
        patch("time.sleep"),
    ):
        with pytest.warns(UserWarning, match="UNNUMBERED_RESIDUES_RETAINED"):
            abnumber_future.wait()


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_warns_once_per_future(
    mock_schema_get, mock_get_msa, abnumber_future: MSAFuture
):
    """The schema is fetched once, however many times the results are read."""
    mock_schema_get.return_value = {
        "warnings": [{"code": "UNNUMBERED_RESIDUES_DROPPED", "message": "dropped 16"}]
    }

    with pytest.warns(UserWarning):
        abnumber_future.get()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        abnumber_future.get()

    assert caught == []
    mock_schema_get.assert_called_once()


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_silent_without_warnings(
    mock_schema_get, mock_get_msa, abnumber_future: MSAFuture
):
    """A job that reported nothing warns about nothing."""
    mock_schema_get.return_value = {"chains": [], "warnings": []}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        abnumber_future.get()

    assert caught == []


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_skips_schema_for_other_methods(
    mock_schema_get, mock_get_msa, msa_future: MSAFuture
):
    """Only AbNumber jobs have an antibody schema to read."""
    msa_future.get()

    mock_schema_get.assert_not_called()


@patch("openprotein.align.api.get_msa")
@patch("openprotein.align.api.antibody_schema_get")
def test_msa_future_get_survives_unreachable_schema(
    mock_schema_get, mock_get_msa, abnumber_future: MSAFuture
):
    """Results stay readable when the schema endpoint does not answer."""
    mock_schema_get.side_effect = APIError("no such endpoint")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        abnumber_future.get()

    assert caught == []
    mock_get_msa.assert_called_once()
