"""Test the models for the fold domain."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from openprotein.base import APISession
from openprotein.common.model_metadata import ModelDescription, ModelMetadata
from openprotein.fold.alphafold2 import AlphaFold2Model
from openprotein.fold.esmfold import ESMFoldModel
from openprotein.fold.esmfold2 import ESMFold2FastModel, ESMFold2Model
from openprotein.fold.future import FoldResultFuture
from openprotein.fold.models import FoldModel
from openprotein.fold.protenix import ProtenixModel, ProtenixV2Model
from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs.schemas import JobStatus, JobType
from openprotein.prompt import PromptAPI


@pytest.fixture
def mock_session():
    """Fixture for a mocked APISession."""
    return MagicMock(spec=APISession)


def test_fold_model_create(mock_session):
    """Test FoldModel.create."""
    # Test creating a known model
    model = FoldModel.create(mock_session, "esmfold")
    assert isinstance(model, ESMFoldModel)

    # Test creating another known model
    model = FoldModel.create(mock_session, "alphafold2")
    assert isinstance(model, AlphaFold2Model)

    # Test creating a generic model
    model = FoldModel.create(mock_session, "some-other-model", default=FoldModel)
    assert isinstance(model, FoldModel)
    assert not isinstance(model, (ESMFoldModel, AlphaFold2Model))

    # Test creating Protenix model
    model = FoldModel.create(mock_session, "protenix")
    assert isinstance(model, ProtenixModel)

    # Test creating ESMFold2 models
    model = FoldModel.create(mock_session, "esmfold2")
    assert isinstance(model, ESMFold2Model)
    assert not isinstance(model, ESMFold2FastModel)

    model = FoldModel.create(mock_session, "esmfold2-fast")
    assert isinstance(model, ESMFold2FastModel)

    # Test creating Protenix-v2 model
    model = FoldModel.create(mock_session, "protenix-v2")
    assert isinstance(model, ProtenixV2Model)
    assert model.id == "protenix-v2"

    # Test ValueError for unsupported model
    with pytest.raises(ValueError):
        FoldModel.create(mock_session, "unsupported-model")


@patch("openprotein.fold.api.fold_model_get")
def test_fold_model_get_metadata(mock_fold_model_get, mock_session):
    """Test FoldModel.get_metadata and caching."""
    mock_metadata = ModelMetadata(
        model_id="esmfold",
        description=ModelDescription(summary="A test model"),
        dimension=128,
        output_types=["pdb"],
        input_tokens=["protein"],
        token_descriptions=[],
    )
    mock_fold_model_get.return_value = mock_metadata

    model = ESMFoldModel(session=mock_session, model_id="esmfold")

    # Access metadata for the first time
    metadata = model.metadata
    assert metadata == mock_metadata
    mock_fold_model_get.assert_called_once_with(mock_session, "esmfold")

    # Access metadata again to test caching
    metadata2 = model.metadata
    assert metadata2 == mock_metadata
    # Assert that the mock was not called again
    mock_fold_model_get.assert_called_once()


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_fold_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test FoldModel.fold."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="esmfold",
    )

    model = ESMFoldModel(session=mock_session, model_id="esmfold")
    future = model.fold(sequences=["SEQ"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="esmfold",
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ"}}]],
        num_recycles=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_protenix_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test ProtenixModel.fold."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="protenix",
    )

    mock_session.prompt = MagicMock(spec=PromptAPI)

    model = ProtenixModel(session=mock_session, model_id="protenix")
    future = model.fold(sequences=["SEQ"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="protenix",
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ", "msa_id": None}}]],
        diffusion_samples=1,
        num_recycles=10,
        num_steps=200,
        templates=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_protenix_v2_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test ProtenixV2Model.fold posts the protenix-v2 model_id."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="protenix-v2",
    )

    mock_session.prompt = MagicMock(spec=PromptAPI)

    model = ProtenixV2Model(session=mock_session, model_id="protenix-v2")
    future = model.fold(sequences=["SEQ"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="protenix-v2",
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ", "msa_id": None}}]],
        diffusion_samples=1,
        num_recycles=10,
        num_steps=200,
        templates=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_esmfold2_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test ESMFold2Model.fold."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="esmfold2",
    )

    model = ESMFold2Model(session=mock_session, model_id="esmfold2")
    future = model.fold(sequences=["SEQ"], num_recycles=2, seed=42)

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="esmfold2",
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ", "msa_id": None}}]],
        diffusion_samples=1,
        num_recycles=2,
        num_steps=100,
        seed=42,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_esmfold2_fast_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
    """Test ESMFold2FastModel.fold."""
    mock_job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_models_post.return_value = mock_job

    mock_fold_get.return_value = FoldMetadata(
        job_id="job123",
        model_id="esmfold2-fast",
    )

    model = ESMFold2FastModel(session=mock_session, model_id="esmfold2-fast")
    future = model.fold(sequences=["SEQ"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id="esmfold2-fast",
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ", "msa_id": None}}]],
        diffusion_samples=1,
        num_recycles=3,
        num_steps=100,
        seed=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


def test_esmfold2_fast_rejects_protein_with_msa_future(mock_session):
    """ESMFold2-Fast must reject Complex inputs whose proteins carry an MSAFuture."""
    from openprotein.align.msa import MSAFuture
    from openprotein.molecules import Complex, Protein

    msa_future = MagicMock(spec=MSAFuture)
    msa_future.id = "msa-xyz"
    protein = Protein(sequence="SEQ")
    protein.msa = msa_future
    complex = Complex(chains={"A": protein})

    model = ESMFold2FastModel(session=mock_session, model_id="esmfold2-fast")
    with pytest.raises(ValueError, match="single-sequence"):
        model.fold(sequences=[complex])


def test_esmfold2_fast_rejects_protein_with_string_msa_id(mock_session):
    """ESMFold2-Fast must reject Complex inputs whose proteins reference an msa_id."""
    from openprotein.molecules import Complex, Protein

    protein = Protein(sequence="SEQ")
    protein.msa = "some-msa-id"
    complex = Complex(chains={"A": protein})

    model = ESMFold2FastModel(session=mock_session, model_id="esmfold2-fast")
    with pytest.raises(ValueError, match="single-sequence"):
        model.fold(sequences=[complex])


def test_esmfold2_fast_accepts_single_sequence_mode(mock_session):
    """ESMFold2-Fast must accept single_sequence_mode without raising."""
    from openprotein.molecules import Complex, Protein

    with patch("openprotein.fold.api.fold_models_post") as mock_post, patch(
        "openprotein.fold.api.fold_get"
    ):
        mock_post.return_value = FoldJob(
            job_id="job123",
            status=JobStatus.PENDING,
            job_type=JobType.embeddings_fold,
            created_date=datetime.now(timezone.utc),
        )

        protein = Protein(sequence="SEQ")
        protein.msa = Protein.single_sequence_mode
        complex = Complex(chains={"A": protein})

        model = ESMFold2FastModel(session=mock_session, model_id="esmfold2-fast")
        future = model.fold(sequences=[complex])
        assert isinstance(future, FoldResultFuture)


@pytest.mark.parametrize("model_id", ["protenix", "protenix-v2"])
def test_fold_result_future_get_confidence_supports_protenix(model_id):
    """Test FoldResultFuture.get_confidence for Protenix and Protenix-v2."""
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    future.get = MagicMock(return_value=[[{"confidence_score": 0.5}]])

    result = future.get_confidence()

    future.get.assert_called_once_with(key="confidence")
    assert result == [[{"confidence_score": 0.5}]]


def test_fold_result_future_get_confidence_supports_protenix_v2():
    """get_confidence is whitelisted for protenix-v2 and returns parsed data."""
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id="protenix-v2")
    future.get = MagicMock(return_value=[[{"confidence_score": 0.5}]])

    result = future.get_confidence()

    future.get.assert_called_once_with(key="confidence")
    assert result == [[{"confidence_score": 0.5}]]


@pytest.mark.parametrize("model_id", ["esmfold2", "esmfold2-fast"])
def test_fold_result_future_get_confidence_supports_esmfold2(model_id):
    """get_confidence dispatches to ESMFold2Confidence schema for ESMFold2 models."""
    from openprotein.fold.esmfold2 import ESMFold2Confidence

    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    future.get = MagicMock(
        return_value=[
            [
                {
                    "ptm": 0.8,
                    "iptm": 0.7,
                    "complex_plddt": 90.0,
                    "chains_ptm": {"0": 0.85},
                    "pair_chains_iptm": {"0": {"0": 0.85}},
                }
            ]
        ]
    )

    result = future.get_confidence()
    future.get.assert_called_once_with(key="confidence")
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
    assert isinstance(result[0][0], dict) or isinstance(result[0][0], ESMFold2Confidence)


@pytest.mark.parametrize(
    "model_cls,model_id",
    [(ESMFold2Model, "esmfold2"), (ESMFold2FastModel, "esmfold2-fast")],
)
@pytest.mark.parametrize(
    "param",
    ["num_recycles", "num_steps", "diffusion_samples"],
)
def test_esmfold2_rejects_runtime_param_below_one(mock_session, model_cls, model_id, param):
    """Runtime params must be >= 1; the SDK rejects bad values before the request."""
    model = model_cls(session=mock_session, model_id=model_id)
    with pytest.raises(ValueError, match=param):
        model.fold(sequences=["SEQ"], **{param: 0})


@pytest.mark.parametrize(
    "model_id,key",
    [
        ("protenix", "pae"),
        ("protenix", "pde"),
        ("protenix", "plddt"),
        ("protenix", "ipae"),
        ("protenix-v2", "pae"),
        ("protenix-v2", "pde"),
        ("protenix-v2", "plddt"),
        ("protenix-v2", "ipae"),
        ("protenix-v2", "confidence"),
    ],
)
def test_fold_result_future_extras_allowed_for_protenix_family(model_id, key):
    """Server produces these extras for protenix/protenix-v2; SDK must allow them."""
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    future.get = MagicMock(return_value=[])
    method = {
        "pae": "get_pae",
        "pde": "get_pde",
        "plddt": "get_plddt",
        "ipae": "get_ipae",
        "confidence": "get_confidence",
    }[key]
    # Should not raise AttributeError.
    getattr(future, method)()


@pytest.mark.parametrize("model_id", ["boltz-1", "boltz-1x"])
def test_fold_result_future_affinity_rejects_non_boltz2(model_id):
    """Only boltz-2 actually produces affinity; boltz-1/boltz-1x must be rejected."""
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    with pytest.raises(AttributeError, match="boltz-2"):
        future.get_affinity()
    with pytest.raises(AttributeError, match="boltz-2"):
        future.get_affinity_batch()
