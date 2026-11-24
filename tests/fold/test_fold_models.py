"""Test the models for the fold domain."""

import typing
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openprotein.align import AlignAPI
from openprotein.base import APISession
from openprotein.common.model_metadata import ModelDescription, ModelMetadata
from openprotein.fold.alphafold2 import AlphaFold2Model
from openprotein.fold.boltz import Boltz1Model, Boltz1xModel, Boltz2Model
from openprotein.fold.esmfold import ESMFoldModel
from openprotein.fold.esmfold2 import ESMFold2FastModel, ESMFold2Model
from openprotein.fold.future import FoldResultFuture
from openprotein.fold.models import FoldModel
from openprotein.fold.opendde import OpenDDEAbagModel, OpenDDEModel
from openprotein.fold.protenix import ProtenixModel, ProtenixV2Model
from openprotein.fold.rosettafold3 import RosettaFold3Model
from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs.schemas import JobStatus, JobType
from openprotein.molecules import Complex, Ligand, Protein
from openprotein.prompt import PromptAPI


@pytest.fixture
def mock_session():
    return MagicMock(spec=APISession)


def test_fold_model_create(mock_session):
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
        seed=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_protenix_v2_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
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
        seed=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_esmfold2_model_fold(mock_fold_get, mock_fold_models_post, mock_session):
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

    with (
        patch("openprotein.fold.api.fold_models_post") as mock_post,
        patch("openprotein.fold.api.fold_get"),
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
    assert isinstance(result[0][0], (dict, ESMFold2Confidence))


@pytest.mark.parametrize(
    "model_cls,model_id",
    [(ESMFold2Model, "esmfold2"), (ESMFold2FastModel, "esmfold2-fast")],
)
@pytest.mark.parametrize(
    "param",
    ["num_recycles", "num_steps", "diffusion_samples"],
)
def test_esmfold2_rejects_runtime_param_below_one(
    mock_session, model_cls, model_id, param
):
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


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [("opendde", OpenDDEModel), ("opendde-abag", OpenDDEAbagModel)],
)
def test_fold_model_create_resolves_opendde_variants(mock_session, model_id, expected):
    """Each variant resolves to its own class and keeps its own model id."""
    model = FoldModel.create(mock_session, model_id)

    assert type(model) is expected
    assert model.id == model_id


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
@pytest.mark.parametrize(
    ("model_id", "model_class"),
    [("opendde", OpenDDEModel), ("opendde-abag", OpenDDEAbagModel)],
)
def test_opendde_model_fold(
    mock_fold_get, mock_fold_models_post, mock_session, model_id, model_class
):
    """The posted body carries the knobs the service actually forwards.

    The body carries the sequences, diffusion samples, recycles, steps and
    seed; templates are not part of it.
    """
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id=model_id)
    mock_session.prompt = MagicMock(spec=PromptAPI)

    model = model_class(session=mock_session, model_id=model_id)
    future = model.fold(sequences=["SEQ"])

    mock_fold_models_post.assert_called_once_with(
        session=mock_session,
        model_id=model_id,
        sequences=[[{"protein": {"id": "A", "sequence": "SEQ", "msa_id": None}}]],
        diffusion_samples=1,
        num_recycles=10,
        num_steps=200,
        seed=None,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.job.job_id == "job123"


@pytest.mark.parametrize("model_id", ["opendde", "opendde-abag"])
def test_opendde_confidence_parses_the_fields_the_model_emits(model_id):
    """Guard the confidence schema against what the model actually emits.

    chain_pair_plddt is the field OpenDDE emits that Protenix does not, so
    reusing ProtenixConfidence would silently drop it.
    """
    from openprotein.fold.opendde import OpenDDEConfidence

    summary = {
        "ranking_score": 0.82,
        "ptm": 0.7,
        "iptm": 0.85,
        "plddt": 91.2,
        "gpde": 1.1,
        "has_clash": 0.0,
        "num_recycles": 10,
        "disorder": 0.0,
        "chain_ptm": [0.7],
        "chain_iptm": [0.85],
        "chain_plddt": [91.2],
        "chain_gpde": [1.1],
        "chain_pair_iptm": [[0.85]],
        "chain_pair_iptm_global": [[0.85]],
        "chain_pair_gpde": [[1.1]],
        "chain_pair_plddt": [[91.2]],
    }
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    future.get = MagicMock(return_value=[[summary]])

    parsed = OpenDDEConfidence.model_validate(summary)

    assert parsed.chain_pair_plddt == [[91.2]]

    # Scores the schema does not declare are kept, not rejected.
    extended = OpenDDEConfidence.model_validate(
        summary | {"pb_ranking_score": 0.7, "has_vdw_pl_clash": 0.0}
    )
    assert extended.pb_ranking_score == 0.7


@pytest.mark.parametrize("model_id", ["opendde", "opendde-abag"])
def test_opendde_exposes_its_outputs(model_id):
    """pae, pde, plddt and confidence are gated per model in FoldResultFuture.

    The service publishes exactly these four plus the mmCIF. ipae is not
    among them, so asking for it has to raise rather than fail at fetch time.
    """
    future = object.__new__(FoldResultFuture)
    future._metadata = FoldMetadata(job_id="job123", model_id=model_id)
    future.get = MagicMock(return_value=[])

    for key in ("pae", "pde", "plddt", "confidence"):
        getattr(future, f"get_{key}")()

    with pytest.raises(AttributeError, match="ipae not supported"):
        future.get_ipae()


MSA_ID = "3f1c2a7e-9b4d-4e2a-8c1f-0d5e6a7b8c9d"


@pytest.fixture
def generic_fold(mock_session):
    from openprotein.fold.fold import FoldAPI

    job = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    metadata = FoldMetadata(job_id="job123", model_id="example-fold-v1")
    with (
        patch("openprotein.fold.api.fold_models_post", return_value=job) as post,
        patch("openprotein.fold.api.fold_get", return_value=metadata),
    ):
        model = FoldAPI(session=mock_session).get_model("example-fold-v1")
        assert type(model) is FoldModel
        yield model, post


class TestGenericFoldModel:
    """A listed model without its own class folds through ``FoldModel``."""

    def test_the_sequences_it_folds_are_sent(self, generic_fold):
        model, post = generic_fold

        future = model.fold(["MKT", "AC:DE"], num_recycles=3)

        assert post.call_args.kwargs["sequences"] == [
            [{"protein": {"id": "A", "sequence": "MKT"}}],
            [
                {"protein": {"id": "A", "sequence": "AC"}},
                {"protein": {"id": "B", "sequence": "DE"}},
            ],
        ]
        assert post.call_args.kwargs["num_recycles"] == 3
        assert [c.get_proteins()["A"].sequence for c in future.complexes] == [
            b"MKT",
            b"AC",
        ]

    @pytest.mark.parametrize(
        ("msa", "msa_id"),
        [(MSA_ID, MSA_ID), (Protein.single_sequence_mode, None)],
    )
    def test_an_attached_msa_sends_an_msa_id_for_every_protein(
        self, generic_fold, mock_session, msa, msa_id
    ):
        model, post = generic_fold
        mock_session.align = MagicMock(spec=AlignAPI)
        mock_session.align.get_seed.return_value = "MKT"
        protein = Protein(sequence="MKT")
        protein.msa = msa

        model.fold([protein, "AC"])

        assert post.call_args.kwargs["sequences"] == [
            [{"protein": {"id": "A", "sequence": "MKT", "msa_id": msa_id}}],
            [{"protein": {"id": "A", "sequence": "AC", "msa_id": None}}],
        ]

    def test_a_protein_without_an_msa_beside_one_with_an_msa_is_refused(
        self, generic_fold
    ):
        model, post = generic_fold
        aligned = Protein(sequence="MKT")
        aligned.msa = Protein.single_sequence_mode

        with pytest.raises(ValueError, match="`.msa` set"):
            model.fold([aligned, Protein(sequence="AC")])
        post.assert_not_called()

    def test_an_empty_list_sends_no_sequences(self, generic_fold):
        model, post = generic_fold
        references = [{"job_id": "job456", "prompt_id": "prompt789"}]

        model.fold([], reference_structures=references)

        assert post.call_args.kwargs["sequences"] is None
        assert post.call_args.kwargs["reference_structures"] == references


class TestFoldModelHierarchy:
    def test_every_model_still_resolves(self, mock_session):
        expected = {
            "boltz-2": "Boltz2Model",
            "boltz-1x": "Boltz1xModel",
            "esmfold": "ESMFoldModel",
            "alphafold2": "AlphaFold2Model",
            "opendde-abag": "OpenDDEAbagModel",
        }
        for model_id, class_name in expected.items():
            model = FoldModel.create(
                session=mock_session, model_id=model_id, default=FoldModel
            )
            assert type(model).__name__ == class_name, model_id

    def test_fold_model_create_resolves_grandchildren(self, mock_session):
        model = FoldModel.create(
            session=mock_session, model_id="boltz-2", default=FoldModel
        )
        assert type(model).__name__ == "Boltz2Model"

    def test_fold_api_lookups_are_typed_as_fold_model(self):
        from openprotein.fold.fold import FoldAPI

        hints = typing.get_type_hints(FoldAPI.get_model)
        assert hints["return"] is FoldModel
        assert typing.get_type_hints(FoldAPI.list_models)["return"] == list[FoldModel]

    def test_bare_fold_model_reports_its_id(self, mock_session):
        from openprotein.fold.models import FoldModel

        assert FoldModel(session=mock_session, model_id="unknown").model_id == "unknown"


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_protenix_model_fold_sends_the_seed(
    mock_fold_get, mock_fold_models_post, mock_session
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id="protenix-v2")
    mock_session.prompt = MagicMock(spec=PromptAPI)

    model = ProtenixV2Model(session=mock_session, model_id="protenix-v2")
    model.fold(sequences=["SEQ"], seed=42)

    assert mock_fold_models_post.call_args.kwargs["seed"] == 42


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
@pytest.mark.parametrize(
    ("model_id", "model_class"),
    [
        ("boltz-1", Boltz1Model),
        ("boltz-1x", Boltz1xModel),
        ("boltz-2", Boltz2Model),
        ("opendde", OpenDDEModel),
        ("opendde-abag", OpenDDEAbagModel),
        ("rosettafold-3", RosettaFold3Model),
    ],
)
def test_fold_sends_the_seed(
    mock_fold_get, mock_fold_models_post, mock_session, model_id, model_class
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id=model_id)
    mock_session.prompt = MagicMock(spec=PromptAPI)

    model = model_class(session=mock_session, model_id=model_id)
    model.fold(sequences=["SEQ"], seed=42)

    assert mock_fold_models_post.call_args.kwargs["seed"] == 42


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
@pytest.mark.parametrize(
    ("model_id", "model_class"),
    [("boltz-2", Boltz2Model), ("protenix", ProtenixModel)],
)
def test_a_colon_string_folds_as_one_multichain_complex(
    mock_fold_get, mock_fold_models_post, mock_session, model_id, model_class
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id=model_id)
    mock_session.prompt = MagicMock(spec=PromptAPI)

    model_class(session=mock_session, model_id=model_id).fold("MKT:AC")

    assert mock_fold_models_post.call_args.kwargs["sequences"] == [
        [
            {"protein": {"id": "A", "sequence": "MKT", "msa_id": None}},
            {"protein": {"id": "B", "sequence": "AC", "msa_id": None}},
        ]
    ]


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_a_later_complex_with_a_chain_the_first_lacks_folds(
    mock_fold_get, mock_fold_models_post, mock_session
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id="boltz-2")
    mock_session.prompt = MagicMock(spec=PromptAPI)

    Boltz2Model(session=mock_session, model_id="boltz-2").fold(["MKT", "MKT:AC"])

    assert [
        len(complex) for complex in mock_fold_models_post.call_args.kwargs["sequences"]
    ] == [1, 2]


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_shared_template_targets_are_checked_before_upload_or_submission(
    mock_fold_get, mock_fold_models_post, mock_session
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id="boltz-2")
    mock_session.prompt = PromptAPI(mock_session)
    template = Protein("ARNDCQ")
    template = template._set_coordinates(
        np.zeros(template.coordinates.shape, dtype=np.float32)
    )
    first = Protein("ARNDCQ").set_msa(Protein.single_sequence_mode)
    first.templates = [template]
    later = Protein("MKT").set_msa(Protein.single_sequence_mode)

    with patch.object(
        mock_session.prompt, "_resolve_query", return_value="query-id"
    ) as upload:
        with pytest.raises(ValueError, match="Template maps.*does not exist"):
            Boltz2Model(session=mock_session, model_id="boltz-2").fold(
                [Complex({"A": first}), Complex({"B": later})]
            )
        upload.assert_not_called()
    mock_fold_models_post.assert_not_called()


@patch("openprotein.fold.api.fold_models_post")
@patch("openprotein.fold.api.fold_get")
def test_boltz2_takes_an_affinity_binder_in_a_lone_complex(
    mock_fold_get, mock_fold_models_post, mock_session
):
    mock_fold_models_post.return_value = FoldJob(
        job_id="job123",
        status=JobStatus.PENDING,
        job_type=JobType.embeddings_fold,
        created_date=datetime.now(timezone.utc),
    )
    mock_fold_get.return_value = FoldMetadata(job_id="job123", model_id="boltz-2")
    mock_session.prompt = MagicMock(spec=PromptAPI)
    protein = Protein(sequence="MKT")
    protein.msa = Protein.single_sequence_mode
    properties = [{"affinity": {"binder": "L"}}]

    Boltz2Model(session=mock_session, model_id="boltz-2").fold(
        Complex({"A": protein, "L": Ligand(ccd="ATP")}), properties=properties
    )

    assert mock_fold_models_post.call_args.kwargs["properties"] == properties
