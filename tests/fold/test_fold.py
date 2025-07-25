"""L2 integration tests for the fold domain."""
from unittest.mock import MagicMock

from openprotein.fold.alphafold2 import AlphaFold2Model
from openprotein.fold.boltz import Boltz1Model
from openprotein.fold.esmfold import ESMFoldModel
from openprotein.fold.fold import FoldAPI
from openprotein.fold.future import FoldResultFuture
from openprotein.jobs.schemas import Job


def test_fold_api_init(mock_session: MagicMock):
    """Test FoldAPI initialization by mocking the underlying session call."""
    # Configure the mock to return a list of models first, then model metadata
    mock_session.get.return_value.json.side_effect = [
        ["esmfold", "alphafold2", "boltz-1"],  # First call for list_models
        {  # Subsequent calls for get_model
            "model_id": "model1",
            "description": {"summary": "A test model"},
            "dimension": 128,
            "output_types": ["pdb"],
            "input_tokens": ["protein"],
            "token_descriptions": [],
        },
    ] * 4  # Repeat the metadata for each get_model call

    fold_api = FoldAPI(mock_session)

    assert isinstance(fold_api.esmfold, ESMFoldModel)
    assert isinstance(fold_api.alphafold2, AlphaFold2Model)
    assert isinstance(fold_api.boltz_1, Boltz1Model)


def test_get_results(mock_session: MagicMock):
    """
    Test FoldAPI.get_results integration.
    """
    # Configure mock responses for the sequence of API calls
    mock_session.get.return_value.json.side_effect = [
        [],  # Initial call in FoldAPI constructor
        {
            "job_id": "test_job_id",
            "model_id": "esmfold",
        },  # Call within get_results (fold_get)
        [b"ACGT"],  # Call within FoldResultFuture constructor (fold_get_sequences)
    ]
    mock_session.get.return_value.status_code = 200

    fold_api = FoldAPI(mock_session)
    mock_job = MagicMock(spec=Job)
    mock_job.job_id = "test_job_id"

    future = fold_api.get_results(mock_job)

    assert isinstance(future, FoldResultFuture)
    # Check that the underlying api calls were made
    assert mock_session.get.call_count == 3
