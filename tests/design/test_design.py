"""L2 integration tests for the design domain."""
from unittest.mock import MagicMock

import pytest

from openprotein.design.design import DesignAPI
from openprotein.design.future import DesignFuture
from openprotein.design.schemas import Criteria, Criterion, ModelCriterion
from openprotein.jobs import JobStatus, JobType


@pytest.fixture
def design_api(mock_session: MagicMock) -> DesignAPI:
    """Fixture for a DesignAPI instance."""
    return DesignAPI(mock_session)


def test_create_design(design_api: DesignAPI, mock_session: MagicMock):
    """Test the create_genetic_algorithm_design method."""
    # Mock for the POST call that creates the design
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "job_type": JobType.designer.value,
        "created_date": "2023-01-01T00:00:00",
    }
    # Mock for the GET call that happens inside the DesignFuture constructor
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "design-123",
        "status": "SUCCESS",
        "progress_counter": 100,
        "created_date": "2023-01-01T00:00:00",
        "algorithm": "genetic-algorithm",
        "num_rows": 10,
        "num_steps": 10,
        "assay_id": "a1",
        "criteria": Criteria(root=[]).model_dump(),
        "allowed_tokens": {},
        "pop_size": 128,
        "n_offsprings": 256,
        "crossover_prob": 0.8,
        "crossover_prob_pointwise": 0.1,
        "mutation_average_mutations_per_seq": 2,
    }

    mock_assay = MagicMock()
    mock_assay.id = "a1"
    mc = ModelCriterion(model_id="m1", measurement_name="p1") > 1
    criteria = Criteria([Criterion([mc])])

    future = design_api.create_genetic_algorithm_design(mock_assay, criteria)
    mock_session.post.assert_called_once()
    assert isinstance(future, DesignFuture)
