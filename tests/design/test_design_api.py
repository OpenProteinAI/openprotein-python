"""Test the api for the design domain."""
from unittest.mock import MagicMock

import pytest

from openprotein.design import api
from openprotein.design.schemas import Criteria, Criterion, ModelCriterion
from openprotein.jobs import JobType


def test_designs_list(mock_session: MagicMock):
    """Test designs_list."""
    mock_session.get.return_value.json.return_value = []
    api.designs_list(mock_session)
    mock_session.get.assert_called_once_with("v1/designer/design")


def test_designer_create_genetic_algorithm(mock_session: MagicMock):
    """Test designer_create_genetic_algorithm."""
    mock_session.post.return_value.json.return_value = {
        "job_id": "job-123",
        "status": "SUCCESS",
        "created_date": "2023-01-01T00:00:00",
        "job_type": JobType.designer.value,
    }
    mc = ModelCriterion(model_id="m1", measurement_name="p1") > 1
    criteria = Criteria([Criterion([mc])])

    api.designer_create_genetic_algorithm(
        mock_session, assay_id="a1", criteria=criteria
    )
    mock_session.post.assert_called_once()
    call_args, call_kwargs = mock_session.post.call_args
    assert call_kwargs["json"]["assay_id"] == "a1"
    assert "criteria" in call_kwargs["json"]
