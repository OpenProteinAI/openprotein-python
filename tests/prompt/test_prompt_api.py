import io
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from openprotein.errors import APIError, InvalidParameterError
from openprotein.jobs import JobStatus
from openprotein.protein import Protein
from openprotein.prompt.api import (
    create_prompt,
    create_query,
    get_prompt,
    get_prompt_metadata,
    get_query,
    get_query_metadata,
    list_prompts,
)
from openprotein.prompt.schemas import PromptMetadata, QueryMetadata


def test_create_prompt(mock_session: MagicMock):
    """Test the create_prompt function."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": datetime.now().isoformat(),
        "num_replicates": 1,
        "status": "SUCCESS",
    }

    context = ["ACGT"]
    metadata = create_prompt(mock_session, context, name="Test Prompt")

    mock_session.post.assert_called_once()
    assert isinstance(metadata, PromptMetadata)
    assert metadata.id == "prompt-123"


def test_get_prompt_metadata(mock_session: MagicMock):
    """Test the get_prompt_metadata function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "prompt-123",
        "name": "Test Prompt",
        "created_date": datetime.now().isoformat(),
        "num_replicates": 1,
        "status": "SUCCESS",
    }

    metadata = get_prompt_metadata(mock_session, "prompt-123")

    mock_session.get.assert_called_with("v1/prompt/prompt-123")
    assert isinstance(metadata, PromptMetadata)
    assert metadata.id == "prompt-123"


def test_list_prompts(mock_session: MagicMock):
    """Test the list_prompts function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = [
        {
            "id": "prompt-123",
            "name": "Test Prompt 1",
            "created_date": datetime.now().isoformat(),
            "num_replicates": 1,
            "status": "SUCCESS",
        },
        {
            "id": "prompt-456",
            "name": "Test Prompt 2",
            "created_date": datetime.now().isoformat(),
            "num_replicates": 1,
            "status": "PENDING",
        },
    ]

    prompts = list_prompts(mock_session)

    mock_session.get.assert_called_with("v1/prompt")
    assert len(prompts) == 2
    assert all(isinstance(p, PromptMetadata) for p in prompts)


def test_create_query(mock_session: MagicMock):
    """Test the create_query function."""
    mock_session.post.return_value.status_code = 200
    mock_session.post.return_value.json.return_value = {
        "id": "query-123",
        "created_date": datetime.now().isoformat(),
    }

    query = "ACGT"
    metadata = create_query(mock_session, query)

    mock_session.post.assert_called_once()
    assert isinstance(metadata, QueryMetadata)
    assert metadata.id == "query-123"


def test_get_query_metadata(mock_session: MagicMock):
    """Test the get_query_metadata function."""
    mock_session.get.return_value.status_code = 200
    mock_session.get.return_value.json.return_value = {
        "id": "query-123",
        "created_date": datetime.now().isoformat(),
    }

    metadata = get_query_metadata(mock_session, "query-123")

    mock_session.get.assert_called_with("v1/prompt/query/query-123")
    assert isinstance(metadata, QueryMetadata)
    assert metadata.id == "query-123"


def test_api_error_handling(mock_session: MagicMock):
    """Test that APIError is raised for non-200 status codes."""
    mock_session.get.return_value.status_code = 404
    mock_session.get.return_value.json.return_value = {"detail": "Not Found"}

    with pytest.raises(APIError, match="Not Found"):
        get_prompt_metadata(mock_session, "non-existent-id")


def test_invalid_parameter_error(mock_session: MagicMock):
    """Test that InvalidParameterError is raised for 400 status code."""
    mock_session.post.return_value.status_code = 400
    mock_session.post.return_value.json.return_value = {"detail": "Invalid parameters"}

    with pytest.raises(InvalidParameterError, match="Invalid parameters"):
        create_prompt(mock_session, context=["ACGT"], name="Invalid Prompt")
