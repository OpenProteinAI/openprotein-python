"""L2 integration tests for the embeddings domain."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.embeddings.embeddings import EmbeddingsAPI
from openprotein.embeddings.poet import PoETModel


def test_embeddings_api_init(mock_session: MagicMock):
    """Test that the EmbeddingsAPI initializes models correctly."""
    # Create mock model instances with 'id' attributes
    mock_model1 = MagicMock()
    mock_model1.id = "prot-seq"
    mock_model2 = MagicMock()
    mock_model2.id = "poet-2"

    mock_models_list = [mock_model1, mock_model2]

    # Patch list_models to return our list of configured mocks
    with patch.object(EmbeddingsAPI, "list_models", return_value=mock_models_list):
        api = EmbeddingsAPI(mock_session)

        # Check that the models were set as attributes with the correct names
        assert hasattr(api, "prot_seq")
        assert hasattr(api, "poet2")
        assert api.prot_seq == mock_model1
        assert api.poet2 == mock_model2


def test_embeddings_api_get_model(mock_session: MagicMock):
    """Test the get_model method of the EmbeddingsAPI."""
    mock_prot_seq = MagicMock()
    mock_prot_seq.id = "prot-seq"

    with patch.object(EmbeddingsAPI, "list_models", return_value=[mock_prot_seq]):
        api = EmbeddingsAPI(mock_session)

        # Test getting the model by its name
        model = api.get_model("prot-seq")
        assert model is mock_prot_seq

        # Test getting the model with a hyphenated name
        model_by_hyphen = api.get_model("prot-seq")
        assert model_by_hyphen is mock_prot_seq


def test_embeddings_api_model_call(mock_session: MagicMock):
    """Test calling a method on a model through the EmbeddingsAPI."""
    mock_poet_instance = MagicMock(spec=PoETModel)
    mock_poet_instance.id = "poet"

    with patch.object(EmbeddingsAPI, "list_models", return_value=[mock_poet_instance]):
        api = EmbeddingsAPI(mock_session)

        # Access the dynamically set attribute and call a method
        api.poet.embed(sequences=[b"ACGT"], prompt="p1")

        # Verify that the embed method on our mock instance was called
        mock_poet_instance.embed.assert_called_once_with(
            sequences=[b"ACGT"], prompt="p1"
        )
