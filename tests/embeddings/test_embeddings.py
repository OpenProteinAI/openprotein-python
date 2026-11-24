"""L2 integration tests for the embeddings domain."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.embeddings import api as embeddings_api_module
from openprotein.embeddings.embeddings import EmbeddingsAPI
from openprotein.embeddings.models import EmbeddingModel
from openprotein.models import ModelsAPI


def test_embeddings_api_init_is_lazy(mock_session: MagicMock):
    """Constructing the API makes no requests; models resolve on access."""
    session = MagicMock()
    session.models = ModelsAPI(session)

    EmbeddingsAPI(session)

    assert session.mock_calls == []


def test_embeddings_api_resolves_legacy_handles(mock_session: MagicMock):
    """Embedding handles retain their legacy types."""
    from contextlib import ExitStack

    from openprotein.embeddings import (
        ESMModel,
        OpenProteinModel,
    )
    from openprotein.embeddings.poet2 import PoET2Model
    from openprotein.models import ModelsAPI

    session = MagicMock()
    session.models = ModelsAPI(mock_session)
    api = EmbeddingsAPI(session)

    with ExitStack() as stack:
        for cls in (PoET2Model, OpenProteinModel, ESMModel):
            stack.enter_context(
                patch.object(cls, "get_metadata", return_value=MagicMock())
            )
        assert isinstance(api.poet2, PoET2Model)
        assert api.poet2.id == "poet-2"
        assert api.prot_seq.id == "prot-seq"
        assert api.esm2.id == "esm2_t33_650M_UR50D"
        assert api.esm2 is api.esm2_t33_650M_UR50D
        assert api.esm2 is not session.models.esm2
        assert type(api.esm1b) is ESMModel
        assert type(api.esm2_t6_8M_UR50D) is ESMModel
        assert type(api.esmc_300m) is ESMModel
        assert api.esmc_300m.id == "esmc-300m"


def test_embeddings_api_get_model(mock_session: MagicMock):
    """get_model resolves hyphenated and aliased Gen1 names."""
    from contextlib import ExitStack

    from openprotein.embeddings import ESMModel, OpenProteinModel
    from openprotein.models import ModelsAPI

    session = MagicMock()
    session.models = ModelsAPI(mock_session)
    api = EmbeddingsAPI(session)

    with ExitStack() as stack:
        for cls in (OpenProteinModel, ESMModel):
            stack.enter_context(
                patch.object(cls, "get_metadata", return_value=MagicMock())
            )
        assert api.get_model("prot-seq").id == "prot-seq"
        assert api.get_model("esm2").id == "esm2_t33_650M_UR50D"
    with (
        patch.object(embeddings_api_module, "list_models", return_value=[]),
        pytest.raises(AttributeError),
    ):
        api.get_model("not-a-model")


def test_embeddings_api_resolves_listed_models_without_a_typed_handle(
    mock_session: MagicMock,
):
    """A listed model without a typed handle resolves by attribute and by name."""
    metadata = MagicMock()
    metadata.id = "prott5-xl"
    api = EmbeddingsAPI(mock_session)

    with patch.object(
        embeddings_api_module, "list_models", return_value=[metadata]
    ) as listing:
        model = api.prott5_xl
        assert type(model) is EmbeddingModel
        assert model.id == "prott5-xl"
        assert api.get_model("prott5-xl") is model
    listing.assert_called_once()


def test_embeddings_api_private_names_raise_without_a_request(
    mock_session: MagicMock,
):
    """Private and dunder names raise AttributeError without reading the listing."""
    api = EmbeddingsAPI(mock_session)

    with patch.object(embeddings_api_module, "list_models") as listing:
        assert not hasattr(api, "_repr_html_")
        assert not hasattr(api, "__deepcopy__")
    listing.assert_not_called()


def test_embeddings_api_without_a_session_raises_without_a_request():
    """An instance with no session resolves no model names."""
    api = EmbeddingsAPI.__new__(EmbeddingsAPI)

    with patch.object(embeddings_api_module, "list_models") as listing:
        assert not hasattr(api, "prott5_xl")
        assert not hasattr(api, "session")
    listing.assert_not_called()


def test_embeddings_api_model_call(mock_session: MagicMock):
    """Legacy embeddings post to the embeddings endpoint."""
    session = MagicMock()
    session.models = ModelsAPI(session)
    api = EmbeddingsAPI(session)
    session.post.return_value.json.return_value = {
        "job_id": "job-1",
        "job_type": "/embeddings/embed",
        "status": "PENDING",
        "created_date": "2026-09-05T08:00:00Z",
    }

    from openprotein.embeddings import ESMModel

    with patch.object(ESMModel, "get_metadata", return_value=MagicMock()):
        api.esm2.embed(sequences=["ACGT"])

    assert session.post.call_args.args == (
        "v1/embeddings/models/esm2_t33_650M_UR50D/embed",
    )
    assert session.models.esm2.id == "esm2_t33_650M_UR50D"


def test_embeddings_list_models_uses_verbose(mock_session: MagicMock):
    """Test that list_models calls api.list_models with verbose=True."""
    mock_metadata1 = MagicMock()
    mock_metadata1.id = "prot-seq"
    mock_metadata2 = MagicMock()
    mock_metadata2.id = "poet"

    with patch.object(
        embeddings_api_module,
        "list_models",
        return_value=[mock_metadata1, mock_metadata2],
    ) as mock_list:
        emb_api = EmbeddingsAPI(mock_session)
        result = emb_api.list_models()
        mock_list.assert_called_once_with(mock_session, verbose=True)
        assert [m.id for m in result] == ["prot-seq", "poet"]
