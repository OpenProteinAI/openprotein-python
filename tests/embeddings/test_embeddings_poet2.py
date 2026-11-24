"""PoET-2 scoring runs the CLM decoder, so `mlm` is refused before any request."""

from unittest.mock import MagicMock

import pytest

from openprotein.embeddings.poet2 import PoET2Model
from openprotein.errors import InvalidParameterError


@pytest.fixture
def poet2(mock_session: MagicMock) -> PoET2Model:
    return PoET2Model(session=mock_session, model_id="poet-2", metadata=MagicMock())


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("score", {"sequences": [b"MQIF"]}),
        ("indel", {"sequence": b"MQIF"}),
        ("single_site", {"sequence": b"MQIF"}),
    ],
)
def test_score_family_rejects_mlm_decoder(
    poet2: PoET2Model, mock_session: MagicMock, method: str, kwargs: dict
):
    with pytest.raises(InvalidParameterError, match="cannot be `mlm` for scoring"):
        getattr(poet2, method)(decoder_type="mlm", **kwargs)
    mock_session.post.assert_not_called()
