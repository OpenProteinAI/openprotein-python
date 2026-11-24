"""Compatibility session namespaces emit no runtime deprecation warnings."""

import time
import warnings
from unittest.mock import MagicMock, patch

import pytest

from openprotein import OpenProtein
from openprotein.base import APISession


@pytest.fixture
def session(tmp_path, monkeypatch):
    from openprotein import auth_store

    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )
    monkeypatch.setattr("openprotein.fold.api.fold_models_list_get", lambda session: [])
    login = MagicMock(status_code=200, ok=True, headers={})
    login.json.return_value = {
        "access_token": "access",
        "refresh_token": "refresh",
        "expires_in": 900,
    }
    with patch.object(APISession, "post", return_value=login):
        session = OpenProtein("user@x.com", "pw", backend="https://api.example.test/")
    session._expires_at = time.time() + 10_000
    return session


@pytest.mark.parametrize(
    "namespace", ["embedding", "embeddings", "fold", "predictor", "svd", "umap"]
)
def test_compatibility_namespaces_emit_no_warnings(session, namespace):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(session, namespace)
