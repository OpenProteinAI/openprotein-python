import json
import os
import stat
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from openprotein.auth_store import TokenRecord, TokenStore
from openprotein.base import APISession, BearerAuth
from openprotein.errors import AuthError, HTTPError


@pytest.fixture
def store(tmp_path):
    return TokenStore(dir=tmp_path)


def test_identity_format(store):
    assert (
        store.identity("https://api.openprotein.ai/api/", "user@x.com")
        == "https://api.openprotein.ai/api/|user@x.com"
    )


def test_save_then_load_round_trips(store):
    identity = store.identity("https://backend/", "user@x.com")
    record = TokenRecord(access_token="a", refresh_token="r", expires_at=1234.0)

    store.save(identity, record)
    loaded = store.load(identity)

    assert loaded == record


def test_missing_file_returns_none(store):
    assert store.load("nonexistent|identity") is None


def test_token_file_mode_is_0600(store, tmp_path):
    store.save("id|x", TokenRecord(access_token="a", refresh_token="r", expires_at=1.0))
    mode = stat.S_IMODE(os.stat(tmp_path / "token.json").st_mode)
    assert mode == 0o600


def test_corrupt_file_treated_as_no_record(store, tmp_path):
    (tmp_path / "token.json").write_text("{ this is not json")
    assert store.load("anything") is None


def test_identity_scoping_is_independent(store):
    dev = store.identity("https://dev/", "user@x.com")
    prod = store.identity("https://prod/", "user@x.com")
    store.save(
        dev, TokenRecord(access_token="dev-a", refresh_token="dev-r", expires_at=1.0)
    )
    store.save(
        prod, TokenRecord(access_token="prod-a", refresh_token="prod-r", expires_at=2.0)
    )

    assert store.load(dev).access_token == "dev-a"
    assert store.load(prod).access_token == "prod-a"


def test_save_preserves_other_identities(store, tmp_path):
    a = store.identity("https://a/", "u")
    b = store.identity("https://b/", "u")
    store.save(a, TokenRecord(access_token="a", refresh_token="ra", expires_at=1.0))
    store.save(b, TokenRecord(access_token="b", refresh_token="rb", expires_at=2.0))

    data = json.loads((tmp_path / "token.json").read_text())
    assert set(data.keys()) == {a, b}


def test_lock_is_reentrant(store):
    # Acquiring the same store lock twice in one process must not deadlock.
    with store.lock():
        with store.lock():
            pass


# ---------------------------------------------------------------------------
# APISession login behavior tests
# ---------------------------------------------------------------------------

BACKEND = "https://test-backend/"


def _resp(status=200, json_body=None):
    """Build a ResponseMock-like object for patching APISession.post / request."""
    mock = MagicMock()
    mock.status_code = status
    mock.ok = status < 400
    mock.json.return_value = json_body or {}
    mock.headers = {}
    mock.text = ""
    return mock


def _http_error(status):
    resp = MagicMock()
    resp.status_code = status
    resp.text = ""
    resp.url = "u"
    return HTTPError(resp)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point APISession's token store at a temp dir, returning that dir."""
    from openprotein import auth_store

    monkeypatch.setattr(
        "openprotein.base.TokenStore",
        lambda *a, **k: auth_store.TokenStore(dir=tmp_path),
    )
    return tmp_path


def test_login_uses_new_endpoint_with_delivery_header(isolated_store):
    login_resp = _resp(
        json_body={
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 900,
        }
    )
    with patch.object(APISession, "post", return_value=login_resp) as post:
        session = APISession("user@x.com", "pw", backend=BACKEND)

    post.assert_called_once_with(
        f"{BACKEND}v1/auth/login",
        data={"username": "user@x.com", "password": "pw"},
        headers={"X-Token-Delivery": "body"},
        timeout=3,
    )
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "access-1"
    assert session._refresh_token == "refresh-1"
    assert session._legacy_mode is False


def test_login_persists_refresh_token_to_store(isolated_store):
    from openprotein.auth_store import TokenStore

    login_resp = _resp(
        json_body={
            "access_token": "access-1",
            "refresh_token": "refresh-1",
            "expires_in": 900,
        }
    )
    with patch.object(APISession, "post", return_value=login_resp):
        APISession("user@x.com", "pw", backend=BACKEND)

    record = TokenStore(dir=isolated_store).load(f"{BACKEND}|user@x.com")
    assert record is not None
    assert record.refresh_token == "refresh-1"
    assert record.access_token == "access-1"


def test_login_without_refresh_token_is_legacy_mode(isolated_store):
    login_resp = _resp(json_body={"access_token": "access-only"})
    with patch.object(APISession, "post", return_value=login_resp):
        session = APISession("user@x.com", "pw", backend=BACKEND)

    assert session._legacy_mode is True
    assert session._refresh_token is None
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "access-only"


def test_login_404_falls_back_to_legacy_endpoint(isolated_store):
    # First call (v1/auth/login) raises 404, second (legacy) succeeds.
    legacy_resp = _resp(json_body={"access_token": "legacy-access"})
    with patch.object(
        APISession,
        "post",
        side_effect=[_http_error(404), legacy_resp],
    ) as post:
        session = APISession("user@x.com", "pw", backend=BACKEND)

    assert session._legacy_mode is True
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "legacy-access"
    assert post.call_args_list[0].args[0] == f"{BACKEND}v1/auth/login"
    assert post.call_args_list[1].args[0] == f"{BACKEND}v1/login/access-token"


def test_login_missing_access_token_raises_autherror(isolated_store):
    with patch.object(APISession, "post", return_value=_resp(json_body={})):
        with pytest.raises(AuthError, match="Unable to authenticate"):
            APISession("user@x.com", "pw", backend=BACKEND)


def test_login_non404_error_raises_autherror(isolated_store):
    # The other arm of _get_auth_token's branch: any non-404 error from the
    # new endpoint is a real auth failure, not a signal to try the legacy one.
    with patch.object(APISession, "post", side_effect=_http_error(500)):
        with pytest.raises(AuthError, match="Authentication failed"):
            APISession("user@x.com", "pw", backend=BACKEND)


# ---------------------------------------------------------------------------
# Session init adoption tests
# ---------------------------------------------------------------------------


def test_init_adopts_stored_record_without_login(isolated_store):
    # Seed the store with a valid record for this identity.
    TokenStore(dir=isolated_store).save(
        f"{BACKEND}|user@x.com",
        TokenRecord(
            access_token="stored-access",
            refresh_token="stored-refresh",
            expires_at=time.time() + 10_000,  # far from expiry
        ),
    )

    with patch.object(APISession, "post") as post:
        session = APISession("user@x.com", "pw", backend=BACKEND)

    post.assert_not_called()
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "stored-access"
    assert session._refresh_token == "stored-refresh"
    assert session._legacy_mode is False


def test_init_logs_in_when_stored_record_has_no_refresh_token(isolated_store):
    # A record without a refresh token must NOT be adopted (it's unusable).
    TokenStore(dir=isolated_store).save(
        f"{BACKEND}|user@x.com",
        TokenRecord(access_token="stale", refresh_token=None, expires_at=1.0),
    )
    login_resp = _resp(
        json_body={
            "access_token": "fresh-access",
            "refresh_token": "fresh-refresh",
            "expires_in": 900,
        }
    )
    with patch.object(APISession, "post", return_value=login_resp) as post:
        session = APISession("user@x.com", "pw", backend=BACKEND)

    post.assert_called_once()
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "fresh-access"


# ---------------------------------------------------------------------------
# _refresh / _relogin tests
# ---------------------------------------------------------------------------


def _logged_in_session(
    isolated_store, access="access-1", refresh="refresh-1", expires_in=900
):
    login_resp = _resp(
        json_body={
            "access_token": access,
            "refresh_token": refresh,
            "expires_in": expires_in,
        }
    )
    with patch.object(APISession, "post", return_value=login_resp):
        return APISession("user@x.com", "pw", backend=BACKEND)


def test_refresh_rotates_token_and_rewrites_store(isolated_store):
    session = _logged_in_session(isolated_store)
    rotated = _resp(
        json_body={
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 900,
        }
    )
    with patch.object(APISession, "post", return_value=rotated) as post:
        session._refresh()

    post.assert_called_once_with(
        f"{BACKEND}v1/auth/refresh",
        json={"refresh_token": "refresh-1"},
        headers={"X-Token-Delivery": "body"},
        timeout=10,
    )
    assert session._refresh_token == "refresh-2"
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "access-2"
    record = TokenStore(dir=isolated_store).load(f"{BACKEND}|user@x.com")
    assert record is not None
    assert record.refresh_token == "refresh-2"


def test_refresh_adopts_fresher_token_from_store(isolated_store):
    session = _logged_in_session(isolated_store)
    # Simulate another process having refreshed: store now holds a fresher record.
    session._expires_at = time.time() + 1  # make ours look stale relative to store
    TokenStore(dir=isolated_store).save(
        f"{BACKEND}|user@x.com",
        TokenRecord(
            access_token="other-access",
            refresh_token="other-refresh",
            expires_at=session._expires_at + 10_000,
        ),
    )

    with patch.object(APISession, "post") as post:
        session._refresh()

    post.assert_not_called()  # adopted the fresher token instead of refreshing
    assert session._refresh_token == "other-refresh"
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "other-access"


def test_refresh_401_triggers_relogin(isolated_store):
    session = _logged_in_session(isolated_store)
    relogin_resp = _resp(
        json_body={
            "access_token": "relogin-access",
            "refresh_token": "relogin-refresh",
            "expires_in": 900,
        }
    )
    # First post (the refresh) raises 401; second post (re-login) succeeds.
    with patch.object(
        APISession, "post", side_effect=[_http_error(401), relogin_resp]
    ) as post:
        session._refresh()

    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "relogin-access"
    assert session._refresh_token == "relogin-refresh"
    assert post.call_args_list[0].args[0] == f"{BACKEND}v1/auth/refresh"
    assert post.call_args_list[1].args[0] == f"{BACKEND}v1/auth/login"
    # The re-login's new token is persisted, same as a normal login/refresh.
    record = TokenStore(dir=isolated_store).load(f"{BACKEND}|user@x.com")
    assert record is not None
    assert record.refresh_token == "relogin-refresh"


def test_refresh_relogin_failure_raises_autherror(isolated_store):
    session = _logged_in_session(isolated_store)
    with patch.object(
        APISession, "post", side_effect=[_http_error(401), _http_error(401)]
    ):
        with pytest.raises(AuthError):
            session._refresh()


def test_refresh_403_triggers_relogin(isolated_store):
    # Some Keycloak deployments return 403 (not 401) for an expired session.
    session = _logged_in_session(isolated_store)
    relogin_resp = _resp(
        json_body={
            "access_token": "relogin-access",
            "refresh_token": "relogin-refresh",
            "expires_in": 900,
        }
    )
    with patch.object(
        APISession, "post", side_effect=[_http_error(403), relogin_resp]
    ) as post:
        session._refresh()

    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "relogin-access"
    assert post.call_args_list[0].args[0] == f"{BACKEND}v1/auth/refresh"
    assert post.call_args_list[1].args[0] == f"{BACKEND}v1/auth/login"


def test_refresh_404_triggers_relogin(isolated_store):
    # A backend that issued a refresh token but no longer exposes /auth/refresh
    # (e.g. mid-rollback) returns 404; recover by re-logging-in rather than
    # surfacing a raw 404. Re-login itself drops to legacy mode if needed.
    session = _logged_in_session(isolated_store)
    relogin_resp = _resp(
        json_body={
            "access_token": "relogin-access",
            "refresh_token": "relogin-refresh",
            "expires_in": 900,
        }
    )
    with patch.object(
        APISession, "post", side_effect=[_http_error(404), relogin_resp]
    ) as post:
        session._refresh()

    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "relogin-access"
    assert post.call_args_list[0].args[0] == f"{BACKEND}v1/auth/refresh"
    assert post.call_args_list[1].args[0] == f"{BACKEND}v1/auth/login"


def test_refresh_non_auth_error_propagates(isolated_store):
    # An error outside the recoverable set (401/403/404) from the refresh
    # endpoint is a real failure and must propagate, not trigger re-login.
    session = _logged_in_session(isolated_store)
    with patch.object(APISession, "post", side_effect=_http_error(500)):
        with pytest.raises(HTTPError):
            session._refresh()


def test_refresh_without_refresh_token_falls_back_to_relogin(isolated_store):
    # No refresh token in memory (e.g. a degenerate state) -> re-login instead.
    session = _logged_in_session(isolated_store)
    session._refresh_token = None
    relogin_resp = _resp(
        json_body={
            "access_token": "relogin-access",
            "refresh_token": "relogin-refresh",
            "expires_in": 900,
        }
    )
    with patch.object(APISession, "post", return_value=relogin_resp) as post:
        session._refresh()

    post.assert_called_once()
    assert post.call_args.args[0] == f"{BACKEND}v1/auth/login"
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "relogin-access"


# ---------------------------------------------------------------------------
# request() proactive + reactive refresh tests
# ---------------------------------------------------------------------------


def test_proactive_refresh_fires_within_skew(isolated_store):
    # Log in with a token already within SKEW of expiry, so the in-memory expiry
    # AND the stored record agree it is near-expired. (If only the in-memory
    # expiry were lowered, _refresh would see the still-far stored record as
    # "fresher" and adopt it instead of doing a network refresh.)
    session = _logged_in_session(isolated_store, expires_in=5)

    rotated = _resp(
        json_body={
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 900,
        }
    )
    ok = _resp(status=200, json_body={"ok": True})
    # super().request is used for the proactive refresh POST, then the GET.
    with patch.object(requests.Session, "request", side_effect=[rotated, ok]) as sreq:
        session.request("GET", "v1/some/endpoint")

    # Refresh POST then the downstream GET — and nothing more.
    assert sreq.call_count == 2
    assert sreq.call_args_list[0].args[1] == f"{BACKEND}v1/auth/refresh"
    assert sreq.call_args_list[1].args[1] == f"{BACKEND}v1/some/endpoint"
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "access-2"


def test_no_proactive_refresh_when_far_from_expiry(isolated_store):
    session = _logged_in_session(isolated_store)
    session._expires_at = time.time() + 10_000  # far from expiry

    ok = _resp(status=200, json_body={"ok": True})
    with patch.object(requests.Session, "request", return_value=ok) as sreq:
        session.request("GET", "v1/some/endpoint")

    # Only the GET happened, no refresh.
    assert sreq.call_count == 1
    assert sreq.call_args_list[0].args[1] == f"{BACKEND}v1/some/endpoint"


def test_401_triggers_refresh_and_single_retry(isolated_store):
    session = _logged_in_session(isolated_store)
    session._expires_at = time.time() + 10_000  # disable proactive path

    first = _resp(status=401)
    refresh = _resp(
        json_body={
            "access_token": "access-2",
            "refresh_token": "refresh-2",
            "expires_in": 900,
        }
    )
    retry_ok = _resp(status=200, json_body={"ok": True})
    with patch.object(
        requests.Session, "request", side_effect=[first, refresh, retry_ok]
    ) as sreq:
        resp = session.request("GET", "v1/some/endpoint")

    assert resp is retry_ok
    # GET (401) -> refresh POST -> GET retry (200): exactly three underlying calls.
    assert sreq.call_count == 3
    assert sreq.call_args_list[1].args[1] == f"{BACKEND}v1/auth/refresh"


def test_legacy_mode_does_not_refresh_on_401(isolated_store):
    session = _logged_in_session(isolated_store)
    session._legacy_mode = True

    unauth = _resp(status=401)
    with patch.object(requests.Session, "request", return_value=unauth):
        with pytest.raises(HTTPError):
            session.request("GET", "v1/some/endpoint")


def test_refreshing_guard_is_thread_local(isolated_store):
    # The recursion guard must not leak across threads: while one thread is
    # mid-refresh, OTHER threads sharing the session (e.g. via stream_parallel)
    # must still be allowed to refresh, or they'd send stale tokens and fail.
    session = _logged_in_session(isolated_store)
    session._expires_at = time.time() - 1  # due for proactive refresh

    session._refreshing = True  # this thread is mid-refresh
    assert session._should_refresh_proactively() is False  # suppressed here

    results = {}

    def other_thread():
        results["proactive"] = session._should_refresh_proactively()

    t = threading.Thread(target=other_thread)
    t.start()
    t.join()
    assert results["proactive"] is True  # not suppressed in another thread


def test_init_adopts_expired_record_and_refreshes_lazily(isolated_store):
    # A stored record past its expiry is still adopted on init (no eager
    # refresh / re-login); the next request's proactive check refreshes it.
    TokenStore(dir=isolated_store).save(
        f"{BACKEND}|user@x.com",
        TokenRecord(
            access_token="old-access",
            refresh_token="old-refresh",
            expires_at=time.time() - 1,  # already expired
        ),
    )
    with patch.object(APISession, "post") as post:
        session = APISession("user@x.com", "pw", backend=BACKEND)

    post.assert_not_called()  # adopted, not re-logged-in
    assert isinstance(session.auth, BearerAuth)
    assert session.auth._token == "old-access"
    assert session._should_refresh_proactively() is True
