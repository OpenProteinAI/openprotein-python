"""Wire calls for the models API."""

from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from openprotein.base import _IDEMPOTENCY_KEY_HEADER, APISession
from openprotein.data._reads import read
from openprotein.jobs import Job
from openprotein.jobs._submission import post_with_retry

from .schemas import ModelMetadata

PATH_PREFIX = "v1/models"
JOBS_PATH_PREFIX = "v1/jobs"
_LIST_PAGE = 1000


def list_models(
    session: APISession,
    verbose: bool = False,
    *,
    path: list[str] | None = None,
) -> list[str] | list[ModelMetadata]:
    """List system model ids, or full metadata with ``verbose``, narrowed by capability.

    A ``path`` value matches any method path containing it, case-insensitively.
    """
    params: dict[str, Any] = {"scope": "system", "limit": _LIST_PAGE}
    if verbose:
        params["verbose"] = "true"
    if path:
        params["path"] = path
    result: list[Any] = []
    while True:
        page = session.get(PATH_PREFIX, params={**params, "offset": len(result)}).json()
        result += page
        if len(page) < _LIST_PAGE:
            break
    if verbose:
        return TypeAdapter(list[ModelMetadata]).validate_python(result)
    return result


def get_model(session: APISession, model_id: str) -> ModelMetadata:
    response = session.get(f"{PATH_PREFIX}/{model_id}", timeout=session.timeout)
    return ModelMetadata.model_validate(response.json())


def get_method_params(session: APISession, model_id: str, path: str) -> dict[str, Any]:
    """Return a method's request params as a JSON Schema."""
    response = session.get(f"{PATH_PREFIX}/{model_id}/{path}/params")
    return response.json()


def run_method(
    session: APISession,
    model_id: str,
    path: str,
    params: dict[str, Any],
) -> Job:
    """Submit one method call, returning the job.

    The body is the params the method's schema describes.
    """
    response = post_with_retry(
        session,
        f"{PATH_PREFIX}/{model_id}/{path}",
        json=params,
        headers={_IDEMPOTENCY_KEY_HEADER: str(uuid4())},
    )
    return Job.create(response.json())


def bare_id(family: str, model_id: str) -> str:
    """The fitted model's id without its ``family/`` prefix."""
    return model_id.removeprefix(f"{family}/")


def fitted_path(family: str, model_id: str | None = None) -> str:
    """The family's route, or one fitted model's, from its bare or prefixed id."""
    route = f"{PATH_PREFIX}/{family}"
    return route if model_id is None else f"{route}/{bare_id(family, model_id)}"


def create_fit(
    session: APISession, family: str, request: dict[str, Any]
) -> dict[str, Any]:
    """Submit a fit; returns the fit job, or a body naming only its ``id``."""
    response = post_with_retry(
        session,
        fitted_path(family),
        json=request,
        headers={_IDEMPOTENCY_KEY_HEADER: str(uuid4())},
    )
    return response.json()


def list_fitted(
    session: APISession,
    family: str,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict[str, Any]]:
    """The caller's fitted models of ``family`` as served, newest first.

    Every page is read, up to ``limit`` models, starting at ``offset``.
    """
    params: dict[str, Any] = {"scope": "mine", "type": family, "verbose": "true"}
    start = offset or 0
    cards: list[dict[str, Any]] = []
    while limit is None or len(cards) < limit:
        size = _LIST_PAGE if limit is None else min(_LIST_PAGE, limit - len(cards))
        page = session.get(
            PATH_PREFIX,
            params={**params, "limit": size, "offset": start + len(cards)},
        ).json()
        cards += page
        if len(page) < size:
            break
    return cards


def get_fitted(session: APISession, family: str, model_id: str) -> dict[str, Any]:
    """The fitted model's card as served; raises a 404 unless its fit succeeded."""
    return read(session, fitted_path(family, model_id)).json()


def delete_fitted(session: APISession, family: str, model_id: str) -> None:
    session.delete(fitted_path(family, model_id))
