"""Wire calls for stored prompts (``session.data.Prompt``)."""

import secrets
from typing import Any

from pydantic import BaseModel

from openprotein.align.schemas import MSASamplingMethod
from openprotein.base import APISession
from openprotein.data._reads import read
from openprotein.errors import MissingParameterError

from .schemas import PromptMetadata, PromptRecord


class PromptPage(BaseModel):
    """One page of prompt summaries, newest first."""

    items: list[PromptMetadata]
    next: str | None = None


PATH_PREFIX = "v1/data/prompts"


def prompts_list(
    session: APISession, after: str | None = None, limit: int | None = None
) -> PromptPage:
    """List the caller's live prompts, newest first."""
    params: dict[str, Any] = {}
    if after is not None:
        params["after"] = after
    if limit is not None:
        params["limit"] = limit
    response = session.get(PATH_PREFIX, params=params or None)
    return PromptPage.model_validate(response.json())


def prompts_get(session: APISession, prompt_id: str) -> PromptRecord:
    """Get one prompt.

    A prompt whose job hasn't finished comes back with that job's ``status``.
    A deleted or unknown id is a 404.
    """
    return PromptRecord.model_validate(
        read(session, f"{PATH_PREFIX}/{prompt_id}").json()
    )


def prompts_put(
    session: APISession,
    prompt_id: str,
    name: str | None = None,
    description: str | None = None,
) -> PromptRecord:
    """Update name/description; only the given fields are sent."""
    body: dict[str, str] = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    response = session.patch(f"{PATH_PREFIX}/{prompt_id}", json=body)
    return PromptRecord.model_validate(response.json())


def prompts_delete(session: APISession, prompt_id: str) -> None:
    """Delete the prompt."""
    session.delete(f"{PATH_PREFIX}/{prompt_id}")


def prompts_post(
    session: APISession,
    replicates: list[bytes],
    name: str | None = None,
    description: str | None = None,
) -> PromptRecord:
    """Upload a prompt, one FASTA file per replicate, in position order."""
    data: dict[str, str] = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    files = [
        ("replicates", (f"replicate-{position}.fasta", content))
        for position, content in enumerate(replicates)
    ]
    response = session.post(PATH_PREFIX, data=data, files=files)
    return PromptRecord.model_validate(response.json())


def prompts_download(session: APISession, prompt_id: str) -> bytes:
    """Download the zip of replicate FASTA files.

    A prompt that isn't ready raises the server's 409; a deleted or unknown id
    is a 404.
    """
    return read(session, f"{PATH_PREFIX}/{prompt_id}/download").content


def prompts_sample_post(
    session: APISession,
    msa_id: str,
    replicates: int = 1,
    seed: int | None = None,
    method: MSASamplingMethod | None = None,
    homology_level: float | None = None,
    max_similarity: float | None = None,
    min_similarity: float | None = None,
    max_msa_sequences: int | None = None,
    max_msa_tokens: int | None = None,
    force_include_first: bool | None = None,
) -> dict[str, Any]:
    """Submit one alignment for sampling into prompt replicates.

    A missing ``seed`` is drawn here. ``replicates`` is always sent. Every
    other option is omitted when unset so the server's default applies. The
    request carries no idempotency key, so it is sent once and never retried.

    Returns
    -------
    dict
        The sampling's job, whose ``job_id`` is the prompt's id.

    Raises
    ------
    MissingParameterError
        If both ``max_msa_sequences`` and ``max_msa_tokens`` are given.
    """
    if max_msa_sequences is not None and max_msa_tokens is not None:
        raise MissingParameterError(
            "Either 'max_msa_sequences' or 'max_msa_tokens' may be set, but not both."
        )

    body: dict[str, Any] = {
        "msa_id": msa_id,
        "seed": secrets.randbits(32) if seed is None else seed,
        "replicates": replicates,
    }
    optional: dict[str, Any] = {
        "method": None if method is None else MSASamplingMethod(method).value,
        "homology_level": homology_level,
        "max_similarity": max_similarity,
        "min_similarity": min_similarity,
        "max_msa_sequences": max_msa_sequences,
        "max_msa_tokens": max_msa_tokens,
        "force_include_first": force_include_first,
    }
    body.update({k: v for k, v in optional.items() if v is not None})

    response = session.post(f"{PATH_PREFIX}/sample_msa", json=body)
    return response.json()
