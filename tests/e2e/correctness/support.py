"""Shared helpers for correctness E2E tests.

Cache strategy: the backend caches inference per-environment, keyed by sequence
(and, for PoET, by prompt_id/query_id). Raw-sequence encoders
(esm2/esmc) therefore can't be forced to recompute a fixed sequence. PoET/PoET-2
CAN: minting a fresh prompt/query from FIXED content yields a new id -> cache
miss -> genuine recompute, while the model stays deterministic on the *content*,
so the output still matches a committed baseline. That makes PoET the workhorse
for the genuine-every-run differential checks.
"""

from __future__ import annotations

import pytest

from openprotein import OpenProtein


def require_embedding_model(session: OpenProtein, model_id: str):
    """Return the embedding model or skip if the backend doesn't expose it."""
    available = {m.id for m in session.embedding.list_models()}
    if model_id not in available:
        pytest.skip(f"{model_id} not available in this backend")
    return session.embedding.get_model(model_id)


def fresh_prompt(session: OpenProtein, context, *, timeout: int):
    """Create a fresh prompt (new prompt_id) from FIXED explicit content.

    A new prompt_id guarantees a cache miss so the inference genuinely recomputes
    on the current workers. Because PoET conditions on prompt *content* (not the
    id), identical content yields a reproducible output comparable to a committed
    baseline. Always use this (explicit content), never ``sample_prompt`` (which
    draws different context sequences each call and is therefore non-reproducible).
    """
    prompt = session.prompt.create_prompt(context)
    assert prompt.wait_until_done(timeout=timeout), "prompt creation did not complete"
    return prompt


def fresh_query(session: OpenProtein, sequence):
    """Create a fresh query (new query_id) from a FIXED sequence.

    Same rationale as ``fresh_prompt`` but for PoET-2's ``query`` lever: a new
    query_id busts the cache while the query content stays fixed.
    """
    return session.prompt.create_query(sequence)
