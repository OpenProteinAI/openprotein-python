"""Correctness tests for embeddings.

Every check genuinely recomputes (no stale cache): encoders (esm2/esmc) pass
force_recompute=True (#235) to bypass the per-sequence cache; PoET (poet/poet-2)
cache-busts via a fresh prompt_id (force_recompute is not wired into PoET2Model).
Both get genuine determinism, batch-invariance, and prod-baseline differentials;
reduction-math and semantic similarity are cache-robust extras. Differential
tolerances are relaxed to absorb accepted cross-environment jitter.
"""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import metrics, well_known
from tests.e2e.correctness.support import (
    fresh_prompt,
    require_embedding_model,
)
from tests.utils.sequences import mutate_sequence, random_sequence_fake

TIMEOUT = scaled_timeout(1.0)
# PoET ops are heavier/slower than encoder embeds on prod; give them more headroom.
POET_TIMEOUT = scaled_timeout(3.0)
ENCODER_MODELS = ["esm2_t33_650M_UR50D", "esmc-300m"]
POET_MODELS = ["poet", "poet-2"]


def _embed_one(model, seq: bytes, reduction):
    # force_recompute=True bypasses the per-sequence cache so every correctness check
    # genuinely recomputes on the current workers (no stale cached value).
    (returned, arr), = model.embed(
        sequences=[seq], reduction=reduction, force_recompute=True
    ).wait(timeout=TIMEOUT)
    assert returned == seq
    return np.asarray(arr)


def _poet_embed(model, query: bytes, prompt) -> np.ndarray:
    # PoET cache-busts via the fresh prompt_id (force_recompute is not wired into
    # PoET2Model), so a fresh prompt already guarantees a genuine recompute.
    results = model.embed(sequences=[query], prompt=prompt).wait(timeout=POET_TIMEOUT)
    return np.asarray(results[0][1])


# --------------------------------------------------------------------------- #
# Encoders (esm2/esmc): cache-robust checks only.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_mean_reduction_equals_mean_of_residues(session: OpenProtein, model_id: str):
    """MEAN reduction equals the mean over the per-residue (reduction=None) embedding.

    Cache-robust: MEAN and per-residue are distinct output types (distinct cache
    keys), so the relationship is genuinely computed, not served from one entry.
    """
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()

    per_residue = _embed_one(model, seq, None)            # shape (L', dim)
    mean_reduced = _embed_one(model, seq, ReductionType.MEAN)  # shape (dim,)

    assert per_residue.ndim == 2
    recomputed = per_residue.mean(axis=0)
    # Looser tol: the server may exclude special tokens from the mean. If this
    # fails, inspect whether per-residue includes BOS/EOS and adjust the axis-0
    # slice accordingly (documented follow-up, not a silent pass).
    metrics.assert_close(mean_reduced, recomputed, rtol=1e-2, atol=1e-2)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_embedding_semantic_similarity(session: OpenProtein, model_id: str):
    """A point mutant is closer to wildtype than an unrelated random sequence.

    Cache-robust: three distinct sequences -> three distinct cache keys.
    """
    model = require_embedding_model(session, model_id)
    wt = well_known.UBIQUITIN
    mutant = mutate_sequence(wt, mutation_rate=1.0 / len(wt))  # ~1 substitution
    unrelated = random_sequence_fake(len(wt))

    e_wt = _embed_one(model, wt.encode(), ReductionType.MEAN)
    e_mut = _embed_one(model, mutant.encode(), ReductionType.MEAN)
    e_unrel = _embed_one(model, unrelated.encode(), ReductionType.MEAN)

    assert metrics.cosine(e_wt, e_mut) > metrics.cosine(e_wt, e_unrel)


# --------------------------------------------------------------------------- #
# Encoders (esm2/esmc): genuine via force_recompute (bypasses the per-sequence cache,
# so determinism/batch-invariance/value differentials are real, not cache-served).
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_embedding_deterministic(session: OpenProtein, model_id: str):
    """Two force_recompute embeds of the same sequence are identical (genuine, not cache-served)."""
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()
    a = _embed_one(model, seq, ReductionType.MEAN)
    b = _embed_one(model, seq, ReductionType.MEAN)
    metrics.assert_close(a, b, rtol=1e-3, atol=1e-3)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_embedding_batch_invariance(session: OpenProtein, model_id: str):
    """A sequence's embedding is the same alone vs inside a batch (stresses rebatching).

    force_recompute on both calls means neither is cache-served, so this genuinely
    exercises the ModelServer rebatching path for encoders.
    """
    model = require_embedding_model(session, model_id)
    target = well_known.UBIQUITIN.encode()
    others = [random_sequence_fake(120).encode() for _ in range(4)]
    alone = _embed_one(model, target, ReductionType.MEAN)
    batch = model.embed(
        sequences=[others[0], others[1], target, others[2], others[3]],
        reduction=ReductionType.MEAN,
        force_recompute=True,
    ).wait(timeout=TIMEOUT)
    by_seq = {seq: np.asarray(arr) for seq, arr in batch}
    # Loosened from 1e-3: fp16 batched matmul introduces ~1e-3 jitter vs the single
    # forward (esmc), which is numerical, not a real batch-position effect.
    metrics.assert_close(by_seq[target], alone, rtol=5e-3, atol=5e-3)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_embedding_matches_prod_baseline(session: OpenProtein, model_id: str, baseline):
    """MEAN embedding of ubiquitin matches the captured prod baseline.

    Reliable now: force_recompute bypasses the cache so staging genuinely recomputes
    (the earlier false "trim" mismatch was a stale cached value). Tolerance absorbs
    accepted cross-env jitter.
    """
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()

    def produce():
        return _embed_one(model, seq, ReductionType.MEAN)

    baseline.check_array(
        (model_id, None, "ubiquitin", "embed_mean"), produce, atol=0.05, rtol=0.05
    )


# --------------------------------------------------------------------------- #
# PoET (poet/poet-2): genuine recompute every run via fresh prompt_id.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_embedding_deterministic(session: OpenProtein, model_id: str):
    """Two fresh prompts over identical content yield identical embeddings.

    Distinct prompt_ids -> both calls cache-miss and genuinely recompute; PoET is
    deterministic on prompt *content*, so the outputs must match. This is a real
    determinism check (unlike the cache-served encoder case).
    """
    model = require_embedding_model(session, model_id)
    query = well_known.POET_QUERY.encode()
    p1 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
    p2 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)

    a = _poet_embed(model, query, p1)
    b = _poet_embed(model, query, p2)
    metrics.assert_close(a, b, rtol=1e-3, atol=1e-3)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_embedding_batch_invariance(session: OpenProtein, model_id: str):
    """A query's embedding is the same alone or inside a batch (stresses rebatching).

    Uses a DIFFERENT fresh prompt for the alone vs batch call so both genuinely
    recompute -- with the same (prompt_id, sequence) the batch call would hit the
    cache entry written by the alone call and pass vacuously. Determinism (above)
    establishes prompt-id/content invariance, so any difference here isolates a
    batch-position effect.
    """
    model = require_embedding_model(session, model_id)
    target = well_known.POET_QUERY.encode()
    others = [random_sequence_fake(80).encode() for _ in range(4)]

    p_alone = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
    p_batch = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)

    alone = _poet_embed(model, target, p_alone)
    batch = model.embed(
        sequences=[others[0], others[1], target, others[2], others[3]], prompt=p_batch
    ).wait(timeout=POET_TIMEOUT)
    by_seq = {seq: np.asarray(arr) for seq, arr in batch}

    metrics.assert_close(by_seq[target], alone, rtol=1e-3, atol=1e-3)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_embedding_matches_prod_baseline(session: OpenProtein, model_id: str, baseline):
    """PoET embedding of the fixed query matches the prod baseline -- genuine every run.

    Each call mints a fresh prompt_id (cache miss), so both the prod capture and the
    staging assert truly recompute on their respective workers.
    """
    model = require_embedding_model(session, model_id)
    query = well_known.POET_QUERY.encode()

    def produce():
        prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
        return _poet_embed(model, query, prompt)

    # PoET genuinely recomputes (fresh prompt) on both envs; accept small cross-env
    # numerical jitter (prod vs new workers) per the accepted-jitter decision.
    baseline.check_array(
        (model_id, None, "ubiquitin@poetctx", "embed"), produce, atol=0.05, rtol=0.05
    )
