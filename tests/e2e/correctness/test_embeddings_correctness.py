"""Correctness tests for embeddings.

Encoders (esm2/esmc) can't be forced to recompute a fixed sequence (the cache is
keyed by sequence, with no prompt_id lever), so the invariant/relational checks
feed a freshly mutated sequence each run -> a cache miss -> a genuine recompute.
The encoder prod-baseline differential must use a fixed sequence to match the
committed baseline, so it is cache-robust (genuine on a cold cache). PoET
(poet/poet-2) cache-busts via a fresh prompt_id, so it carries the genuine
determinism, batch-invariance, and prod-baseline differential coverage.
Differential tolerances are relaxed to absorb accepted cross-environment jitter.
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
    # Callers pass a freshly mutated sequence (cache miss) for a genuine recompute;
    # the prod-baseline differential keeps a fixed sequence and is cache-robust.
    ((returned, arr),) = model.embed(sequences=[seq], reduction=reduction).wait(
        timeout=TIMEOUT
    )
    assert returned == seq
    return np.asarray(arr)


def _poet_embed(model, query: bytes, prompt) -> np.ndarray:
    # PoET cache-busts via the fresh prompt_id, so a fresh prompt already guarantees
    # a genuine recompute.
    results = model.embed(sequences=[query], prompt=prompt).wait(timeout=POET_TIMEOUT)
    return np.asarray(results[0][1])


# --------------------------------------------------------------------------- #
# Encoders (esm2/esmc): invariant / relational checks, fresh mutated sequence
# each run so every call cache-misses and genuinely recomputes.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_mean_reduction_equals_mean_of_residues(session: OpenProtein, model_id: str):
    """MEAN reduction equals the mean over the per-residue (reduction=None) embedding.

    A freshly mutated sequence each run cache-misses, so both the MEAN and the
    per-residue embeddings genuinely recompute on the current workers.
    """
    model = require_embedding_model(session, model_id)
    seq = mutate_sequence(well_known.UBIQUITIN).encode()

    per_residue = _embed_one(model, seq, None)  # shape (L', dim)
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

    A fresh wildtype (mutated ubiquitin) each run, plus a distinct point mutant and
    an unrelated random sequence, gives three distinct cache keys -> all genuinely
    recompute.
    """
    model = require_embedding_model(session, model_id)
    wt = mutate_sequence(well_known.UBIQUITIN)
    mutant = mutate_sequence(wt, mutation_rate=1.0 / len(wt))  # ~1 substitution
    unrelated = random_sequence_fake(len(wt))

    e_wt = _embed_one(model, wt.encode(), ReductionType.MEAN)
    e_mut = _embed_one(model, mutant.encode(), ReductionType.MEAN)
    e_unrel = _embed_one(model, unrelated.encode(), ReductionType.MEAN)

    assert metrics.cosine(e_wt, e_mut) > metrics.cosine(e_wt, e_unrel)


# --------------------------------------------------------------------------- #
# Encoders (esm2/esmc): prod-baseline differential.
#
# A fixed sequence is required to match the committed baseline, so this is
# cache-robust rather than a forced recompute -- genuine on a cold cache, and a
# systematic cross-worker difference still surfaces here. Genuine determinism and
# batch-invariance are carried by PoET below (its fresh prompt_id busts the cache
# on fixed content); raw-sequence encoders have no equivalent lever.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_embedding_matches_prod_baseline(
    session: OpenProtein, model_id: str, baseline
):
    """MEAN embedding of ubiquitin matches the captured prod baseline.

    Fixed sequence (to match the committed baseline), so this is cache-robust: the
    staging value is genuine on a cold cache, and a systematic cross-worker
    difference still surfaces. Tolerance absorbs accepted cross-env jitter.
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
def test_poet_embedding_matches_prod_baseline(
    session: OpenProtein, model_id: str, baseline
):
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
