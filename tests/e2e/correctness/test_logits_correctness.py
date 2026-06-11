"""Correctness tests for logits.

Every check genuinely recomputes: encoders pass force_recompute=True (#235) to bypass
the per-sequence cache; PoET cache-busts via a fresh prompt_id. Encoder argmax tokens
are compared exactly; raw logit values use a relaxed tolerance for accepted cross-env
jitter.
"""

import numpy as np
import pytest

from openprotein import OpenProtein
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import metrics, well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model

TIMEOUT = scaled_timeout(1.0)
# PoET logits are heavier/slower than embeddings on prod; give them more headroom.
POET_TIMEOUT = scaled_timeout(3.0)
ENCODER_MODELS = ["esm2_t33_650M_UR50D", "esmc-300m"]
POET_MODELS = ["poet", "poet-2"]


def _logits_one(model, seq: bytes) -> np.ndarray:
    # force_recompute=True bypasses the per-sequence cache (genuine recompute).
    (returned, arr), = model.logits(
        sequences=[seq], force_recompute=True
    ).wait(timeout=TIMEOUT)
    assert returned == seq
    return np.asarray(arr)


def _poet_logits(model, query: bytes, prompt) -> np.ndarray:
    # PoET cache-busts via the fresh prompt_id (force_recompute is not wired into
    # PoET2Model), so a fresh prompt already guarantees a genuine recompute.
    results = model.logits(sequences=[query], prompt=prompt).wait(timeout=POET_TIMEOUT)
    return np.asarray(results[0][1])


# --------------------------------------------------------------------------- #
# Encoders: structural checks + cold-cache baselines.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_logits_shape_and_finite(session: OpenProtein, model_id: str):
    """Logits are finite and 2-D with one row per position (+ specials)."""
    model = require_embedding_model(session, model_id)
    arr = _logits_one(model, well_known.UBIQUITIN.encode())
    assert arr.ndim == 2
    assert arr.shape[0] >= len(well_known.UBIQUITIN)
    assert np.isfinite(arr).all()


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_logits_argmax_tokens_match_prod(session: OpenProtein, model_id: str, baseline):
    """Per-position argmax token indices match the prod baseline exactly.

    Reliable via force_recompute (the cache previously served a stale trimmed length).
    """
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()

    def produce():
        return np.argmax(_logits_one(model, seq), axis=-1)

    baseline.check_tokens((model_id, None, "ubiquitin", "logits_argmax"), produce)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_logits_values_match_prod(session: OpenProtein, model_id: str, baseline):
    """Raw logit values match the prod baseline within accepted cross-env jitter."""
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()

    def produce():
        return _logits_one(model, seq)

    baseline.check_array(
        (model_id, None, "ubiquitin", "logits"), produce, atol=0.1, rtol=0.05
    )


# --------------------------------------------------------------------------- #
# PoET: genuine determinism + baselines via fresh prompts.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_logits_deterministic(session: OpenProtein, model_id: str):
    """Two fresh prompts over identical content yield identical logits (genuine recompute)."""
    model = require_embedding_model(session, model_id)
    query = well_known.POET_QUERY.encode()
    p1 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
    p2 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
    a = _poet_logits(model, query, p1)
    b = _poet_logits(model, query, p2)
    assert np.isfinite(a).all()
    metrics.assert_close(a, b, rtol=1e-3, atol=1e-3)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_logits_match_prod_baseline(session: OpenProtein, model_id: str, baseline):
    """PoET logits for the fixed query match the prod baseline -- genuine every run."""
    model = require_embedding_model(session, model_id)
    query = well_known.POET_QUERY.encode()

    def produce():
        prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
        return _poet_logits(model, query, prompt)

    # Genuine recompute (fresh prompt); accept small cross-env jitter on logit values.
    baseline.check_array(
        (model_id, None, "ubiquitin@poetctx", "logits"), produce, atol=0.1, rtol=0.05
    )
