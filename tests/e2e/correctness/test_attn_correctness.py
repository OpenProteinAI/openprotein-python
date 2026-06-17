"""Correctness tests for attention.

Attention is an encoder-only output (esm2/esmc); PoET raises NotImplementedError.
The shape/trim/non-negativity invariants feed a freshly mutated sequence each run
(cache miss -> genuine recompute on the current workers); the per-head summary
prod-baseline keeps a fixed sequence to match the committed baseline and is
cache-robust (genuine on a cold cache).
"""

import numpy as np
import pytest

from openprotein import OpenProtein
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import well_known
from tests.e2e.correctness.support import require_embedding_model
from tests.utils.sequences import mutate_sequence

TIMEOUT = scaled_timeout(1.0)
ENCODER_MODELS = ["esm2_t33_650M_UR50D", "esmc-300m"]


def _attn_one(model, seq: bytes) -> np.ndarray:
    # Callers pass a freshly mutated sequence (cache miss) for a genuine recompute;
    # the prod-baseline differential keeps a fixed sequence and is cache-robust.
    ((returned, arr),) = model.attn(sequences=[seq]).wait(timeout=TIMEOUT)
    assert returned == seq
    return np.asarray(arr)


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_attn_shape_and_finite(session: OpenProtein, model_id: str):
    """Attention is finite and square on its two trailing (length) axes.

    A freshly mutated sequence each run cache-misses, so the attention genuinely
    recomputes on the current workers.
    """
    model = require_embedding_model(session, model_id)
    a = _attn_one(model, mutate_sequence(well_known.UBIQUITIN).encode())
    assert a.ndim >= 3
    assert a.shape[-1] == a.shape[-2]  # length x length
    assert np.isfinite(a).all()


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_attn_trimmed_on_length_axes_not_heads(session: OpenProtein, model_id: str):
    """Two sequences of different lengths trim the trailing length axes, not the head axis.

    Directly guards the 4D-trim regression class: the length mask must apply to the
    two trailing (L, L) axes; the leading (head) axes must be identical across
    sequences of different length. A freshly mutated base each run (short is its
    prefix) keeps the two sequences distinct -> distinct cache keys -> genuine
    recompute.
    """
    model = require_embedding_model(session, model_id)
    base = mutate_sequence(well_known.UBIQUITIN)
    short = base[:40].encode()  # length 40
    long = base.encode()  # length 76

    a_short = _attn_one(model, short)
    a_long = _attn_one(model, long)

    delta_seq = len(long) - len(short)
    delta_attn = a_long.shape[-1] - a_short.shape[-1]
    assert a_short.shape[-1] == a_short.shape[-2]
    assert a_long.shape[-1] == a_long.shape[-2]
    assert delta_attn == delta_seq, (
        f"trailing axis grew by {delta_attn}, expected {delta_seq} "
        "(length mask likely applied to the wrong axis)"
    )
    # ...while leading (head/layer) axes are identical.
    assert a_short.shape[:-2] == a_long.shape[:-2]


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_attn_weights_nonnegative(session: OpenProtein, model_id: str):
    """Attention weights are non-negative.

    Empirically the platform returns post-softmax weights in [0, 1], but
    special-token *columns* are trimmed and values are
    float16, so per-row sums are neither exactly 1 nor cleanly <= 1 (trimmed mass +
    float16 accumulation over the length axis). Non-negativity is the only
    normalization invariant that holds robustly across encoder models.

    A freshly mutated sequence each run cache-misses, so the attention genuinely
    recomputes on the current workers.
    """
    model = require_embedding_model(session, model_id)
    a = _attn_one(model, mutate_sequence(well_known.UBIQUITIN).encode())
    assert np.all(a >= -1e-6), "attention weights must be non-negative"


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_attn_summary_matches_prod(session: OpenProtein, model_id: str, baseline):
    """Per-head mean attention (a compact summary) matches the prod baseline.

    Fixed sequence (to match the committed baseline), so this is cache-robust:
    genuine on a cold cache, and a systematic difference (e.g. a different attention
    special-token trim) still surfaces here. Relaxed tolerance absorbs cross-env
    jitter.
    """
    model = require_embedding_model(session, model_id)
    seq = well_known.UBIQUITIN.encode()

    def produce():
        a = _attn_one(model, seq)
        # Collapse the two length axes to a per-head mean -> shape = leading axes.
        return a.mean(axis=(-1, -2))

    baseline.check_array(
        (model_id, None, "ubiquitin", "attn_head_mean"), produce, atol=0.05, rtol=0.05
    )
