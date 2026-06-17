"""Ground-truth dtype contract for model array outputs.

The wire dtype of each output is part of the API contract: clients depend on it
(fp16 per-residue embeddings, fp32 reduced embeddings, fp16 logits/attention), and
a backend change that silently flips a dtype — e.g. a reduced embedding stored
fp16 instead of fp32 — breaks parity without moving values enough for a closeness
check to notice. These tests assert the *exact* numpy dtype each array output
comes back as. Values are covered by the ``@differential`` baseline tests.

The expected dtypes were captured from prod (https://api.openprotein.ai/api/) via
``scripts/probe_dtypes.py`` on 2026-06-16 — observed, not guessed:

    model    embed[residue]  embed[MEAN]  logits   attn
    esm2     float16         float32      float16  float16
    poet     float16         float32      float16  (not exposed)
    poet-2   float16         float32      float16  (not exposed)

Why MEAN is fp32 (not fp16): a ``MEAN``/``SUM`` request is dispatched through the
``MEAN_SUM`` reduction path on the backend, which computes the reduction in (and
returns) float32 — so the reduced embedding is fp32 even though the per-residue
embedding is fp16.

Not asserted here: ``score`` (a scalar that arrives as float64 over the wire — an
SDK/JSON artifact, not a worker-precision contract); esmc-300m (not available on
the probed backend); PoET attention (not exposed by the SDK).
"""

import numpy as np
import pytest
from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType

from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model
from tests.utils.sequences import mutate_sequence

TIMEOUT = scaled_timeout(1.0)
POET_TIMEOUT = scaled_timeout(3.0)

ENCODER_MODELS = ["esm2_t33_650M_UR50D"]
POET_MODELS = ["poet", "poet-2"]

# Observed prod wire dtypes (scripts/probe_dtypes.py, 2026-06-16).
EXPECTED = {
    "esm2_t33_650M_UR50D": {
        "embed_residue": np.float16,
        "embed_mean": np.float32,
        "logits": np.float16,
        "attn": np.float16,
    },
    "poet": {
        "embed_residue": np.float16,
        "embed_mean": np.float32,
        "logits": np.float16,
    },
    "poet-2": {
        "embed_residue": np.float16,
        "embed_mean": np.float32,
        "logits": np.float16,
    },
}


# --------------------------------------------------------------------------- #
# Encoders (esm2): embed (per-residue + MEAN), logits, attention.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", ENCODER_MODELS)
def test_encoder_output_dtypes_match_prod(session: OpenProtein, model_id: str):
    model = require_embedding_model(session, model_id)
    seq = mutate_sequence(well_known.UBIQUITIN).encode()
    expected = EXPECTED[model_id]

    ((_, residue),) = model.embed(sequences=[seq], reduction=None).wait(timeout=TIMEOUT)
    assert np.asarray(residue).dtype == expected["embed_residue"]

    ((_, mean),) = model.embed(sequences=[seq], reduction=ReductionType.MEAN).wait(
        timeout=TIMEOUT
    )
    assert np.asarray(mean).dtype == expected["embed_mean"]

    ((_, logits),) = model.logits(sequences=[seq]).wait(timeout=TIMEOUT)
    assert np.asarray(logits).dtype == expected["logits"]

    ((_, attn),) = model.attn(sequences=[seq]).wait(timeout=TIMEOUT)
    assert np.asarray(attn).dtype == expected["attn"]


# --------------------------------------------------------------------------- #
# PoET (poet/poet-2): embed (per-residue + MEAN), logits. One fresh prompt per
# run; dtype is independent of caching.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", POET_MODELS)
def test_poet_output_dtypes_match_prod(session: OpenProtein, model_id: str):
    model = require_embedding_model(session, model_id)
    query = well_known.POET_QUERY.encode()
    expected = EXPECTED[model_id]
    prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)

    residue = model.embed(sequences=[query], prompt=prompt, reduction=None).wait(
        timeout=POET_TIMEOUT
    )[0][1]
    assert np.asarray(residue).dtype == expected["embed_residue"]

    mean = model.embed(
        sequences=[query], prompt=prompt, reduction=ReductionType.MEAN
    ).wait(timeout=POET_TIMEOUT)[0][1]
    assert np.asarray(mean).dtype == expected["embed_mean"]

    logits = model.logits(sequences=[query], prompt=prompt).wait(timeout=POET_TIMEOUT)[
        0
    ][1]
    assert np.asarray(logits).dtype == expected["logits"]
