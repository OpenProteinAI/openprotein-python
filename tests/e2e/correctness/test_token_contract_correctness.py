"""Per-model special-token (length) contract -- cache-immune.

Each embedding-model family adds a fixed number of special tokens to its per-residue
output (ESM/PoET: +2 BOS/EOS; ablang2/ProtT5: +1; prot-seq/rotaprot/ESMC: 0). We
assert `output_length - residue_count == expected_delta` using a NOVEL random sequence
every run, so the call is a guaranteed cache miss (genuine recompute) and needs no
committed prod baseline.

This is the robust replacement for fixed-sequence logits-shape differentials: the
backend caches inference per sequence, so a fixed-sequence baseline returns the stale
cached length on a warm cache (which produced false "trim" mismatches for the ESM
family until we switched to novel inputs). The platform is moving to residue-aligned
(delta 0) in a future release; until then each model must keep its prod convention.
"""

import numpy as np
import pytest

from openprotein import OpenProtein
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model
from tests.utils.sequences import mutate_sequence, random_sequence_fake

TIMEOUT = scaled_timeout(1.0)
POET_TIMEOUT = scaled_timeout(3.0)

# (model_id, expected special-token delta vs residue count) -- prod convention.
EXPECTED_DELTA = [
    ("esm2_t33_650M_UR50D", 2),
    ("esm1b_t33_650M_UR50S", 2),
    ("esm1v_t33_650M_UR90S_1", 2),
    ("prot-seq", 0),
    ("prott5-xl", 1),
    ("rotaprot-large-uniref50w", 0),
    ("esmc-300m", 0),
    ("poet", 2),
    ("poet-2", 2),
    ("ablang2", 1),
]


def _output_length(session: OpenProtein, model, model_id: str) -> tuple[int, int]:
    """Embed a NOVEL input (cache miss) and return (output_length, residue_count)."""
    if model_id == "ablang2":
        # ablang2 rejects non-antibody input; mutate a real pair for novelty.
        heavy = mutate_sequence(well_known.ANTIBODY_HEAVY, mutation_rate=0.05)
        light = mutate_sequence(well_known.ANTIBODY_LIGHT, mutation_rate=0.05)
        residues = len(heavy) + len(light)
        res = model.embed(
            sequences=[f"{heavy}:{light}".encode()], reduction=None
        ).wait(timeout=TIMEOUT)
    elif model_id.startswith("poet"):
        seq = random_sequence_fake(60)
        residues = len(seq)
        prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=POET_TIMEOUT)
        res = model.embed(
            sequences=[seq.encode()], reduction=None, prompt=prompt
        ).wait(timeout=POET_TIMEOUT)
    else:
        seq = random_sequence_fake(60)
        residues = len(seq)
        res = model.embed(sequences=[seq.encode()], reduction=None).wait(timeout=TIMEOUT)
    arr = np.asarray(res[0][1])
    length = arr.shape[-2] if arr.ndim >= 2 else arr.shape[0]
    return length, residues


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id,expected_delta", EXPECTED_DELTA)
def test_special_token_length_contract(
    session: OpenProtein, model_id: str, expected_delta: int
):
    """output_length - residue_count must equal the model's prod special-token delta."""
    model = require_embedding_model(session, model_id)
    length, residues = _output_length(session, model, model_id)
    delta = length - residues
    assert delta == expected_delta, (
        f"{model_id}: output_len={length}, residues={residues}, delta={delta:+d}, "
        f"expected {expected_delta:+d}"
    )
