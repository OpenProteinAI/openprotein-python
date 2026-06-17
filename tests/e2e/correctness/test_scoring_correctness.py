"""Correctness tests for sequence scoring / variant effects.

PoET-only (poet/poet-2). Every probe mints a fresh prompt_id from FIXED content,
so it genuinely recomputes on the current workers while staying comparable to a
committed baseline.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from openprotein import OpenProtein
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import metrics, well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model

TIMEOUT = scaled_timeout(2.0)
SCORE_TIMEOUT = scaled_timeout(3.0)
SCORE_MODELS = ["poet", "poet-2"]
AMIE_PATH = Path("tests/data/AMIE_PSEAE_Whitehead.wide.csv")
AMIE_MEASUREMENT = "acetamide_normalized_fitness"
# Zero-shot PLM variant-effect prediction on AMIE typically reaches Spearman ~0.3+.
MIN_VARIANT_EFFECT_SPEARMAN = 0.2


def _score(model, sequence: bytes, prompt) -> float:
    # PoET cache-busts via the fresh prompt_id, so each score genuinely recomputes.
    rows = model.score(sequences=[sequence], prompt=prompt).wait(timeout=TIMEOUT)
    assert len(rows) == 1
    return float(np.asarray(rows[0].score).ravel()[0])


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.parametrize("model_id", SCORE_MODELS)
def test_native_scores_above_shuffled(session: OpenProtein, model_id: str):
    """A natural sequence scores higher likelihood than a shuffled version of itself.

    Both scores use one fresh prompt; the two sequences differ, so each is a distinct
    cache key and genuinely computed.
    """
    model = require_embedding_model(session, model_id)
    prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=TIMEOUT)

    native = well_known.POET_QUERY
    shuffled = "".join(random.Random(0).sample(list(native), len(native)))

    native_score = _score(model, native.encode(), prompt)
    shuffled_score = _score(model, shuffled.encode(), prompt)
    assert native_score > shuffled_score


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
@pytest.mark.parametrize("model_id", SCORE_MODELS)
def test_variant_effect_correlates_with_amie(
    session: OpenProtein, model_id: str, baseline
):
    """single_site scores over AMIE single mutants correlate with measured DMS fitness.

    PoET is conditioned on a fixed slice of AMIE homologs (fresh prompt_id each run),
    so the correlation genuinely recomputes and is comparable to a committed baseline.
    """
    if not AMIE_PATH.exists():
        pytest.skip("AMIE DMS table not present")
    model = require_embedding_model(session, model_id)
    wildtype, variants = well_known.load_amie_dms(
        AMIE_PATH, measurement=AMIE_MEASUREMENT
    )
    context = well_known.amie_prompt_context(AMIE_PATH, n=24)
    prompt = fresh_prompt(session, context, timeout=SCORE_TIMEOUT)

    rows = model.single_site(sequence=wildtype.encode(), prompt=prompt).wait(
        timeout=SCORE_TIMEOUT
    )
    # mut_code is "<wt><pos1based><mut>" (e.g. "A1R", "L10V"). single_site also emits a
    # per-position "WT" identity row (no position) -- skip any row whose middle isn't a
    # position number.
    score_by_mut: dict[tuple[int, str], float] = {}
    for row in rows:
        code = row.mut_code
        if not code[1:-1].isdigit():
            continue
        mut_aa, pos0 = code[-1], int(code[1:-1]) - 1
        score_by_mut[(pos0, mut_aa)] = float(np.asarray(row.score).ravel()[0])

    measured, predicted = [], []
    for pos0, mut_aa, fitness in variants:
        key = (pos0, mut_aa)
        if key in score_by_mut:
            measured.append(fitness)
            predicted.append(score_by_mut[key])

    assert len(measured) >= 50, f"too few overlapping variants: {len(measured)}"
    rho = metrics.spearman(np.array(predicted), np.array(measured))

    # L2 floor: zero-shot variant effect must clear a minimum quality bar.
    assert (
        rho >= MIN_VARIANT_EFFECT_SPEARMAN
    ), f"variant-effect Spearman {rho:.3f} too low"

    # L3: the correlation tracks the captured prod baseline within a generous band
    # (small cross-env drift accepted; single_site is conditioned on a fresh prompt).
    baseline.check_scalar(
        (model_id, None, "amie", "variant_effect_spearman"), lambda: rho, atol=0.02
    )
