"""Correctness tests for GP predictors.

A GP trained on assay embeddings genuinely recomputes every run: a fresh assay ->
fresh embeddings_id -> retrained predictor. Training is deterministic given the
(fixed) split and embeddings, so the held-out Spearman and predictions are
reproducible and comparable to a committed baseline.

Ground-truth (L2/L4): the predictor must recover real signal from the AMIE deep
mutational scan -- held-out Spearman above a floor. Differential (L3): that Spearman
and the held-out prediction means also track the prod baseline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import metrics
from tests.e2e.correctness.support import require_embedding_model
from tests.utils.strings import random_string

PRED_TIMEOUT = scaled_timeout(3.0)
AMIE_PATH = Path("tests/data/AMIE_PSEAE_Whitehead.wide.csv")
PROPERTY = "acetamide_normalized_fitness"
MODEL_ID = "esm2_t33_650M_UR50D"
TRAIN_N = 600
TEST_N = 150
# A GP on ESM2 embeddings recovers AMIE fitness well; require a meaningful floor.
MIN_PREDICTOR_SPEARMAN = 0.4


def _amie_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/test split of AMIE rows that have the target measurement."""
    df = pd.read_csv(AMIE_PATH)[["sequence", PROPERTY]].dropna().reset_index(drop=True)
    train = df.iloc[:TRAIN_N].reset_index(drop=True)
    test = df.iloc[TRAIN_N : TRAIN_N + TEST_N].reset_index(drop=True)
    return train, test


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
def test_predictor_recovers_amie_signal(session: OpenProtein, baseline):
    """GP trained on AMIE recovers held-out fitness; Spearman + means track prod baseline."""
    if not AMIE_PATH.exists():
        pytest.skip("AMIE DMS table not present")
    train_df, test_df = _amie_split()
    if len(train_df) < TRAIN_N or len(test_df) < 30:
        pytest.skip("not enough AMIE rows with the target measurement")

    model = require_embedding_model(session, MODEL_ID)

    # Fresh assay -> fresh embeddings_id -> genuinely retrained predictor every run.
    assay = session.data.create(
        table=train_df,
        name=f"AMIE_corr_{random_string(8)}",
        description="AMIE train split for predictor correctness",
    )
    predictor = model.fit_gp(assay=assay, properties=[PROPERTY], reduction=ReductionType.MEAN)
    assert predictor.wait(timeout=PRED_TIMEOUT), "GP training failed"

    test_seqs = [s.encode() for s in test_df["sequence"].tolist()]
    mus, vs = predictor.predict(sequences=test_seqs).wait(timeout=PRED_TIMEOUT)
    assert mus.shape == (len(test_seqs), 1)
    assert np.all(vs >= 0.0)

    measured = test_df[PROPERTY].to_numpy()
    rho = metrics.spearman(mus[:, 0], measured)

    # L4 ground-truth: must recover real signal.
    assert rho >= MIN_PREDICTOR_SPEARMAN, f"held-out Spearman {rho:.3f} below floor"

    # L3 differential: Spearman and the held-out prediction means track prod, with
    # generous tolerance -- the GP recompute over fresh embeddings drifts slightly
    # vs prod and that's accepted (accepted-jitter decision).
    baseline.check_scalar(
        (MODEL_ID, None, "amie_split", "heldout_spearman"), lambda: rho, atol=0.05
    )
    baseline.check_array(
        (MODEL_ID, None, "amie_split", "heldout_mu"), lambda: mus[:, 0], atol=0.1, rtol=0.1
    )
