"""Correctness tests for sequence generation / inverse folding.

* **PoET-2 generate**: fresh prompt_id (cache miss) + fixed seed -> genuine recompute
  that's still reproducible against a committed baseline.
* **ESM-IF1 inverse folding**: native-sequence recovery is a *property of the output*
  (holds whether fresh or cached), so it's a robust ground-truth check. Its prod
  baseline is genuine only on the first call after a deploy / cache clear.
"""

from pathlib import Path

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.errors import HTTPError
from openprotein.molecules import Protein
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness import metrics, well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model

GENERATE_TIMEOUT = scaled_timeout(2.0)
STRUCTURE = Path("tests/data/8bo9.cif")
SDN_LINKER = b"MHHHHHHSDN"
# ESM-IF1 native-backbone sequence recovery is typically ~0.5; allow margin.
MIN_ESMIF_RECOVERY = 0.3


def _esmif1(session: OpenProtein):
    """Return the ESM-IF1 model or skip if the backend lacks it.

    `session.models` eagerly constructs every foundation model (fetching metadata),
    so a missing esm-if1 raises HTTPError 404 on attribute access -- catch and skip.
    """
    try:
        return session.models.esmif1
    except HTTPError:
        pytest.skip("esmif1 not available in this backend")


def _load_backbone() -> Protein:
    if not STRUCTURE.exists():
        pytest.skip(f"missing {STRUCTURE}")
    protein = Protein.from_filepath(path=STRUCTURE, chain_id="A")
    return protein[len(SDN_LINKER):]


def _native_str(backbone: Protein) -> str:
    seq = backbone.sequence
    return seq.decode() if isinstance(seq, bytes) else seq


# --------------------------------------------------------------------------- #
# PoET-2 generate: genuine recompute via fresh prompt + fixed seed.
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
def test_poet2_generate_deterministic(session: OpenProtein):
    """Same seed + identical prompt content (but fresh prompt_ids) -> identical samples.

    Distinct prompt_ids force both runs to recompute (cache miss); the fixed seed and
    fixed content make the sampled sequences reproducible.
    """
    model = require_embedding_model(session, "poet-2")
    p1 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=GENERATE_TIMEOUT)
    p2 = fresh_prompt(session, well_known.POET_CONTEXT, timeout=GENERATE_TIMEOUT)

    out1 = model.generate(prompt=p1, num_samples=4, temperature=1.0, seed=7)
    assert out1.wait_until_done(timeout=GENERATE_TIMEOUT)
    out2 = model.generate(prompt=p2, num_samples=4, temperature=1.0, seed=7)
    assert out2.wait_until_done(timeout=GENERATE_TIMEOUT)

    seqs1 = sorted(r.sequence for r in out1.get())
    seqs2 = sorted(r.sequence for r in out2.get())
    assert seqs1 == seqs2


# NOTE: no prod-baseline on generated scores. Generation is seeded sampling whose
# RNG/impl differs across worker versions, so a generate score is not comparable
# prod-vs-staging (it diverged by ~5-50 cross-env). Within-env determinism (above) and
# native-recovery ground-truth (below) are the meaningful generate checks.


# --------------------------------------------------------------------------- #
# ESM-IF1 inverse folding: property-based ground-truth (+ cold-cache baseline).
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
@pytest.mark.correctness
def test_esmif1_recovers_native_at_low_temperature(session: OpenProtein):
    """At low temperature ESM-IF1 recovers a meaningful fraction of native residues.

    Ground-truth: recovery is a property of the output, valid whether fresh or cached.
    """
    model = _esmif1(session)
    backbone = _load_backbone()
    native = _native_str(backbone)

    future = model.generate(query=backbone, num_samples=4, temperature=0.1, seed=0)
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    designs = [r.sequence for r in future.get()]

    recoveries = [
        metrics.sequence_recovery(d, native) for d in designs if len(d) == len(native)
    ]
    assert recoveries, "no design matched the native length"
    assert max(recoveries) >= MIN_ESMIF_RECOVERY, f"best recovery {max(recoveries):.3f} too low"


@pytest.mark.e2e
@pytest.mark.correctness
@pytest.mark.differential
def test_esmif1_recovery_matches_prod_baseline(session: OpenProtein, baseline):
    """Mean native-recovery on the 8bo9 backbone tracks the prod baseline.

    COLD-CACHE CAVEAT: ESM-IF1 has no prompt lever; genuine only on the first call
    after a deploy / cache clear.
    """
    model = _esmif1(session)
    backbone = _load_backbone()
    native = _native_str(backbone)

    def produce():
        future = model.generate(query=backbone, num_samples=4, temperature=0.1, seed=0)
        assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
        designs = [r.sequence for r in future.get()]
        recoveries = [metrics.sequence_recovery(d, native) for d in designs if len(d) == len(native)]
        return float(np.mean(recoveries))

    baseline.check_scalar(("esmif1", None, "8bo9_A", "native_recovery_mean"), produce)
