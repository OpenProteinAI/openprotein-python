"""End-to-end tests for multi-chain (":"-delimited) support across the platform.

Multi-chain inputs join chains with ":". For non-AbLang2 models the backend
replaces ":" with a chain linker, embeds, then mean-pools the linker residues
back into a single column at the delimiter position; AbLang2 tokenizes ":"
natively. Either way, each delimiter occupies exactly one per-residue column.

These tests target the DEV backend, where multi-chain for non-AbLang2 models is
rolling out. Point the SDK at dev WITHOUT editing ~/.openprotein/config.toml --
``connect()`` reads ``OPENPROTEIN_API_BACKEND`` ahead of the toml backend:

    OPENPROTEIN_API_BACKEND=https://dev.api.openprotein.ai/api/ \\
        pixi run pytest -m e2e tests/e2e/test_multichain_e2e.py

Coverage:
  * embeddings: per-residue length is residue-aligned -- a multi-chain sequence
    yields exactly one extra column per ":" vs the single-chain concatenation of
    the same residues (the model's BOS/EOS "+2" cancels in this differential).
  * score + single-site scoring on multi-chain sequences (poet, poet-2); the ":"
    delimiter is never mutated.
  * fitting an SVD on multi-chain sequences and embedding multi-chain with it.
  * GP on top of a multi-chain SVD, and GP with a reduction on multi-chain
    sequences; predict + single-site predict on multi-chain inputs.
  * genetic-algorithm design driven by a multi-chain GP (on-SVD and
    with-reduction): every designed sequence preserves the ":" layout (same
    length, ":" frozen at its column, never introduced elsewhere).

NOTE: the predictor single-site ":"-preservation assertions
(``_assert_single_site_preserves_delimiter``) depend on backend-service-predictor
PR #57. Until it deploys to dev they are expected to fail -- they are the
acceptance check for that fix.

NOTE: the GA design assertions (``_assert_design_preserves_delimiter``) depend on
backend-service-design PR #14 (chain-aware GA) AND on the multi-chain predictor
predict path, which currently fails server-side on dev for both GP-on-SVD and
GP-with-reduction. Until both land on dev these tests are expected to fail --
they are the acceptance check for end-to-end multi-chain design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.design import ModelCriterion
from tests.e2e.config import scaled_timeout
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model
from tests.utils.sequences import (
    ANTIBODY_HEAVY_SEQUENCE,
    ANTIBODY_LIGHT_SEQUENCE,
    mutate_sequence,
    random_sequence_real,
)
from tests.utils.strings import random_string

TIMEOUT = scaled_timeout(2.0)
POET_TIMEOUT = scaled_timeout(3.0)
PRED_TIMEOUT = scaled_timeout(3.0)

POET2 = "poet-2"
SVD_GP_MODELS = ["esm2_t33_650M_UR50D", POET2, "prot-seq"]
SCORE_MODELS = ["poet", POET2]

pytestmark = pytest.mark.e2e


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _generic_chains() -> tuple[str, str]:
    """Two unrelated real-ish chains for general-purpose models."""
    return random_sequence_real(64), random_sequence_real(48)


def _antibody_chains() -> tuple[str, str]:
    """Heavy/light antibody variable domains (AbLang2 rejects random strings)."""
    return (
        mutate_sequence(ANTIBODY_HEAVY_SEQUENCE, mutation_rate=0.02),
        mutate_sequence(ANTIBODY_LIGHT_SEQUENCE, mutation_rate=0.02),
    )


def _prompt_context() -> list[str]:
    """Small single-chain context for PoET prompt creation."""
    return [random_sequence_real(64) for _ in range(3)]


def _residue_axis_len(arr) -> int:
    """Per-residue length L from a ``(L, D)`` or ``(N, L, D)`` embedding array."""
    a = np.asarray(arr)
    assert a.ndim in (2, 3), f"unexpected per-residue embedding ndim {a.ndim}"
    return a.shape[-2]


def _per_residue(model, seq: bytes, *, prompt=None):
    """Per-residue (reduction=None) embedding for a single sequence."""
    kwargs: dict = {"sequences": [seq], "reduction": None}
    if prompt is not None:
        kwargs["prompt"] = prompt
    ((_returned, arr),) = model.embed(**kwargs).wait(
        timeout=POET_TIMEOUT if prompt is not None else TIMEOUT
    )
    return np.asarray(arr)


def _multichain_same_length_dataset(n: int, heavy0: str, light0: str) -> list[str]:
    """``n`` substitution-only variants of ``heavy0:light0`` (constant raw length)."""
    return [
        f"{mutate_sequence(heavy0, mutation_rate=0.06)}:{mutate_sequence(light0, mutation_rate=0.06)}"
        for _ in range(n)
    ]


def _make_assay(session: OpenProtein, sequences: list[str]):
    """Create a multi-chain assay with one synthetic numeric property."""
    df = pd.DataFrame(
        {
            "sequence": sequences,
            "property": [float(s.count("A")) for s in sequences],
        }
    )
    return session.data.create(
        table=df,
        name=f"E2E_MultiChain_{random_string(8)}",
        description="Multi-chain E2E assay",
    )


def _delimiter_positions(seq: str) -> list[int]:
    return [i for i, c in enumerate(seq) if c == ":"]


def _assert_single_site_preserves_delimiter(predictor, base: str) -> None:
    """Predictor single-site over a multi-chain base must keep ":" intact in every
    generated mutant (acceptance check for backend-service-predictor PR #57)."""
    delim = _delimiter_positions(base)
    assert delim, "base sequence has no chain delimiter"
    result = predictor.single_site(sequence=base.encode())
    mus, _vs = result.wait(timeout=PRED_TIMEOUT)
    mutants = result.sequences
    assert len(mutants) > 0
    assert np.asarray(mus).shape[0] == len(mutants)
    for s in mutants:
        sd = s.decode() if isinstance(s, (bytes, bytearray)) else str(s)
        assert len(sd) == len(base), f"single-site changed length: {sd!r}"
        for p in delim:
            assert sd[p] == ":", f"single-site mutated the delimiter at {p}: {sd!r}"


# --------------------------------------------------------------------------- #
# Embeddings: multi-chain per-residue output length
# --------------------------------------------------------------------------- #

# Linker-mean models: the backend replaces ":" with a linker and mean-pools it
# into one column, so the multi-chain per-residue length is the single-chain
# (same residues) length + one column per delimiter. The model's BOS/EOS offset
# cancels in this differential. (model_id, needs_prompt)
_DIFFERENTIAL_LENGTH_MODELS = [
    ("esm2_t33_650M_UR50D", False),
    ("esmc-300m", False),
    ("prot-seq", False),
    ("poet", True),
    ("poet-2", True),
]


@pytest.mark.parametrize(
    "model_id,needs_prompt",
    _DIFFERENTIAL_LENGTH_MODELS,
    ids=[m[0] for m in _DIFFERENTIAL_LENGTH_MODELS],
)
def test_multichain_embedding_length_differential(
    session: OpenProtein, model_id: str, needs_prompt: bool
):
    """A multi-chain sequence yields exactly one extra per-residue column per ":"
    vs the single-chain concatenation of the same residues."""
    model = require_embedding_model(session, model_id)
    heavy, light = _generic_chains()
    single = (heavy + light).encode()
    multi = f"{heavy}:{light}".encode()
    n_delim = 1

    prompt = (
        fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)
        if needs_prompt
        else None
    )
    single_len = _residue_axis_len(_per_residue(model, single, prompt=prompt))
    multi_len = _residue_axis_len(_per_residue(model, multi, prompt=prompt))

    assert multi_len == single_len + n_delim, (
        f"{model_id}: expected multi-chain per-residue length "
        f"{single_len + n_delim} (single {single_len} + {n_delim} delimiter "
        f"column), got {multi_len}"
    )


def test_multichain_embedding_length_ablang2(session: OpenProtein):
    """AbLang2 tokenizes ":" natively (it rejects a fused single chain), so its
    per-residue output aligns 1:1 with the raw input string -- the delimiter
    occupies exactly one column."""
    model = require_embedding_model(session, "ablang2")
    heavy, light = _antibody_chains()
    multi = f"{heavy}:{light}"
    multi_len = _residue_axis_len(_per_residue(model, multi.encode()))

    assert multi_len == len(multi), (
        f"ablang2: expected per-residue length {len(multi)} (1:1 with the raw "
        f"input including ':'), got {multi_len}"
    )


# --------------------------------------------------------------------------- #
# Scoring: score + single-site on multi-chain sequences (PoET / PoET-2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", SCORE_MODELS)
def test_multichain_score(session: OpenProtein, model_id: str):
    """score() accepts a multi-chain sequence and returns a finite score."""
    model = require_embedding_model(session, model_id)
    heavy, light = _generic_chains()
    multi = f"{heavy}:{light}".encode()
    prompt = fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)

    rows = model.score(sequences=[multi], prompt=prompt).wait(timeout=POET_TIMEOUT)
    assert len(rows) == 1
    score = float(np.asarray(rows[0].score).ravel()[0])
    assert np.isfinite(score)


@pytest.mark.parametrize("model_id", SCORE_MODELS)
def test_multichain_single_site_score(session: OpenProtein, model_id: str):
    """single_site() over a multi-chain base never mutates ":" and covers exactly
    the residue positions."""
    model = require_embedding_model(session, model_id)
    heavy, light = _generic_chains()
    base = f"{heavy}:{light}"
    n_res = len(heavy) + len(light)
    prompt = fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)

    rows = model.single_site(sequence=base.encode(), prompt=prompt).wait(
        timeout=POET_TIMEOUT
    )

    positions = set()
    for row in rows:
        code = row.mut_code
        if not code[1:-1].isdigit():  # skip "WT" identity rows
            continue
        assert code[0] != ":", f"{model_id}: delimiter was mutated ({code})"
        positions.add(int(code[1:-1]))
    assert len(positions) == n_res, (
        f"{model_id}: expected {n_res} mutated residue positions "
        f"(delimiter excluded), got {len(positions)}"
    )


# --------------------------------------------------------------------------- #
# SVD on multi-chain sequences
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", SVD_GP_MODELS)
def test_multichain_svd_fit_and_embed(session: OpenProtein, model_id: str):
    """Fit an SVD on same-length multi-chain sequences, then embed multi-chain
    sequences with it."""
    model = require_embedding_model(session, model_id)
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(24, heavy0, light0)
    n_components = 8

    extra = {}
    if model_id == POET2:
        extra["prompt"] = fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)

    svd = model.fit_svd(
        sequences=[s.encode() for s in seqs], n_components=n_components, **extra
    ).wait(timeout=POET_TIMEOUT)
    assert svd.n_components == n_components

    ((_seq, emb),) = svd.embed(sequences=[seqs[0].encode()]).wait(timeout=POET_TIMEOUT)
    assert np.asarray(emb).shape == (n_components,)


# --------------------------------------------------------------------------- #
# GP predictors on multi-chain sequences
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_id", SVD_GP_MODELS)
def test_multichain_gp_on_svd(session: OpenProtein, model_id: str):
    """Train a GP on an SVD fitted on multi-chain sequences; predict and
    single-site predict on multi-chain inputs."""
    model = require_embedding_model(session, model_id)
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(40, heavy0, light0)

    extra = {}
    if model_id == POET2:
        extra["prompt"] = fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)
    svd = model.fit_svd(
        sequences=[s.encode() for s in seqs], n_components=8, **extra
    ).wait(timeout=POET_TIMEOUT)

    assay = _make_assay(session, seqs)
    predictor = svd.fit_gp(assay=assay, properties=[assay.measurement_names[0]])
    assert predictor.wait(timeout=PRED_TIMEOUT), (
        f"GP-on-SVD training failed for {model_id}"
    )

    base = seqs[0]
    mus, vs = predictor.predict(sequences=[base.encode()]).wait(timeout=PRED_TIMEOUT)
    assert np.asarray(mus).shape == (1, 1)
    assert np.all(np.asarray(vs) >= 0.0)

    _assert_single_site_preserves_delimiter(predictor, base)


@pytest.mark.parametrize("model_id", SVD_GP_MODELS)
def test_multichain_gp_with_reduction(session: OpenProtein, model_id: str):
    """Train a GP directly on reduced (MEAN) embeddings of multi-chain sequences;
    predict and single-site predict on multi-chain inputs."""
    model = require_embedding_model(session, model_id)
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(40, heavy0, light0)
    assay = _make_assay(session, seqs)

    extra = {}
    if model_id == POET2:
        extra["prompt"] = fresh_prompt(session, _prompt_context(), timeout=POET_TIMEOUT)
    predictor = model.fit_gp(
        assay=assay,
        properties=[assay.measurement_names[0]],
        reduction=ReductionType.MEAN,
        **extra,
    )
    assert predictor.wait(timeout=PRED_TIMEOUT), (
        f"GP-with-reduction training failed for {model_id}"
    )

    base = seqs[0]
    mus, vs = predictor.predict(sequences=[base.encode()]).wait(timeout=PRED_TIMEOUT)
    assert np.asarray(mus).shape == (1, 1)
    assert np.all(np.asarray(vs) >= 0.0)

    _assert_single_site_preserves_delimiter(predictor, base)


# --------------------------------------------------------------------------- #
# Genetic-algorithm design on multi-chain sequences
# --------------------------------------------------------------------------- #

DESIGN_TIMEOUT = scaled_timeout(4.0)
DESIGN_MODEL = "esm2_t33_650M_UR50D"


def _assert_design_preserves_delimiter(results, base: str) -> None:
    """Every GA-designed sequence must keep ``base``'s chain layout intact: same
    length, exactly one ":" per delimiter at its original column, and finite
    scores. This is the acceptance check that the design GA never mutates or
    introduces ":" (backend-service-design PR #14)."""
    delim = _delimiter_positions(base)
    assert delim, "base sequence has no chain delimiter"
    assert len(results) > 0, "design produced no results"
    for r in results:
        seq = r.sequence
        assert isinstance(seq, str) and seq, f"empty design sequence: {seq!r}"
        assert len(seq) == len(base), f"design changed length: {seq!r}"
        assert seq.count(":") == len(delim), f"design changed delimiter count: {seq!r}"
        for p in delim:
            assert seq[p] == ":", f"design mutated the delimiter at {p}: {seq!r}"
        assert isinstance(r.scores, np.ndarray)
        assert r.scores.size > 0 and np.isfinite(r.scores).all(), (
            f"non-finite/empty design scores: {r.scores!r}"
        )


def _run_multichain_design(session: OpenProtein, assay, property_name: str, predictor):
    """Build a maximize criterion against ``predictor`` and run a tiny GA design."""
    criterion = (
        ModelCriterion(model_id=predictor.id, measurement_name=property_name) > 0
    )
    design = session.design.create_genetic_algorithm_design(
        assay=assay, criteria=criterion, num_steps=2, pop_size=4
    )
    assert design.wait_until_done(timeout=DESIGN_TIMEOUT), "multi-chain design failed"
    return design.get()


def test_multichain_ga_design_gp_on_svd(session: OpenProtein):
    """GA design driven by a GP trained on an SVD of multi-chain sequences; every
    designed sequence must preserve the ":" layout (acceptance check -- see the
    module NOTE on backend-service-design PR #14 + the multi-chain predict path)."""
    model = require_embedding_model(session, DESIGN_MODEL)
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(40, heavy0, light0)
    assay = _make_assay(session, seqs)
    property_name = assay.measurement_names[0]

    svd = model.fit_svd(sequences=[s.encode() for s in seqs], n_components=8).wait(
        timeout=PRED_TIMEOUT
    )
    predictor = svd.fit_gp(assay=assay, properties=[property_name])
    assert predictor.wait(timeout=PRED_TIMEOUT), "GP-on-SVD training failed"

    results = _run_multichain_design(session, assay, property_name, predictor)
    _assert_design_preserves_delimiter(results, seqs[0])


def test_multichain_ga_design_gp_with_reduction(session: OpenProtein):
    """GA design driven by a GP trained on reduced (MEAN) embeddings of multi-chain
    sequences; every designed sequence must preserve the ":" layout (acceptance
    check -- see the module NOTE on backend-service-design PR #14 + the multi-chain
    predict path)."""
    model = require_embedding_model(session, DESIGN_MODEL)
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(40, heavy0, light0)
    assay = _make_assay(session, seqs)
    property_name = assay.measurement_names[0]

    predictor = model.fit_gp(
        assay=assay, properties=[property_name], reduction=ReductionType.MEAN
    )
    assert predictor.wait(timeout=PRED_TIMEOUT), "GP-with-reduction training failed"

    results = _run_multichain_design(session, assay, property_name, predictor)
    _assert_design_preserves_delimiter(results, seqs[0])
