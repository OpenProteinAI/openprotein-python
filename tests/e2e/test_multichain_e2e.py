"""End-to-end tests for multi-chain (":"-delimited) support across the platform.

Multi-chain inputs join chains with ":". For non-AbLang2 models the backend
replaces ":" with a chain linker, embeds, then mean-pools the linker residues
back into a single column at the delimiter position; AbLang2 tokenizes ":"
natively. Either way, each delimiter occupies exactly one per-residue column.

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
  * alignment jobs (mafft / clustalo / abnumber) on multi-chain inputs: each
    chain is aligned independently and rejoined with a single ":" column, and
    gap-stripping each chain recovers the input chains. Chain order is
    positional, so "L:H" is as valid as "H:L".
  * multi-chain input that cannot be aligned: ragged chain counts and empty
    chain positions are refused at submission; abnumber
    ``drop_minority_chains`` and a chain position holding more than one chain
    type fail the job.
  * prompt sampling from a multi-chain MSA: sampling scores residue columns, so
    the ":" break is dropped from the encoded MSA and restored in the sampled
    rows. Every context entry comes back as a Complex with the seed's chain
    layout, across the sequence and residue budgets, ensemble replicates,
    similarity bounds, and seed inclusion. Covers both a search-built MSA and
    one a user uploads directly (the non-a3m reader), including that reader's
    validation of a misaligned chain break.
  * prompt context entries may leave a chain slot empty -- a homolog covering
    one chain of the complex -- and only an entry with no residues at all is
    refused.
  * abnumber ``unnumbered_policy`` on a paired row whose heavy chain carries a
    constant region: the default fails the job, ``drop`` aligns the variable
    domains alone, and ``retain`` keeps every residue. Either way the job
    reports what it did in ``get_warnings()``, scoped to the chain position
    it affected.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.align.schemas import (
    AbNumberScheme,
    MSASamplingMethod,
    UnnumberedPolicy,
)
from openprotein.common.reduction import ReductionType
from openprotein.design import ModelCriterion
from openprotein.errors import HTTPError, InvalidParameterError
from openprotein.molecules import Complex

from tests.e2e.config import scaled_timeout
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model
from tests.utils.sequences import (
    ANTIBODY_HEAVY_CONSTANT_SEQUENCE,
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
    generated mutant."""
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
    introduces ":"."""
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
    designed sequence must preserve the ":" layout."""
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
    sequences; every designed sequence must preserve the ":" layout."""
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


# --------------------------------------------------------------------------- #
# Alignment jobs on multi-chain sequences (mafft / clustalo / abnumber)
# --------------------------------------------------------------------------- #
#
# Submitting ":"-delimited sequences to the alignment endpoints exercises the
# split / align-each-chain / rejoin path: every chain position is aligned
# independently, then rejoined with a single ":" column.

ALIGN_TIMEOUT = scaled_timeout(2.0)


def _assert_multichain_alignment(rows, inputs: list[str], n_delim: int = 1) -> None:
    """A multi-chain alignment preserves the ":" layout: exactly ``n_delim``
    delimiter(s) per row, all delimiters at the same column, uniform row length,
    and gap-stripping each chain recovers the original input chains."""
    aligned = [row[1] for row in rows]
    assert len(aligned) == len(inputs), "expected one aligned row per input"
    assert all(s.count(":") == n_delim for s in aligned), (
        f"expected exactly {n_delim} delimiter(s) per row: {aligned}"
    )
    assert len({s.index(":") for s in aligned}) == 1, "delimiters not column-aligned"
    assert len({len(s) for s in aligned}) == 1, "alignment rows are not equal length"
    recovered = {
        ":".join(c.replace("-", "").replace(".", "") for c in s.split(":"))
        for s in aligned
    }
    assert recovered == set(inputs), "gap-stripped chains do not recover the inputs"


def test_multichain_mafft_alignment(session: OpenProtein):
    """MAFFT aligns multi-chain sequences chain-wise, preserving the ":" column."""
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(3, heavy0, light0)

    future = session.align.mafft(sequences=seqs)
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    _assert_multichain_alignment(list(future.get()), seqs)


def test_multichain_clustalo_alignment(session: OpenProtein):
    """ClustalO aligns multi-chain sequences chain-wise, preserving the ":" column."""
    heavy0, light0 = _generic_chains()
    seqs = _multichain_same_length_dataset(3, heavy0, light0)

    future = session.align.clustalo(sequences=seqs)
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    _assert_multichain_alignment(list(future.get()), seqs)


def _paired_antibodies(n: int, *, heavy_first: bool = True) -> list[str]:
    """``n`` paired antibody rows. Substitution-only mutation keeps every chain
    numberable (indels would break the variable-domain framing)."""
    first, second = (
        (ANTIBODY_HEAVY_SEQUENCE, ANTIBODY_LIGHT_SEQUENCE)
        if heavy_first
        else (ANTIBODY_LIGHT_SEQUENCE, ANTIBODY_HEAVY_SEQUENCE)
    )
    return [
        f"{mutate_sequence(first, mutation_rate=0.02)}:"
        f"{mutate_sequence(second, mutation_rate=0.02)}"
        for _ in range(n)
    ]


def test_multichain_abnumber_alignment(session: OpenProtein):
    """AbNumber numbers each chain of a paired heavy:light antibody independently
    and rejoins with the ":" column."""
    seqs = _paired_antibodies(3)

    future = session.align.abnumber(sequences=seqs, scheme=AbNumberScheme.CHOTHIA)
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    _assert_multichain_alignment(list(future.get()), seqs)
    assert future.get_warnings() == [], "fully numbered input has nothing to report"


def test_multichain_abnumber_light_heavy_order(session: OpenProtein):
    """Chain order is positional, so "L:H" aligns as readily as "H:L"."""
    seqs = _paired_antibodies(3, heavy_first=False)

    future = session.align.abnumber(sequences=seqs, scheme=AbNumberScheme.CHOTHIA)
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    _assert_multichain_alignment(list(future.get()), seqs)


# --------------------------------------------------------------------------- #
# Unnumbered residues on multi-chain input
# --------------------------------------------------------------------------- #
#
# ANARCI numbers the variable domain only, so a heavy chain carrying its
# constant region has residues with no antibody position. ``unnumbered_policy``
# decides their fate, per chain position: here only the heavy position is
# affected, the light one is numbered end to end.


def _paired_antibodies_with_constant(n: int) -> list[str]:
    """``n`` paired rows whose heavy chain runs past the variable domain into
    CH1, so every heavy position carries unnumbered residues."""
    return [
        f"{mutate_sequence(ANTIBODY_HEAVY_SEQUENCE, mutation_rate=0.02)}"
        f"{ANTIBODY_HEAVY_CONSTANT_SEQUENCE}:"
        f"{mutate_sequence(ANTIBODY_LIGHT_SEQUENCE, mutation_rate=0.02)}"
        for _ in range(n)
    ]


def _variable_domains_only(seqs: list[str]) -> set[str]:
    """``seqs`` with the constant region cut off the heavy position."""
    return {
        ":".join(c.replace(ANTIBODY_HEAVY_CONSTANT_SEQUENCE, "") for c in s.split(":"))
        for s in seqs
    }


def _assert_unnumbered_warning(future, code: str, seqs: list[str]) -> None:
    """The job reports the unnumbered residues under ``code``, scoped to the
    heavy position: only chain 0 is affected, and every constant region counts
    towards the total."""
    (warning,) = future.get_warnings()
    assert warning.code == code
    assert warning.message

    metadata = warning.metadata
    assert metadata is not None
    assert metadata["chain_count"] == 1, "only the heavy position is affected"
    assert metadata["residue_count"] == len(ANTIBODY_HEAVY_CONSTANT_SEQUENCE) * len(
        seqs
    )

    (chain,) = metadata["chains"]
    assert chain["chain_index"] == 0
    assert chain["sequence_count"] == len(seqs)
    assert chain["max_prefix_length"] == 0
    assert chain["max_suffix_length"] == len(ANTIBODY_HEAVY_CONSTANT_SEQUENCE)


def test_multichain_abnumber_unnumbered_error_is_the_default(session: OpenProtein):
    """Residues outside the numbered domain fail the job unless the caller picks
    a policy, so they are never discarded by default."""
    seqs = _paired_antibodies_with_constant(2)

    future = session.align.abnumber(sequences=seqs, scheme=AbNumberScheme.CHOTHIA)
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)

    assert future.status == "FAILURE"


def test_multichain_abnumber_unnumbered_drop(session: OpenProtein):
    """``drop`` aligns the numbered variable domains and discards the rest, so
    gap-stripping recovers the input minus its constant region."""
    seqs = _paired_antibodies_with_constant(3)

    future = session.align.abnumber(
        sequences=seqs,
        scheme=AbNumberScheme.CHOTHIA,
        unnumbered_policy=UnnumberedPolicy.DROP,
    )
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    with pytest.warns(UserWarning, match="UNNUMBERED_RESIDUES_DROPPED"):
        rows = list(future.get())

    _assert_multichain_alignment(rows, list(_variable_domains_only(seqs)))
    _assert_unnumbered_warning(future, "UNNUMBERED_RESIDUES_DROPPED", seqs)


def test_multichain_abnumber_unnumbered_retain(session: OpenProtein):
    """``retain`` keeps every unnumbered residue, so the ":" layout still holds
    and gap-stripping recovers the full input including the constant region."""
    seqs = _paired_antibodies_with_constant(3)

    future = session.align.abnumber(
        sequences=seqs,
        scheme=AbNumberScheme.CHOTHIA,
        unnumbered_policy=UnnumberedPolicy.RETAIN,
    )
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)
    assert future.status == "SUCCESS"

    with pytest.warns(UserWarning, match="UNNUMBERED_RESIDUES_RETAINED"):
        rows = list(future.get())

    _assert_multichain_alignment(rows, seqs)

    aligned = [row[1] for row in rows]
    assert all(ANTIBODY_HEAVY_CONSTANT_SEQUENCE in s for s in aligned), (
        "retain pads the constant region against the domain boundary, so it "
        "stays contiguous"
    )

    _assert_unnumbered_warning(future, "UNNUMBERED_RESIDUES_RETAINED", seqs)


# --------------------------------------------------------------------------- #
# Multi-chain layout and semantics that are refused
# --------------------------------------------------------------------------- #
#
# Layout violations are structural and are caught at submission, before a job
# exists. Chain-type violations need ANARCI numbering, so they surface as a
# failed job.


def test_multichain_ragged_chain_counts_rejected(session: OpenProtein):
    """Chain positions are aligned independently and rejoined by index, so every
    row has to carry the same number of chains."""
    heavy0, light0 = _generic_chains()

    with pytest.raises(HTTPError, match="Status code 400"):
        session.align.mafft(sequences=[f"{heavy0}:{light0}", heavy0])


def test_multichain_empty_chain_rejected(session: OpenProtein):
    """A trailing ":" leaves a chain position with no residues in it."""
    heavy0, light0 = _generic_chains()

    with pytest.raises(HTTPError, match="Status code 400"):
        session.align.mafft(sequences=[f"{heavy0}:", f"{heavy0}:{light0}"])


def test_multichain_abnumber_drop_minority_chains_rejected(session: OpenProtein):
    """``drop_minority_chains`` removes whole rows, which the index-wise rejoin
    cannot honour, so it is only available for single-chain input."""
    seqs = _paired_antibodies(2)

    future = session.align.abnumber(
        sequences=seqs,
        scheme=AbNumberScheme.CHOTHIA,
        drop_minority_chains=True,
    )
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)

    assert future.status == "FAILURE"


def test_multichain_abnumber_mixed_chain_types_rejected(session: OpenProtein):
    """One chain position is numbered as a single alignment, so every row must
    agree on that position's chain type."""
    heavy = mutate_sequence(ANTIBODY_HEAVY_SEQUENCE, mutation_rate=0.02)
    light = mutate_sequence(ANTIBODY_LIGHT_SEQUENCE, mutation_rate=0.02)
    other_light = mutate_sequence(ANTIBODY_LIGHT_SEQUENCE, mutation_rate=0.02)

    future = session.align.abnumber(
        sequences=[f"{heavy}:{light}", f"{other_light}:{light}"],
        scheme=AbNumberScheme.CHOTHIA,
    )
    future.wait_until_done(verbose=True, timeout=ALIGN_TIMEOUT)

    assert future.status == "FAILURE"


# --------------------------------------------------------------------------- #
# Prompt sampling from multi-chain MSAs
# --------------------------------------------------------------------------- #
#
# Sampling scores residue columns, so the ":" break is dropped from the encoded
# MSA and restored in the sampled rows. Every sampled context entry therefore
# carries the seed's chain layout and comes back as a Complex.

PROMPT_TIMEOUT = scaled_timeout(4.0)
PROMPT_CONTEXT_SIZE = 8


def _multichain_msa(session: OpenProtein, chain_a: str, chain_b: str) -> MSAFuture:
    """MSA search seeded with a two-chain complex."""
    return session.align.create_msa(seed=f"{chain_a}:{chain_b}".encode())


def _uploaded_multichain_msa(
    session: OpenProtein, query: str, rows: list[str]
) -> MSAFuture:
    """A ready-made multi-chain MSA a user uploads directly, bypassing homology
    search. This is the non-a3m path: a plain FASTA block, no insertions."""
    records = [f">query\n{query}"] + [
        f">homolog_{i}\n{row}" for i, row in enumerate(rows)
    ]
    fasta = ("\n".join(records) + "\n").encode()
    return session.align.upload_msa(io.BytesIO(fasta))


def _assert_multichain_prompt(replicates, n_chains: int = 2) -> None:
    """Every sampled context entry carries the seed's chain layout: a Complex of
    ``n_chains`` protein chains, none of them holding a further ":"."""
    assert replicates, "prompt returned no replicates"
    for entries in replicates:
        assert entries, "prompt replicate has no context entries"
        for entry in entries:
            assert isinstance(entry, Complex), (
                f"multi-chain context entry came back as {type(entry).__name__}"
            )
            proteins = entry.get_proteins()
            assert len(proteins) == n_chains, (
                f"expected {n_chains} chains, got {len(proteins)}"
            )
            assert len(entry.get_chains()) == len(proteins), (
                "non-protein chain in prompt"
            )
            for chain in proteins.values():
                assert b":" not in chain.sequence, f"chain retains a break: {chain!r}"


def _sample(msa: MSAFuture, **kwargs):
    """Sample a prompt and return its replicates, asserting the job succeeded."""
    prompt = msa.sample_prompt(**kwargs)
    prompt.wait_until_done(verbose=True, timeout=PROMPT_TIMEOUT)
    assert prompt.status == "SUCCESS"
    return prompt.get()


def test_multichain_prompt_sampling(session: OpenProtein):
    """Sampling a multi-chain MSA yields context entries with the seed's layout."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    _assert_multichain_prompt(_sample(msa, num_sequences=PROMPT_CONTEXT_SIZE))


def test_multichain_prompt_sampling_by_residues(session: OpenProtein):
    """The residue budget counts the break column, so a token-bounded prompt keeps
    the same chain layout as a sequence-bounded one."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    _assert_multichain_prompt(_sample(msa, num_residues=2048))


def test_multichain_prompt_sampling_ensemble(session: OpenProtein):
    """Each ensemble replicate is sampled with its own seed and keeps the layout."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    replicates = _sample(msa, num_sequences=PROMPT_CONTEXT_SIZE, num_ensemble_prompts=3)
    assert len(replicates) == 3
    _assert_multichain_prompt(replicates)


def test_multichain_prompt_sampling_similarity_bounds(session: OpenProtein):
    """Similarity bounds are measured over residue columns. A break counted as a
    matching column shifts every row's similarity toward the seed's, which is what
    these bounds gate on."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    _assert_multichain_prompt(
        _sample(
            msa,
            num_sequences=PROMPT_CONTEXT_SIZE,
            max_similarity=0.95,
            min_similarity=0.2,
        )
    )


def test_multichain_prompt_sampling_random_method(session: OpenProtein):
    """RANDOM skips the homology weights, so it isolates the similarity filter."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    _assert_multichain_prompt(
        _sample(
            msa,
            num_sequences=PROMPT_CONTEXT_SIZE,
            method=MSASamplingMethod.RANDOM,
        )
    )


def test_multichain_prompt_sampling_includes_seed(session: OpenProtein):
    """``always_include_seed_sequence`` puts the seed complex in the context with
    both chains at full length."""
    chain_a, chain_b = _generic_chains()
    msa = _multichain_msa(session, chain_a, chain_b)

    replicates = _sample(
        msa,
        num_sequences=PROMPT_CONTEXT_SIZE,
        always_include_seed_sequence=True,
    )
    _assert_multichain_prompt(replicates)

    wanted = sorted([chain_a, chain_b])
    found = [
        sorted(p.sequence.decode() for p in entry.get_proteins().values())
        for entries in replicates
        for entry in entries
    ]
    assert wanted in found, "seed complex is not in the sampled context"


def test_multichain_prompt_sampling_from_uploaded_msa(session: OpenProtein):
    """A user-uploaded MSA has no homology-search a3m behind it -- it hits the
    worker's plain-FASTA/CSV reader, which has to drop the chain break itself
    just like the a3m reader does."""
    chain_a, chain_b = _generic_chains()
    query = f"{chain_a}:{chain_b}"
    homologs = _multichain_same_length_dataset(10, chain_a, chain_b)
    msa = _uploaded_multichain_msa(session, query, homologs)

    _assert_multichain_prompt(_sample(msa, num_sequences=PROMPT_CONTEXT_SIZE))


def test_multichain_prompt_uploaded_msa_misaligned_break_fails(session: OpenProtein):
    """Every row of an uploaded MSA has to place its break at the query's column.
    A homolog whose break lands one column early (same total width, wrong chain
    boundary) fails the prompt job instead of sampling on a silently misaligned
    complex."""
    chain_a, chain_b = _generic_chains()
    query = f"{chain_a}:{chain_b}"
    good = _multichain_same_length_dataset(5, chain_a, chain_b)
    misaligned = f"{chain_a[:-1]}:{chain_a[-1]}{chain_b}"
    msa = _uploaded_multichain_msa(session, query, good + [misaligned])

    prompt = msa.sample_prompt(num_sequences=PROMPT_CONTEXT_SIZE)
    prompt.wait_until_done(verbose=True, timeout=PROMPT_TIMEOUT)

    assert prompt.status == "FAILURE"


def test_multichain_prompt_empty_chain_round_trips(session: OpenProtein):
    """A chain position with no residues is an empty slot, and it survives the
    round trip.

    Sampling emits this shape for a homolog that covers only one chain of the
    seed: the absent chain gap-strips to an empty segment, so the row renders as
    ":SEQ". The empty slot is what says which chain the residues cover, so it has
    to reach the service and come back rather than being refused or collapsed.
    """
    chain_a, _chain_b = _generic_chains()

    for sequence, expected in [
        (f":{chain_a}", [b"", chain_a.encode()]),
        (f"{chain_a}:", [chain_a.encode(), b""]),
    ]:
        prompt = session.prompt.create_prompt([sequence])
        (entry,) = prompt.get()[0]
        assert isinstance(entry, Complex), (
            f"empty-slot entry came back as {type(entry).__name__}"
        )
        assert [p.sequence for p in entry.get_proteins().values()] == expected


def test_multichain_prompt_entry_without_residues_rejected(session: OpenProtein):
    """An entry that is empty in every chain carries nothing, so the client refuses
    it rather than creating a prompt with a contentless context entry."""
    with pytest.raises(InvalidParameterError, match="no residues"):
        session.prompt.create_prompt([":"])
