"""E2E tests for the structure-generation domain (rfdiffusion, boltzgen, etc.)."""

from pathlib import Path

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.molecules import Complex, Protein, Template
from tests.e2e.config import scaled_timeout

GENERATE_TIMEOUT = scaled_timeout(2.0)


def _assert_generated_sequences(future, expected_count: int) -> None:
    sequences = future.get(output="sequences")
    loglikelihoods = future.get(output="loglikelihood")
    assert len(sequences) == len(loglikelihoods) == expected_count
    for sequence in sequences:
        assert isinstance(sequence, str)
        assert len(sequence) > 0


@pytest.mark.e2e
def test_e2e_structure_generate_then_sequence_generate_with_design(
    session: OpenProtein,
):
    """
    Run a structure-generation -> sequence-generation workflow.

    This validates that a structure generation future can be consumed directly
    by sequence generation through `design`.
    """
    n_designs = 1
    n_sequences = 2
    design_future = session.models.rfdiffusion.generate(
        query=Protein.from_expr(60), num_samples=n_designs
    )
    assert design_future.wait_until_done(timeout=GENERATE_TIMEOUT)

    designs = design_future.get()
    assert len(designs) == n_designs
    assert isinstance(designs[0], Complex)

    seq_future = session.models.proteinmpnn.generate(
        design=design_future,
        num_samples=n_sequences,
        temperature=0.1,
    )
    assert seq_future.wait_until_done(timeout=GENERATE_TIMEOUT)
    _assert_generated_sequences(seq_future, expected_count=n_designs * n_sequences)


@pytest.mark.e2e
def test_e2e_proteinmpnn_generate_with_query_fanout(session: OpenProtein):
    """Validate list-valued `query` fan-out behavior for ProteinMPNN generation."""
    num_queries = 2
    n_sequences = 2
    n_unmasked_positions = 18

    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    # Use a full structure example recommended for ProteinMPNN inverse folding.
    query_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")

    # Remove N-terminal linker residues with undefined structure.
    sdn_linker = b"MHHHHHHSDN"
    query_protein = query_protein[len(sdn_linker) :]

    structured_positions = np.where(~query_protein.get_structure_mask())[0] + 1
    assert len(structured_positions) >= n_unmasked_positions
    rng = np.random.default_rng(0)

    query_ids = []
    for _ in range(num_queries):
        unmasked_positions = np.sort(
            rng.choice(
                structured_positions,
                size=n_unmasked_positions,
                replace=False,
            )
        )
        masked_query = query_protein.copy().mask_sequence_except_at(unmasked_positions)
        query = session.prompt.create_query(query=masked_query, force_structure=True)
        query_ids.append(query.id)

    assert len(query_ids) == num_queries

    future = session.models.proteinmpnn.generate(
        query=query_ids,
        num_samples=n_sequences,
        temperature=0.1,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    _assert_generated_sequences(future, expected_count=num_queries * n_sequences)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ["proteinmpnn", "solublempnn"])
def test_e2e_proteinmpnn_embed_multichain(session: OpenProtein, model_id: str):
    """Per-residue rows line up with the sequence: NaN at each `:`, zero where
    the structure has no residue."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    structure = Complex(chains=Complex.from_filepath(structure_filepath).get_proteins())
    proteins = list(structure.get_proteins().values())
    query = session.prompt.create_query(query=structure)
    sequence = ":".join(protein.sequence.decode() for protein in proteins)
    is_break = np.array([residue == ":" for residue in sequence])
    unresolved = np.zeros(len(sequence), dtype=bool)
    unresolved[~is_break] = np.concatenate(
        [protein.get_structure_mask() for protein in proteins]
    )
    model = getattr(session.models, model_id)
    dimension = model.metadata.dimension

    [per_residue] = model.embed(sequences=[sequence], query=query, reduction=None).wait(
        timeout=GENERATE_TIMEOUT
    )

    assert per_residue.shape == (len(sequence), dimension)
    assert np.isnan(per_residue[is_break]).all()
    assert np.isfinite(per_residue[~is_break]).all()
    assert (per_residue[unresolved] == 0).all()


@pytest.mark.e2e
def test_e2e_proteinmpnn_score_native_against_structure(session: OpenProtein):
    """ProteinMPNN scores the native sequence against its backbone, the same
    way every time."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    query_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")

    future = session.models.proteinmpnn.score(
        sequences=[query_protein.sequence, query_protein.sequence],
        query=query_protein,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    first, second = future.get()

    assert first == second
    per_residue_ll = first / len(query_protein.sequence)
    assert -4.0 <= per_residue_ll <= 0.0


@pytest.mark.e2e
def test_e2e_proteinmpnn_has_no_indel(session: OpenProtein):
    with pytest.raises(ValueError, match="has no indel"):
        session.models.proteinmpnn.indel(b"ACDEFGHIKLMNPQRSTVWY", insert="A")


@pytest.mark.e2e
def test_e2e_proteinmpnn_single_site_against_structure(session: OpenProtein):
    """ProteinMPNN scores every substitution of a resolved window, and every
    unmutated row scores the same."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    window = Protein.from_filepath(path=structure_filepath, chain_id="A")[10:40]
    assert not window.get_structure_mask().any()

    future = session.models.proteinmpnn.single_site(window.sequence, query=window)
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    scores = np.array(future.get(), dtype=float)

    assert len(scores) == 20 * len(window.sequence)
    assert np.isfinite(scores).all()


@pytest.mark.e2e
def test_e2e_esmif1_generate_basic(session: OpenProtein):
    """ESM-IF1 samples sequences conditioned on a backbone structure."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    n_sequences = 2
    query_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")

    future = session.models.esmif1.generate(
        query=query_protein,
        num_samples=n_sequences,
        temperature=1.0,
        random_seed=42,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    _assert_generated_sequences(future, expected_count=n_sequences)
    # Note: samples are not asserted to be distinct. With a fixed seed (and on a
    # well-determined backbone) ESM-IF1 can legitimately return identical samples,
    # so sequence diversity is not a reliable invariant here.


@pytest.mark.e2e
def test_e2e_esmif1_score_native_against_structure(session: OpenProtein):
    """ESM-IF1 scores the native sequence against its backbone."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    query_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")

    future = session.models.esmif1.score(
        sequences=[query_protein.sequence],
        query=query_protein,
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    [score] = future.get()

    # ESM-IF1 returns ll_fullseq: the summed log-likelihood over the sequence,
    # which scales with length. Normalize to a per-residue average before
    # bounding. Natural proteins score ~[-2.5, -1.0] per residue; allow [-4, 0].
    per_residue_ll = float(score) / len(query_protein.sequence)
    assert -4.0 <= per_residue_ll <= 0.0


@pytest.mark.e2e
def test_e2e_rfdiffusion_generate_basic(session: OpenProtein):
    """Standalone smoke test: rfdiffusion designs a structure de novo from a fully masked query."""
    n_designs = 1
    future = session.models.rfdiffusion.generate(
        query=Protein.from_expr(60), num_samples=n_designs
    )
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)

    designs = future.get()
    assert len(designs) == n_designs
    assert isinstance(designs[0], Complex)


@pytest.mark.e2e
def test_e2e_rfdiffusion_query_with_mask_structure(session: OpenProtein):
    """rfdiffusion fills in a partially-masked structure passed via `query`.

    RFdiffusion ignores sequence, so a position without structure has no sequence.
    """
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    base = Protein.from_filepath(path=structure_filepath, chain_id="A")
    sdn_linker = b"MHHHHHHSDN"
    base = base[len(sdn_linker) :]

    structured_positions = np.where(~base.get_structure_mask())[0] + 1
    assert len(structured_positions) >= 16

    rng = np.random.default_rng(1)
    kept_positions = np.sort(rng.choice(structured_positions, size=16, replace=False))
    masked = (
        base.copy()
        .mask_structure_except_at(kept_positions)
        .mask_sequence_except_at(kept_positions)
    )

    future = session.models.rfdiffusion.generate(query=masked, num_samples=1)
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    designs = future.get()
    assert len(designs) == 1
    assert isinstance(designs[0], Complex)


@pytest.mark.e2e
def test_e2e_boltzgen_query_with_mask_structure(session: OpenProtein):
    """boltzgen fills in a partially-masked structure passed via `query`."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    base = Protein.from_filepath(path=structure_filepath, chain_id="A")
    sdn_linker = b"MHHHHHHSDN"
    base = base[len(sdn_linker) :]

    structured_positions = np.where(~base.get_structure_mask())[0] + 1
    assert len(structured_positions) >= 16

    rng = np.random.default_rng(2)
    kept_positions = np.sort(rng.choice(structured_positions, size=16, replace=False))
    masked = base.copy().mask_structure_except_at(kept_positions)

    future = session.models.boltzgen.generate(query=masked, num_samples=1)
    assert future.wait_until_done(timeout=GENERATE_TIMEOUT)
    designs = future.get()
    assert len(designs) == 1
    assert isinstance(designs[0], Complex)


@pytest.mark.e2e
@pytest.mark.parametrize("fold_attr", ["boltz2", "protenix"])
def test_e2e_binder_design_full_flow_with_template(
    session: OpenProtein,
    fold_attr: str,
):
    """
    Closed-loop binder design: rfdiffusion -> proteinmpnn -> fold with both
    the target and the rfdiffusion-output binder passed as chain-mapped
    templates.
    """
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    available_fold_models = {model.id for model in session.fold.list_models()}
    fold_model = getattr(session.fold, fold_attr)
    if fold_model.id not in available_fold_models:
        pytest.skip(f"{fold_model.id} is not available in this backend")

    target = Protein.from_filepath(path=structure_filepath, chain_id="A")
    sdn_linker = b"MHHHHHHSDN"
    target = target[len(sdn_linker) :]
    assert len(target) > 30

    binder_len = 50
    # Build a query Complex with the target on chain A (full structure) and a
    # fully-masked binder placeholder on chain B (all-X sequence, no
    # structure). RFdiffusion designs the unstructured chain conditioned on
    # the structured one.
    query_complex = Complex(chains={"A": target, "B": Protein.from_expr(binder_len)})
    rfdiff_future = session.models.rfdiffusion.generate(
        query=query_complex, num_samples=1
    )
    assert rfdiff_future.wait_until_done(timeout=GENERATE_TIMEOUT)
    rfdiff_designs = rfdiff_future.get(output="structures")
    assert len(rfdiff_designs) == 1
    rfdiff_complex = rfdiff_designs[0]
    assert isinstance(rfdiff_complex, Complex)

    seq_future = session.models.proteinmpnn.generate(
        design=rfdiff_future,
        num_samples=1,
        temperature=0.1,
    )
    assert seq_future.wait_until_done(timeout=GENERATE_TIMEOUT)
    seq_results = seq_future.get(output="sequences")
    assert len(seq_results) == 1
    binder_sequence = seq_results[0].split(":")[1].encode()

    binder_protein = Protein(sequence=binder_sequence)
    # The fold call requires every input Protein to have an msa attached or
    # be flagged as single_sequence_mode. The target loaded from CIF doesn't
    # carry an msa and the freshly built binder Protein doesn't either.
    target_for_fold = target.copy()
    target_for_fold.msa = Protein.single_sequence_mode
    binder_protein.msa = Protein.single_sequence_mode
    complex_input = Complex(chains={"A": target_for_fold, "B": binder_protein})

    templates = [
        Template(template=rfdiff_complex, mapping={"A": "A", "B": "B"}),
    ]

    fold_future = fold_model.fold(sequences=[complex_input], templates=templates)
    assert fold_future.wait_until_done(timeout=GENERATE_TIMEOUT)
    fold_results = fold_future.get()
    assert len(fold_results) == 1
    structure = fold_results[0]
    assert structure
