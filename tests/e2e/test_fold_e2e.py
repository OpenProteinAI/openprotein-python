"""End-to-end tests for the fold domain."""

from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.errors import APIError, HTTPError
from openprotein.fold.future import FoldResultFuture
from openprotein.molecules import Complex, Ligand, Protein, Structure
from openprotein.molecules.template import Template
from tests.e2e.config import scaled_timeout

E2E_TIMEOUT = scaled_timeout(1.5)


@pytest.fixture
def fold_complex(protein_complex_with_msa: tuple[Complex, MSAFuture]) -> Complex:
    """Return a per-test copy to avoid mutating the shared session fixture."""
    complex, _ = protein_complex_with_msa
    return deepcopy(complex)


@pytest.mark.e2e
def test_e2e_list_models(session: OpenProtein):
    """Test listing all available fold models."""
    models = session.fold.list_models()
    assert models
    assert len(models) > 0


@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_id,uses_multichain_entity",
    [
        ("alphafold2", True),
        ("boltz-1", True),
        ("boltz-1x", True),
        ("boltz-2", True),
        ("rosettafold-3", True),
        ("esmfold", True),
        ("minifold", False),  # monomer only; validate over-limit handling on one chain
    ],
)
def test_e2e_fold_rejects_total_sequence_length_above_model_max(
    session: OpenProtein,
    model_id: str,
    uses_multichain_entity: bool,
):
    """Test fold API rejects sequence/entity lengths above the model metadata max."""
    available_model_ids = {model.id for model in session.fold.list_models()}
    if model_id not in available_model_ids:
        pytest.skip(f"{model_id} is not available in this backend")

    model = session.fold.get_model(model_id)
    max_len = model.metadata.max_sequence_length
    if max_len is None or max_len < 2:
        pytest.skip("Model does not report a usable max_sequence_length")

    if uses_multichain_entity:
        chain_a_len = max_len // 2
        chain_b_len = max_len - chain_a_len + 1  # total length is max_len + 1

        chain_a = Protein("A" * chain_a_len)
        chain_a.msa = Protein.single_sequence_mode
        chain_b = Protein("A" * chain_b_len)
        chain_b.msa = Protein.single_sequence_mode
        input_sequence = Complex(chains={"A": chain_a, "B": chain_b})

        # Each chain is <= max_len, so this validates total entity length handling.
        assert len(chain_a) <= max_len
        assert len(chain_b) <= max_len
        assert len(chain_a) + len(chain_b) == max_len + 1
    else:
        chain = Protein("A" * (max_len + 1))
        chain.msa = Protein.single_sequence_mode
        input_sequence = chain
        assert len(chain) == max_len + 1

    with pytest.raises(HTTPError, match="Status code"):
        model.fold(sequences=[input_sequence])


@pytest.mark.e2e
@pytest.mark.parametrize(
    "invalid_chain_id",
    [
        "a",  # chain IDs should be uppercase
        "A B",  # spaces are invalid
        "A*",  # symbols are invalid
    ],
)
def test_e2e_fold_rejects_invalid_chain_ids(
    session: OpenProtein,
    invalid_chain_id: str,
):
    """Test fold API rejects entities containing invalid chain IDs."""
    model = session.fold.boltz1

    invalid_chain = Protein("A" * 16)
    invalid_chain.msa = Protein.single_sequence_mode
    valid_chain = Protein("A" * 16)
    valid_chain.msa = Protein.single_sequence_mode
    complex_input = Complex(chains={invalid_chain_id: invalid_chain, "B": valid_chain})

    with pytest.raises(HTTPError, match="Status code"):
        model.fold(sequences=[complex_input])


@pytest.mark.e2e
def test_e2e_fold_with_esmfold(
    session: OpenProtein,
    mutated_sequences: Callable[..., list[str]],
):
    """Test folding a single chain with ESMFold."""
    sequence = mutated_sequences(num_sequences=2)[1]
    future = session.fold.esmfold.fold(sequences=[sequence])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_alphafold2_complex(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding a multi-chain complex with AlphaFold2."""
    future = session.fold.alphafold2.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    # The MSA is already waited on in the fixture, so we just wait for the fold
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_alphafold2_complex_with_num_models(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding a multi-chain complex with num_models > 1 with AlphaFold2."""
    future = session.fold.alphafold2.fold(sequences=[fold_complex], num_models=2)
    assert isinstance(future, FoldResultFuture)
    # The MSA is already waited on in the fixture, so we just wait for the fold
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)
    assert len(structure) == 2


@pytest.mark.e2e
def test_e2e_fold_with_boltz1(
    session: OpenProtein,
    fold_complex: Complex,
):
    """
    Test folding with Boltz-1.
    """
    future = session.fold.boltz1.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz1x(
    session: OpenProtein,
    fold_complex: Complex,
):
    """
    Test folding with Boltz-1x.
    """
    future = session.fold.boltz1x.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz_cyclic_protein(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with a Boltz model and a cyclic protein."""
    fold_complex.get_protein("A").cyclic = True  # set first to cyclic

    future = session.fold.boltz1.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz_bond_constraint(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with a Boltz model using a 'bond' constraint."""
    constraints = [{"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 20, "CA"]}}]
    future = session.fold.boltz1.fold(sequences=[fold_complex], constraints=constraints)
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz_pocket_constraint(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with a Boltz model using a 'pocket' constraint."""
    ligand = Ligand(smiles="CCO")
    constraints = [
        {
            "pocket": {
                "binder": "D",
                "contacts": [["A", 15], ["B", 22]],
                "max_distance": 10.0,
            }
        }
    ]
    fold_complex.set_chain("D", ligand)
    future = session.fold.boltz2.fold(sequences=[fold_complex], constraints=constraints)
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz_contact_constraint(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with a Boltz-1 model using a 'contact' constraint."""
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    future = session.fold.boltz2.fold(sequences=[fold_complex], constraints=constraints)
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_boltz2(
    session: OpenProtein,
    fold_complex: Complex,
):
    """
    Test folding with Boltz-2.
    """
    future = session.fold.boltz2.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)
    # affinity not found
    with pytest.raises(APIError, match="affinity not found for request"):
        _ = future.get_affinity()


@pytest.mark.e2e
def test_e2e_fold_with_boltz2_with_affinity(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with Boltz-2 and requesting the affinity property."""
    ligand = Ligand(smiles="CCO")  # Use a different chain ID
    properties = [{"affinity": {"binder": "D"}}]
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    fold_complex.set_chain("D", ligand)
    future = session.fold.boltz2.fold(
        sequences=[fold_complex],
        properties=properties,
        constraints=constraints,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    affinity = future.get_affinity()
    assert affinity


@pytest.mark.e2e
def test_e2e_fold_with_protenix(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding a multi-chain complex with Protenix."""
    future = session.fold.protenix.fold(sequences=[fold_complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


@pytest.mark.e2e
def test_e2e_fold_with_protenix_custom_params(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Test folding with Protenix using custom diffusion/recycle/step parameters."""
    future = session.fold.protenix.fold(
        sequences=[fold_complex],
        diffusion_samples=2,
        num_recycles=3,
        num_steps=50,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)


# ----------------------------------------------------------------------- #
# Templates with `index` scoping (binder-design refold ergonomics) and
# batch-extras retrieval. These exercise the new fold API surface end to
# end against a live backend.
# ----------------------------------------------------------------------- #


@pytest.fixture
def template_protein() -> Protein:
    """A Protein carrying coordinates, suitable for use as a fold template."""
    cif_path = Path("tests/data/test_protein.cif")
    return Protein.from_filepath(cif_path, chain_id="A", verbose=False)


@pytest.mark.e2e
def test_e2e_boltz2_indexed_templates(
    session: OpenProtein,
    fold_complex: Complex,
    template_protein: Protein,
):
    """Two templates each scoped to a different sequence in the batch.

    Verifies the wire path: per-sequence template scoping (`index`) survives
    serialization, the server validates and resolves both templates, and
    each batch entry folds successfully.
    """
    # Two-sequence batch — index 0 sees template_protein via Template A,
    # index 1 sees the same protein but via Template B (chain mapping
    # remains "A" so we're really testing that `index` carves the batch
    # without changing the underlying template).
    batch = [deepcopy(fold_complex), deepcopy(fold_complex)]
    templates = [
        Template(template=template_protein, mapping="A", index=[0]),
        Template(template=template_protein, mapping="A", index=[1]),
    ]
    future = session.fold.boltz2.fold(sequences=batch, templates=templates)
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert len(results) == 2
    for s in results:
        assert isinstance(s, Structure)


@pytest.mark.e2e
def test_e2e_boltz2_baseline_plus_indexed_template(
    session: OpenProtein,
    fold_complex: Complex,
    template_protein: Protein,
):
    """Baseline template (no `index`) + an extra template scoped to one sequence.

    Confirms the mixed shape: the server applies the baseline to every
    sequence and the indexed template only to the sequence(s) it names.
    """
    batch = [deepcopy(fold_complex), deepcopy(fold_complex)]
    templates = [
        Template(template=template_protein, mapping="A"),  # applies to all
        Template(template=template_protein, mapping="A", index=[1]),  # only seq 1
    ]
    future = session.fold.boltz2.fold(sequences=batch, templates=templates)
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert len(results) == 2
    for s in results:
        assert isinstance(s, Structure)


@pytest.mark.e2e
def test_e2e_boltz2_index_out_of_range_rejected(
    session: OpenProtein,
    fold_complex: Complex,
    template_protein: Protein,
):
    """Server rejects a template with an index past the batch size."""
    batch = [deepcopy(fold_complex)]  # one sequence → only index 0 is valid
    templates = [
        Template(template=template_protein, mapping="A", index=[0, 5]),
    ]
    with pytest.raises(HTTPError, match="Status code"):
        session.fold.boltz2.fold(sequences=batch, templates=templates)


@pytest.mark.e2e
def test_e2e_fold_batch_pae_retrieval(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Batch PAE endpoint returns one stacked [N, ...] ndarray for the job."""
    batch = [deepcopy(fold_complex), deepcopy(fold_complex)]
    future = session.fold.boltz2.fold(sequences=batch)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)

    pae_batch = future.get_pae_batch()
    assert isinstance(pae_batch, np.ndarray)
    assert pae_batch.ndim >= 2
    assert pae_batch.shape[0] == 2

    # Sanity-check the per-unit endpoint agrees on the leading dim.
    pae_units = future.get_pae()
    assert len(pae_units) == 2


@pytest.mark.e2e
def test_e2e_fold_ipae_per_unit_and_batch(
    session: OpenProtein,
    fold_complex: Complex,
):
    """iPAE is a synthetic scalar per unit; batch returns shape [N]."""
    batch = [deepcopy(fold_complex), deepcopy(fold_complex)]
    future = session.fold.boltz2.fold(sequences=batch)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)

    ipae_batch = future.get_ipae_batch()
    assert isinstance(ipae_batch, np.ndarray)
    assert ipae_batch.shape == (2,)

    ipae_units = future.get_ipae()
    assert len(ipae_units) == 2
    for arr in ipae_units:
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (1,)
    # Per-unit values should agree with the batch values (NaN-safe compare).
    flat = np.asarray([float(arr[0]) for arr in ipae_units])
    np.testing.assert_allclose(flat, ipae_batch, equal_nan=True, rtol=1e-5)


@pytest.mark.e2e
def test_e2e_fold_confidence_batch(
    session: OpenProtein,
    fold_complex: Complex,
):
    """Confidence batch endpoint returns a length-N list (one fan-in HTTP call)."""
    batch = [deepcopy(fold_complex), deepcopy(fold_complex)]
    future = session.fold.boltz2.fold(sequences=batch)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)

    confidence_batch = future.get_confidence_batch()
    assert isinstance(confidence_batch, list)
    assert len(confidence_batch) == 2
    for entry in confidence_batch:
        # Every unit completed in this small job, so no Nones expected.
        assert entry is not None


@pytest.mark.e2e
@pytest.mark.parametrize("fold_attr", ["boltz2", "protenix"])
def test_e2e_fold_with_templates(
    session: OpenProtein,
    fold_attr: str,
):
    """Fold a sequence with a known structure passed as a single-chain template."""
    structure_filepath = Path("tests/data/8bo9.cif")
    if not structure_filepath.exists():
        pytest.skip(f"Missing test structure file: {structure_filepath}")

    available_fold_models = {model.id for model in session.fold.list_models()}
    fold_model = getattr(session.fold, fold_attr)
    if fold_model.id not in available_fold_models:
        pytest.skip(f"{fold_model.id} is not available in this backend")

    template_protein = Protein.from_filepath(path=structure_filepath, chain_id="A")
    sdn_linker = b"MHHHHHHSDN"
    template_protein = template_protein[len(sdn_linker) :]

    # Wrap as Complex so the Template mapping="A" form is valid.
    # A standalone Protein target rejects any explicit mapping per
    # Template.validate_for_target.
    fold_chain = Protein(sequence=template_protein.sequence)
    fold_chain.msa = Protein.single_sequence_mode
    fold_input = Complex(chains={"A": fold_chain})
    templates = [Template(template=template_protein, mapping="A")]

    future = fold_model.fold(sequences=[fold_input], templates=templates)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)
