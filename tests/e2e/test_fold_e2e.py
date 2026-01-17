"""End-to-end tests for the fold domain."""

from typing import List, Tuple

import pytest

from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.fold.future import FoldResultFuture
from openprotein.molecules import Complex, Ligand, Protein, Structure
from tests.utils.sequences import random_mutated_sequences

BASE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
E2E_TIMEOUT = 900  # 15 minutes for long-running E2E tests


@pytest.mark.e2e
def test_e2e_list_models(session: OpenProtein):
    """Test listing all available fold models."""
    models = session.fold.list_models()
    assert models
    assert len(models) > 0


@pytest.mark.e2e
def test_e2e_fold_with_esmfold(session: OpenProtein):
    """Test folding a single chain with ESMFold."""
    sequence = random_mutated_sequences(BASE_SEQUENCE, num_sequences=2)[1]
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
    protein_complex_with_msa: tuple[Complex, MSAFuture],
):
    """Test folding a multi-chain complex with AlphaFold2."""
    complex, _ = protein_complex_with_msa
    future = session.fold.alphafold2.fold(sequences=[complex])
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
    protein_complex_with_msa: tuple[Complex, MSAFuture],
):
    """Test folding a multi-chain complex with num_models > 1 with AlphaFold2."""
    complex, _ = protein_complex_with_msa
    future = session.fold.alphafold2.fold(sequences=[complex], num_models=2)
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
    protein_complex_with_msa: tuple[Complex, MSAFuture],
):
    """
    Test folding with Boltz-1.
    """
    complex, _ = protein_complex_with_msa
    future = session.fold.boltz1.fold(sequences=[complex])
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """
    Test folding with Boltz-1x.
    """
    complex, _ = protein_complex_with_msa
    future = session.fold.boltz1x.fold(sequences=[complex])
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """Test folding with a Boltz model and a cyclic protein."""
    complex, _ = protein_complex_with_msa
    complex.get_protein("A").cyclic = True  # set first to cyclic

    future = session.fold.boltz1.fold(sequences=[complex])
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """Test folding with a Boltz model using a 'bond' constraint."""
    complex, _ = protein_complex_with_msa
    constraints = [{"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 20, "CA"]}}]
    future = session.fold.boltz1.fold(sequences=[complex], constraints=constraints)
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """Test folding with a Boltz model using a 'pocket' constraint."""
    complex, _ = protein_complex_with_msa
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
    complex.set_chain("D", ligand)
    future = session.fold.boltz2.fold(sequences=[complex], constraints=constraints)
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """Test folding with a Boltz-1 model using a 'contact' constraint."""
    complex, _ = protein_complex_with_msa
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    future = session.fold.boltz2.fold(sequences=[complex], constraints=constraints)
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
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """
    Test folding with Boltz-2.
    """
    complex, _ = protein_complex_with_msa
    future = session.fold.boltz2.fold(sequences=[complex])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    structure = results[0]
    assert structure
    assert isinstance(structure, Structure)
    # affinity not found
    with pytest.raises(ValueError, match="affinity not found for request"):
        _ = future.get_affinity()


@pytest.mark.e2e
def test_e2e_fold_with_boltz2_with_affinity(
    session: OpenProtein,
    protein_complex_with_msa: Tuple[Complex, MSAFuture],
):
    """Test folding with Boltz-2 and requesting the affinity property."""
    complex, _ = protein_complex_with_msa
    ligand = Ligand(smiles="CCO")  # Use a different chain ID
    properties = [{"affinity": {"binder": "D"}}]
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    complex.set_chain("D", ligand)
    future = session.fold.boltz2.fold(
        sequences=[complex],
        properties=properties,
        constraints=constraints,
    )
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    affinity = future.get_affinity()
    assert affinity
