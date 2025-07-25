"""End-to-end tests for the fold domain."""

from typing import List, Tuple

import pytest

from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.chains import Ligand
from openprotein.fold.future import FoldComplexResultFuture, FoldResultFuture
from openprotein.protein import Protein
from tests.utils.sequences import generate_mutated_sequences

BASE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
E2E_TIMEOUT = 600  # 10 minutes for long-running E2E tests


@pytest.mark.e2e
def test_e2e_list_models(api_session: OpenProtein):
    """Test listing all available fold models."""
    models = api_session.fold.list_models()
    assert models
    assert len(models) > 0


@pytest.mark.e2e
def test_e2e_fold_with_esmfold(api_session: OpenProtein):
    """Test folding a single chain with ESMFold."""
    sequence = generate_mutated_sequences(BASE_SEQUENCE, num_sequences=2)[1]
    future = api_session.fold.esmfold.fold(sequences=[sequence])
    assert isinstance(future, FoldResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    results = future.get()
    assert isinstance(results, list)
    assert len(results) == 1
    pdb_string = results[0][1].decode("utf-8")
    assert pdb_string
    assert "ATOM" in pdb_string


@pytest.mark.e2e
def test_e2e_fold_with_alphafold2_complex(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding a multi-chain complex with AlphaFold2."""
    proteins, msa_future = protein_complex_with_msa
    future = api_session.fold.alphafold2.fold(proteins=proteins)
    assert isinstance(future, FoldComplexResultFuture)
    # The MSA is already waited on in the fixture, so we just wait for the fold
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz1(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """
    Test folding with Boltz-1.
    """
    proteins, _ = protein_complex_with_msa
    future = api_session.fold.boltz1.fold(proteins=proteins)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz1x(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """
    Test folding with Boltz-1x.
    """
    proteins, _ = protein_complex_with_msa
    future = api_session.fold.boltz1x.fold(proteins=proteins)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz_cyclic_protein(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding with a Boltz model and a cyclic protein."""
    proteins, _ = protein_complex_with_msa
    proteins[0].cyclic = True  # set first to cyclic

    future = api_session.fold.boltz1.fold(proteins=proteins)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz_bond_constraint(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding with a Boltz model using a 'bond' constraint."""
    proteins, _ = protein_complex_with_msa
    constraints = [{"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 20, "CA"]}}]
    future = api_session.fold.boltz1.fold(proteins=proteins, constraints=constraints)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz_pocket_constraint(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding with a Boltz model using a 'pocket' constraint."""
    proteins, _ = protein_complex_with_msa
    ligand = Ligand(chain_id="D", smiles="CCO")  # Use a different chain ID
    constraints = [
        {
            "pocket": {
                "binder": "D",
                "contacts": [["A", 15], ["B", 22]],
                "max_distance": 10.0,
            }
        }
    ]
    future = api_session.fold.boltz1.fold(
        proteins=proteins, ligands=[ligand], constraints=constraints
    )
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz_contact_constraint(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding with a Boltz-1 model using a 'contact' constraint."""
    proteins, _ = protein_complex_with_msa
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    future = api_session.fold.boltz1.fold(proteins=proteins, constraints=constraints)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")


@pytest.mark.e2e
def test_e2e_fold_with_boltz2(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """
    Test folding with Boltz-2.
    """
    proteins, _ = protein_complex_with_msa
    future = api_session.fold.boltz2.fold(proteins=proteins)
    assert isinstance(future, FoldComplexResultFuture)
    assert future.wait_until_done(timeout=E2E_TIMEOUT)
    result_bytes = future.get()
    assert result_bytes and "ATOM" in result_bytes.decode("utf-8")
    # affinity not found
    with pytest.raises(ValueError, match="affinity not found for request"):
        _ = future.affinity


@pytest.mark.e2e
def test_e2e_fold_with_boltz2_with_affinity(
    api_session: OpenProtein,
    protein_complex_with_msa: Tuple[List[Protein], MSAFuture],
):
    """Test folding with Boltz-2 and requesting the affinity property."""
    proteins, _ = protein_complex_with_msa
    ligand = Ligand(chain_id="D", smiles="CCO")  # Use a different chain ID
    properties = [{"affinity": {"binder": "D"}}]
    constraints = [
        {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
    ]
    future = api_session.fold.boltz2.fold(
        proteins=proteins,
        ligands=[ligand],
        properties=properties,
        constraints=constraints,
    )
    assert isinstance(future, FoldComplexResultFuture)
