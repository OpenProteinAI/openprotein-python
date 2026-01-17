import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Type

import numpy as np

import pytest

import gemmi

import openprotein.common.residue_contants as RC
from openprotein.molecules import Protein, Binding
from openprotein.molecules.protein import _ATOM_TYPES, _SIDE_CHAIN_ATOM_IDXS


TEST_SEQUENCE = b"ARNMK"
TEST_COORDINATES = -np.arange(5 * 37 * 3, dtype=np.float32).reshape(5, 37, 3)
TEST_PLDDT = 50 + np.arange(5, dtype=np.float32)
TEST_STRUCTURE_PATHS = [
    "tests/data/test_protein_minimal.pdb",
    "tests/data/test_protein.cif",
]


def protein_with_structure():
    length = np.random.randint(1, 8)
    # TODO: handle optional
    amino_acids = list(RC.restype_1to3.keys()) + ["X"]  # ["X", "?"]
    protein = Protein(
        sequence="".join(
            [amino_acids[i] for i in np.random.randint(0, len(amino_acids), length)]
        )
    )
    protein = protein._set_coordinates(
        np.random.random(protein.coordinates.shape).astype(np.float32)
    )
    return protein


def test_protein_serde_version():
    protein = protein_with_structure()
    block = gemmi.cif.read_string(protein.to_string()).sole_block()
    assert block.find_value("_openprotein_version") == "1"


def test_protein_serde():
    # TODO: when starting with name=None, after deser, name is "model" and not None
    #       this may not be an issue... but it would be ideal for the name to stay None
    protein = Protein(b"ARNMKIP")
    new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
    assert new.sequence == protein.sequence
    with pytest.raises(Exception):
        _ = Protein.from_string(protein.to_string(), format="cif", chain_id="B")

    protein = protein_with_structure()
    new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
    assert new.sequence == protein.sequence
    # test that we can read pdb too
    new = Protein.from_string(
        gemmi.make_structure_from_block(
            gemmi.cif.read_string(protein.to_string()).sole_block()
        ).make_pdb_string(),
        format="pdb",
        chain_id="A",
        use_bfactor_as_plddt=False,
    )
    assert new.sequence == protein.sequence


@pytest.mark.parametrize("cyclic", (False, True))
def test_protein_serde_cyclic(cyclic: bool):
    for protein in [Protein(b"MKIXX"), protein_with_structure()]:
        protein.cyclic = cyclic
        assert protein.cyclic == cyclic
        new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
        assert new.cyclic == protein.cyclic


@pytest.mark.parametrize("msa", (None, str(uuid.uuid4()), Protein.NullMSA))
def test_protein_serde_msa_id(msa: str | None | Type[Protein.NullMSA]):
    for protein in [Protein(b"XXMKI"), protein_with_structure()]:
        protein = protein_with_structure()
        protein.msa = msa
        assert protein.msa == msa
        new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
        assert new.msa == protein.msa


@pytest.mark.parametrize("group", (0, 1, 2))
def test_protein_serde_group(group: int):
    protein = protein_with_structure()
    protein = protein.set_group_at(positions=[1], value=group)
    assert np.array_equal(
        protein.get_group_at(positions=[1]), np.array([group], dtype=int)
    )
    new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
    assert np.array_equal(
        new.get_group_at(positions=[1]), protein.get_group_at(positions=[1])
    )


@pytest.mark.parametrize(
    # TODO: test both str and enum
    "binding",
    (Binding.UNKNOWN.value, Binding.NOT_BINDING.value, Binding.BINDING.value),
)
def test_protein_serde_binding(binding: str):
    protein = protein_with_structure()
    protein = protein.set_binding_at(positions=[1], value=binding)
    assert np.array_equal(
        protein.get_binding_at(positions=[1]), np.array([binding], dtype="<U1")
    )
    new = Protein.from_string(protein.to_string(), format="cif", chain_id="A")
    assert np.array_equal(
        new.get_binding_at(positions=[1]),
        protein.get_binding_at(positions=[1]),
    )


@pytest.mark.parametrize(
    "range_str,seq_str",
    (("1", "X"), ("2", "XX"), ("1..1", "X"), ("3..3", "XXX"), ("1..3", "X??")),
)
@pytest.mark.parametrize("left", (True, False))
def test_add_range(range_str: str, seq_str: str, left: bool):
    protein = Protein(sequence="ARNMKIP")
    if left:
        new = range_str + protein
        assert new.sequence == seq_str.encode() + protein.sequence
    else:
        new = protein + range_str
        assert new.sequence == protein.sequence + seq_str.encode()


@pytest.mark.parametrize(
    "range_str",
    ("0", "-1", "-2", "2..1", "1..0", "0..0", "-3..-3", "-1,-2", "-2..-1", -10, 0, 1),
)
@pytest.mark.parametrize("left", (True, False))
def test_add_invalid_range(range_str: Any, left: bool):
    protein = Protein(sequence="MKIPARN")
    with pytest.raises(Exception):
        if left:
            _ = range_str + protein
        else:
            _ = protein + range_str


def test_can_parse_boltzgen_entity_one_letter_missing_residues():
    protein_a = Protein.from_filepath(
        "tests/data/boltzgen_entity_one_letter_missing_residues.cif", chain_id="A"
    )
    assert len(protein_a) == 203 - 2  # last label_seq is 203, missing residues 1 and 2
    protein_b = Protein.from_filepath(
        "tests/data/boltzgen_entity_one_letter_missing_residues.cif", chain_id="B"
    )
    assert len(protein_b) == 127 - 1  # last label_seq is 127, missing residue 1
    assert (
        b"X" not in protein_b.sequence
    ), "missing residues in _atom_site should be inferred form _entity_poly_seq"


def test_can_parse_9bkq_assembly2():
    protein = Protein.from_filepath("tests/data/9bkq-assembly2.cif", chain_id="B")
    assert len(protein) == 203


@pytest.mark.parametrize("side_chain_only", (False, True))
def test_protein_mask_structure_with_empty_positions(side_chain_only: bool):
    protein = protein_with_structure()
    new_protein = protein.copy().mask_structure_at([], side_chain_only=side_chain_only)
    assert np.array_equal(new_protein.coordinates, protein.coordinates, equal_nan=True)
    new_protein = protein.copy().mask_structure_except_at(
        [], side_chain_only=side_chain_only
    )
    masked_atom_idxs = (
        np.arange(len(_ATOM_TYPES)) if not side_chain_only else _SIDE_CHAIN_ATOM_IDXS
    )
    non_masked_atom_idxs = np.array(
        [i for i in range(len(_ATOM_TYPES)) if i not in masked_atom_idxs], dtype=int
    )
    assert np.isnan(new_protein.coordinates[:, masked_atom_idxs]).all()
    assert np.array_equal(
        new_protein.coordinates[:, non_masked_atom_idxs],
        protein.coordinates[:, non_masked_atom_idxs],
        equal_nan=True,
    )


@pytest.fixture(params=TEST_STRUCTURE_PATHS)
def structure_path(request) -> Iterator[str]:
    yield request.param


def test_protein(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    assert protein.sequence == TEST_SEQUENCE
    assert np.array_equal(protein.coordinates, TEST_COORDINATES)
    assert np.array_equal(protein.plddt, TEST_PLDDT)


def assert_proteins_equal(x: Protein, y: Protein):
    # NB: this function doesn't check if names are equal
    assert x.sequence == y.sequence
    assert np.array_equal(x.coordinates, y.coordinates, equal_nan=True)
    assert np.array_equal(x.plddt, y.plddt, equal_nan=True)


def test_partially_masked_sequence(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    protein.sequence = "AXXMX"
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_fully_masked_sequence(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    protein.sequence = "X" * len(protein.sequence)
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_partially_masked_plddt(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    _plddt = protein.plddt.copy()
    _plddt[[0, 1]] = np.nan
    protein._plddt = _plddt.copy()
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_fully_masked_plddt(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    protein._plddt = np.full_like(protein.plddt, np.nan)
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_partially_masked_coordinates(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    # NB: for each position, CA coordinates nan -> plddt nan
    _coordinates, _plddt = protein.coordinates.copy(), protein.plddt.copy()
    _coordinates[[0, 2, 3]], _plddt[[0, 2, 3]] = np.nan, np.nan
    protein = protein._set_coordinates(_coordinates)
    protein._plddt = _plddt.copy()
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_partially_masked_coordinates_non_ca_atom_only(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    # NB: for each position, CA coordinates nan -> plddt nan
    _coordinates = protein.coordinates.copy()
    _coordinates[[1, 4], 0] = np.nan
    protein = protein._set_coordinates(_coordinates)
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def test_partially_masked_coordinates_ca_atom_only(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    # NB: for each position, CA coordinates nan -> plddt nan
    _coordinates = protein.coordinates.copy()
    _coordinates[[1, 4], 1] = np.nan
    protein = protein._set_coordinates(_coordinates)
    with pytest.raises(AssertionError):
        protein.make_cif_string()


def test_mixed_partial_masking(structure_path: str):
    protein = Protein.from_filepath(structure_path, chain_id="A")
    protein.sequence = "ARXMX"
    # NB: for each position, CA coordinates nan -> plddt nan
    _coordinates, _plddt = protein.coordinates.copy(), protein.plddt.copy()
    _coordinates[[1, 4], 0], _plddt[[0, 1]] = np.nan, np.nan
    protein = protein._set_coordinates(_coordinates)
    protein._plddt = _plddt.copy()
    new_protein = Protein.from_string(
        protein.make_cif_string(), format="cif", chain_id="A"
    )
    assert_proteins_equal(protein, new_protein)


def main():
    protein = Protein(name="test", sequence=TEST_SEQUENCE)._set_coordinates(
        TEST_COORDINATES
    )
    protein._plddt = TEST_PLDDT.copy()
    block = gemmi.cif.read_string(protein.to_string()).sole_block()
    structure = gemmi.make_structure_from_block(block)
    pdb_path = Path("tests/data/test_protein_minimal.pdb")
    cif_path = Path("tests/data/test_protein.cif")
    if not pdb_path.is_file():
        pdb_path.write_text(structure.make_minimal_pdb())
    if not cif_path.is_file():
        cif_path.write_text(structure.make_mmcif_document().as_string())
    assert structure.make_minimal_pdb() == open(pdb_path, "r").read()
    assert structure.make_mmcif_document().as_string() == open(cif_path, "r").read()


if __name__ == "__main__":
    main()
