import re

import gemmi
import numpy as np
import pytest

from openprotein.molecules import Complex, Protein, Structure

MODEL_NUM_TAG = "_atom_site.pdbx_PDB_model_num"


@pytest.fixture(scope="module")
def complex_() -> Complex:
    return Complex(Complex.from_filepath("tests/data/1A3N.cif").get_proteins())


def _cif_model_nums(cif: str) -> list[str]:
    block = gemmi.cif.read_string(cif).sole_block()
    return list(block.find_values(MODEL_NUM_TAG))


def _pdb_model_nums(pdb: str) -> list[int]:
    return [int(m) for m in re.findall(r"^MODEL\s+(\d+)", pdb, flags=re.MULTILINE)]


def _with_zero_based_cif_model_nums(cif: str) -> str:
    doc = gemmi.cif.read_string(cif)
    column = doc.sole_block().find_values(MODEL_NUM_TAG)
    for i, num in enumerate(column):
        column[i] = str(int(num) - 1)
    return doc.as_string()


def _two_model_structure(complex_: Complex) -> Structure:
    return Structure([complex_.copy(), complex_.copy().transform(t=np.ones(3))])


def test_complex_cif_atoms_all_model_one(complex_):
    nums = _cif_model_nums(complex_.to_string("cif"))
    assert len(nums) > 0
    assert set(nums) == {"1"}


def test_single_complex_structure_cif_atoms_all_model_one(complex_):
    nums = _cif_model_nums(Structure([complex_]).to_string("cif"))
    assert len(nums) > 0
    assert set(nums) == {"1"}


def test_protein_cif_atoms_all_model_one(complex_):
    protein = complex_.get_protein("A")
    nums = _cif_model_nums(protein.to_string("cif"))
    assert len(nums) > 0
    assert set(nums) == {"1"}


@pytest.mark.parametrize("n_models", (2, 3))
def test_multi_model_cif_is_numbered_one_to_n(complex_, n_models):
    structure = Structure(
        [complex_.copy().transform(t=np.full(3, float(i))) for i in range(n_models)]
    )
    nums = _cif_model_nums(structure.to_string("cif"))
    assert set(nums) == {str(i) for i in range(1, n_models + 1)}
    assert nums == sorted(nums)


def test_single_model_pdb_is_model_one(complex_):
    pdb = complex_.to_string("pdb")
    assert _pdb_model_nums(pdb) in ([], [1])
    assert [m.num for m in gemmi.read_pdb_string(pdb)] == [1]


def test_multi_model_pdb_is_numbered_one_to_n(complex_):
    structure = _two_model_structure(complex_)
    pdb = structure.to_string("pdb")
    assert _pdb_model_nums(pdb) == [1, 2]


@pytest.mark.parametrize("format", ("cif", "pdb"))
def test_complex_round_trip(complex_, format):
    new = Complex.from_string(complex_.to_string(format), format=format)
    assert np.allclose(
        new.get_protein("A").coordinates,
        complex_.get_protein("A").coordinates,
        atol=1e-2,
        equal_nan=True,
    )


@pytest.mark.parametrize("format", ("cif", "pdb"))
def test_multi_model_round_trip(complex_, format):
    structure = _two_model_structure(complex_)
    new = Structure.from_string(structure.to_string(format), format=format)
    assert len(new) == 2
    for i in range(2):
        assert np.allclose(
            new[i].get_protein("A").coordinates,
            structure[i].get_protein("A").coordinates,
            atol=1e-2,
            equal_nan=True,
        )


def test_reads_single_model_zero_numbered_cif(complex_):
    cif = _with_zero_based_cif_model_nums(complex_.to_string("cif"))
    assert set(_cif_model_nums(cif)) == {"0"}
    new = Complex.from_string(cif, format="cif", model_idx=0)
    assert np.allclose(
        new.get_protein("A").coordinates,
        complex_.get_protein("A").coordinates,
        equal_nan=True,
    )
    protein = Protein.from_string(cif, format="cif", chain_id="A", model_idx=0)
    assert np.allclose(
        protein.coordinates, complex_.get_protein("A").coordinates, equal_nan=True
    )


def test_reads_multi_model_zero_numbered_cif(complex_):
    structure = _two_model_structure(complex_)
    cif = _with_zero_based_cif_model_nums(structure.to_string("cif"))
    assert set(_cif_model_nums(cif)) == {"0", "1"}
    new = Structure.from_string(cif, format="cif")
    assert len(new) == 2
    for i in range(2):
        assert np.allclose(
            new[i].get_protein("A").coordinates,
            structure[i].get_protein("A").coordinates,
            atol=1e-2,
            equal_nan=True,
        )
    second = Complex.from_string(cif, format="cif", model_idx=1)
    assert np.allclose(
        second.get_protein("A").coordinates,
        structure[1].get_protein("A").coordinates,
        atol=1e-2,
        equal_nan=True,
    )
