import numpy as np

import gemmi

import pytest

from openprotein.molecules import Structure, Complex, Protein, DNA, RNA, Ligand


@pytest.mark.parametrize("name", (None, "test"))
def test_serde_full_masked_structure_and_name(name: str | None):
    sequence = b"X" * 10
    structure = Structure([Complex({"A": Protein(sequence)})], name=name)
    assert structure.name == name
    assert structure[0].get_protein(chain_id="A").sequence == sequence
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    # TODO: when starting with name=None, after deser, name is "model" and not None
    #       this may not be an issue... but it would be ideal for the name to stay None
    if name is not None:
        assert new_structure.name == structure.name
    assert new_structure[0].get_protein(chain_id="A").sequence == sequence


def test_serde_multiple_fully_masked_structure():
    sequence_a = b"X" * 10
    sequence_c = b"A" * 17
    structure = Structure(
        [Complex({"A": Protein(sequence_a), "C": Protein(sequence_c)})]
    )
    assert structure[0].get_protein(chain_id="A").sequence == sequence_a
    assert structure[0].get_protein(chain_id="C").sequence == sequence_c
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert new_structure[0].get_protein(chain_id="A").sequence == sequence_a
    assert new_structure[0].get_protein(chain_id="C").sequence == sequence_c


def test_serde_multiple_complexes_with_full_masked_structure():
    sequence = b"X" * 10
    structure = Structure(
        [Complex({"A": Protein(sequence)}), Complex({"A": Protein(sequence)})]
    )
    with pytest.raises(Exception):
        _ = structure.to_string()


def test_can_serde_hemoglobin_cif():
    # tests 2 protein chains with multiplicity 2, and 1 ligand with multiplicity 4
    structure = Structure.from_filepath("tests/data/1A3N.cif")
    assert len(structure) == 1
    complex = structure[0]
    assert list(complex.get_proteins().keys()) == ["A", "B", "C", "D"]
    assert list(complex.get_dnas().keys()) == []
    assert list(complex.get_rnas().keys()) == []
    assert list(complex.get_ligands().keys()) == ["E", "F", "G", "H"]

    block = gemmi.cif.read_string(structure.to_string()).sole_block()
    gemmi_structure = gemmi.make_structure_from_block(block)
    assert len(gemmi_structure.entities) == 3

    structure = Structure.from_string(structure.to_string(), format="cif")
    assert len(structure) == 1
    complex = structure[0]
    assert list(complex.get_proteins().keys()) == ["A", "B", "C", "D"]
    assert list(complex.get_dnas().keys()) == []
    assert list(complex.get_rnas().keys()) == []
    assert list(complex.get_ligands().keys()) == ["E", "F", "G", "H"]


def test_can_serde_protein_dna_rna_cif():
    structure = Structure.from_filepath("tests/data/1MSW.cif")
    assert len(structure) == 1
    complex = structure[0]
    assert list(complex.get_proteins().keys()) == ["D"]
    assert list(complex.get_dnas().keys()) == ["A", "B"]
    assert list(complex.get_rnas().keys()) == ["C"]
    assert list(complex.get_ligands().keys()) == []

    block = gemmi.cif.read_string(structure.to_string()).sole_block()
    gemmi_structure = gemmi.make_structure_from_block(block)
    assert len(gemmi_structure.entities) == 4

    structure = Structure.from_string(structure.to_string(), format="cif")
    assert len(structure) == 1
    complex = structure[0]
    assert list(complex.get_proteins().keys()) == ["D"]
    assert list(complex.get_dnas().keys()) == ["A", "B"]
    assert list(complex.get_rnas().keys()) == ["C"]
    assert list(complex.get_ligands().keys()) == []


def test_copy():
    complex = Complex(
        chains={
            "A": Protein.from_expr(10),
            "B": DNA("ACGT"),
            "C": RNA("ACGU"),
            "D": Ligand(ccd="HEM"),
        }
    )
    structure = Structure(complexes=[complex])
    copy = structure.copy()
    assert len(copy) == len(structure)
    copy = copy[0]
    assert copy.get_chains().keys() == complex.get_chains().keys()
    # replacing chain in original complex should have no effect on copy
    old_protein_copy = complex.get_protein("A").copy()
    complex = complex.set_chain("A", Protein.from_expr(20))
    assert len(complex.get_protein("A")) != len(old_protein_copy)
    assert len(copy.get_protein("A")) == len(old_protein_copy)
    # modifying chain in original complex should have no effect on copy
    old_protein = complex.get_protein("A")
    old_protein.sequence = "A" * len(old_protein)
    assert len(complex.get_protein("A").sequence) != old_protein_copy.sequence
    assert len(copy.get_protein("A").sequence) != old_protein_copy.sequence


def test_can_serde_only_protein_pdb():
    structure = Structure.from_filepath("tests/data/1A3N.cif")
    assert len(structure) == 1
    structure[0] = Complex(structure[0].get_proteins())
    complex = structure[0]
    pdb_structure = Structure.from_string(
        structure.to_string(format="pdb"), format="pdb"
    )
    assert len(pdb_structure) == 1
    pdb_complex = pdb_structure[0]
    assert pdb_complex.get_chains().keys() == complex.get_chains().keys()
    assert pdb_complex.get_proteins().keys() == complex.get_proteins().keys()

    pdb_structure = Structure.from_filepath("tests/data/1GCM.pdb")
    assert len(pdb_structure) == 1
    pdb_complex = pdb_structure[0]
    assert list(pdb_complex.get_chains().keys()) == ["A", "B", "C"]
    assert list(pdb_complex.get_proteins().keys()) == ["A", "B", "C"]


def test_can_serde_some_but_no_all_chains_having_no_structure():
    # test 1/4 chains with structure all masked
    structure = Structure.from_filepath("tests/data/1A3N.cif")
    assert len(structure) == 1
    structure[0] = Complex(structure[0].get_proteins())
    complex = structure[0]
    _ = complex.get_protein("A").mask_structure()
    assert complex.get_protein(chain_id="A").get_structure_mask().all()
    assert not complex.get_protein(chain_id="B").get_structure_mask().all()
    assert not complex.get_protein(chain_id="C").get_structure_mask().all()
    assert not complex.get_protein(chain_id="D").get_structure_mask().all()
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert len(new_structure) == 1
    new_complex = new_structure[0]
    assert new_complex.get_chains().keys() == complex.get_chains().keys()
    assert new_complex.get_proteins().keys() == complex.get_proteins().keys()
    for chain_id in new_complex.get_proteins().keys():
        assert (
            new_complex.get_protein(chain_id=chain_id).sequence
            == complex.get_protein(chain_id=chain_id).sequence
        )
        assert (
            new_complex.get_protein(chain_id=chain_id).get_structure_mask()
            == complex.get_protein(chain_id=chain_id).get_structure_mask()
        ).all()
    # test 2/4 chains with structure all masked
    structure = Structure.from_filepath("tests/data/1A3N.cif")
    assert len(structure) == 1
    structure[0] = Complex(structure[0].get_proteins())
    complex = structure[0]
    _ = complex.get_protein("A").mask_structure()
    _ = complex.get_protein("D").mask_structure()
    assert complex.get_protein(chain_id="A").get_structure_mask().all()
    assert not complex.get_protein(chain_id="B").get_structure_mask().all()
    assert not complex.get_protein(chain_id="C").get_structure_mask().all()
    assert complex.get_protein(chain_id="D").get_structure_mask().all()
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert len(new_structure) == 1
    new_complex = new_structure[0]
    assert new_complex.get_chains().keys() == complex.get_chains().keys()
    assert new_complex.get_proteins().keys() == complex.get_proteins().keys()
    for chain_id in new_complex.get_proteins().keys():
        assert (
            new_complex.get_protein(chain_id=chain_id).sequence
            == complex.get_protein(chain_id=chain_id).sequence
        )
        assert (
            new_complex.get_protein(chain_id=chain_id).get_structure_mask()
            == complex.get_protein(chain_id=chain_id).get_structure_mask()
        ).all()


def test_serde_multiple_models():
    complex = Complex(Complex.from_filepath("tests/data/1A3N.cif").get_proteins())
    structure = Structure([complex.copy(), complex.copy().transform(t=np.ones(3))])
    assert (
        structure[1].get_protein(chain_id="A").coordinates
        != structure[0].get_protein(chain_id="A").coordinates
    ).all(), "coordinates of first and second complex should be different"
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert len(new_structure) == len(
        structure
    ), "new structure should have same number of complexes"
    assert (
        new_structure[1].get_protein(chain_id="A").coordinates
        != new_structure[0].get_protein(chain_id="A").coordinates
    ).all(), "coordinates of first and second complex should be different"
    for i in range(len(structure)):
        assert np.allclose(
            new_structure[i].get_protein(chain_id="A").coordinates,
            structure[i].get_protein(chain_id="A").coordinates,
            equal_nan=True,
        ), "new and old complex coordinates should match"

    structure[0].get_protein(chain_id="A").set_cyclic(True)
    assert structure[0].get_protein(chain_id="A").cyclic
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert new_structure[0].get_protein(chain_id="A").cyclic

    structure[1].get_protein(chain_id="A").set_cyclic(True)
    assert structure[1].get_protein(chain_id="A").cyclic
    with pytest.raises(NotImplementedError):
        new_structure = Structure.from_string(structure.to_string(), format="cif")

    structure[1].get_protein(chain_id="A").set_cyclic(False)
    assert not structure[1].get_protein(chain_id="A").cyclic
    structure[0].get_protein(chain_id="A").set_msa(Protein.single_sequence_mode)
    assert structure[0].get_protein(chain_id="A").msa == Protein.single_sequence_mode
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert new_structure[0].get_protein(chain_id="A").cyclic
    assert (
        new_structure[0].get_protein(chain_id="A").msa == Protein.single_sequence_mode
    )

    structure[1].get_protein(chain_id="A").set_msa(Protein.single_sequence_mode)
    assert structure[1].get_protein(chain_id="A").msa == Protein.single_sequence_mode
    with pytest.raises(NotImplementedError):
        new_structure = Structure.from_string(structure.to_string(), format="cif")

    structure[0].get_protein(chain_id="A").set_msa(None)
    assert structure[0].get_protein(chain_id="A").msa is None
    with pytest.raises(NotImplementedError):
        new_structure = Structure.from_string(structure.to_string(), format="cif")

    structure[1].get_protein(chain_id="A").set_msa(None)
    assert structure[1].get_protein(chain_id="A").msa is None
    new_structure = Structure.from_string(structure.to_string(), format="cif")
    assert structure[0].get_protein(chain_id="A").msa is None
    assert structure[1].get_protein(chain_id="A").msa is None
