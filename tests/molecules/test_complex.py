from openprotein.molecules import Complex, Protein, DNA, RNA, Ligand

import gemmi

import pytest


@pytest.mark.parametrize("name", (None, "test"))
def test_serde_full_masked_structure_and_name(name: str | None):
    sequence = b"X" * 10
    complex = Complex({"A": Protein(sequence)}, name=name)
    assert complex.name == name
    assert complex.get_protein(chain_id="A").sequence == sequence
    new_complex = Complex.from_string(complex.to_string(), format="cif")
    # TODO: when starting with name=None, after deser, name is "model" and not None
    #       this may not be an issue... but it would be ideal for the name to stay None
    if name is not None:
        assert new_complex.name == complex.name
    assert new_complex.get_protein(chain_id="A").sequence == sequence


def test_serde_multiple_fully_masked_structure():
    sequence_a = b"X" * 10
    sequence_c = b"A" * 17
    complex = Complex({"A": Protein(sequence_a), "C": Protein(sequence_c)})
    assert complex.get_protein(chain_id="A").sequence == sequence_a
    assert complex.get_protein(chain_id="C").sequence == sequence_c
    new_complex = Complex.from_string(complex.to_string(), format="cif")
    assert new_complex.get_protein(chain_id="A").sequence == sequence_a
    assert new_complex.get_protein(chain_id="C").sequence == sequence_c


def test_can_serde_hemoglobin_cif():
    # tests 2 protein chains with multiplicity 2, and 1 ligand with multiplicity 4
    complex = Complex.from_filepath("tests/data/1A3N.cif")
    assert list(complex.get_proteins().keys()) == ["A", "B", "C", "D"]
    assert list(complex.get_dnas().keys()) == []
    assert list(complex.get_rnas().keys()) == []
    assert list(complex.get_ligands().keys()) == ["E", "F", "G", "H"]

    block = gemmi.cif.read_string(complex.to_string()).sole_block()
    structure = gemmi.make_structure_from_block(block)
    assert len(structure.entities) == 3

    complex = Complex.from_string(complex.to_string(), format="cif")
    assert list(complex.get_proteins().keys()) == ["A", "B", "C", "D"]
    assert list(complex.get_dnas().keys()) == []
    assert list(complex.get_rnas().keys()) == []
    assert list(complex.get_ligands().keys()) == ["E", "F", "G", "H"]


def test_can_serde_protein_dna_rna_cif():
    complex = Complex.from_filepath("tests/data/1MSW.cif")
    assert list(complex.get_proteins().keys()) == ["D"]
    assert list(complex.get_dnas().keys()) == ["A", "B"]
    assert list(complex.get_rnas().keys()) == ["C"]
    assert list(complex.get_ligands().keys()) == []

    block = gemmi.cif.read_string(complex.to_string()).sole_block()
    structure = gemmi.make_structure_from_block(block)
    assert len(structure.entities) == 4

    complex = Complex.from_string(complex.to_string(), format="cif")
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
    copy = complex.copy()
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
    complex = Complex(Complex.from_filepath("tests/data/1A3N.cif").get_proteins())
    pdb_complex = Complex.from_string(complex.to_string(format="pdb"), format="pdb")
    assert pdb_complex.get_chains().keys() == complex.get_chains().keys()
    assert pdb_complex.get_proteins().keys() == complex.get_proteins().keys()

    pdb_complex = Complex.from_filepath("tests/data/1GCM.pdb")
    assert list(pdb_complex.get_chains().keys()) == ["A", "B", "C"]
    assert list(pdb_complex.get_proteins().keys()) == ["A", "B", "C"]


def test_can_serde_some_but_no_all_chains_having_no_structure():
    # test 1/4 chains with structure all masked
    complex = Complex(Complex.from_filepath("tests/data/1A3N.cif").get_proteins())
    _ = complex.get_protein("A").mask_structure()
    assert complex.get_protein(chain_id="A").get_structure_mask().all()
    assert not complex.get_protein(chain_id="B").get_structure_mask().all()
    assert not complex.get_protein(chain_id="C").get_structure_mask().all()
    assert not complex.get_protein(chain_id="D").get_structure_mask().all()
    new_complex = Complex.from_string(complex.to_string(), format="cif")
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
    complex = Complex(Complex.from_filepath("tests/data/1A3N.cif").get_proteins())
    _ = complex.get_protein("A").mask_structure()
    _ = complex.get_protein("D").mask_structure()
    assert complex.get_protein(chain_id="A").get_structure_mask().all()
    assert not complex.get_protein(chain_id="B").get_structure_mask().all()
    assert not complex.get_protein(chain_id="C").get_structure_mask().all()
    assert complex.get_protein(chain_id="D").get_structure_mask().all()
    new_complex = Complex.from_string(complex.to_string(), format="cif")
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
