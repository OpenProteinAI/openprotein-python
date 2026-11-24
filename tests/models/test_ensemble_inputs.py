import re
from pathlib import Path

import gemmi
import numpy as np
import pytest

from openprotein.models.ensemble import _inputs as _ensemble_inputs
from openprotein.molecules import Complex, Protein, Structure

DATA = Path(__file__).parents[1] / "data"
FIELD = "inputs[0].reference_structure"


def _residues(pdb: str) -> int:
    return len(gemmi.read_pdb_string(pdb)[0][0])


def _flow(value, sequence) -> str:
    if isinstance(sequence, Protein):
        sequence = sequence.sequence.decode()
    return _ensemble_inputs.flow_reference(value, sequence, FIELD)


@pytest.mark.parametrize("representation", ["protein", "complex", "structure"])
def test_flow_molecules_become_matching_protein_pdb(ensemble_reference, representation):
    complex = Complex({"A": ensemble_reference})
    value = {
        "protein": ensemble_reference,
        "complex": complex,
        "structure": Structure([complex, complex.copy()]),
    }[representation]
    parsed = gemmi.read_pdb_string(_flow(value, ensemble_reference))
    assert len(parsed) == 1
    assert len(parsed[0]) == 1
    assert len(parsed[0][0]) == len(ensemble_reference)
    sequence = "".join(
        gemmi.find_tabulated_residue(residue.name).one_letter_code
        for residue in parsed[0][0]
    )
    assert sequence == ensemble_reference.sequence.decode()
    for residue in parsed[0][0]:
        atom_name = "CA" if residue.name == "GLY" else "CB"
        assert residue.find_atom(atom_name, "*") is not None


def test_flow_raw_mmcif_requires_explicit_molecule_parsing(ensemble_reference):
    with pytest.raises(ValueError, match="PDB"):
        _flow(ensemble_reference.to_string(format="cif"), ensemble_reference)


def test_flow_pdb_waters_are_not_submitted_as_residues(ensemble_reference):
    lines = ensemble_reference.to_string(format="pdb").splitlines()
    end = next(
        (
            index
            for index, line in enumerate(lines)
            if line.startswith(("ENDMDL", "END"))
        ),
        len(lines),
    )
    lines.insert(
        end,
        "HETATM99998  O   HOH A 901      10.000  11.000  12.000"
        "  1.00 20.00           O  ",
    )
    sent = _flow("\n".join(lines) + "\n", ensemble_reference)
    assert "HOH" not in sent
    assert len(gemmi.read_pdb_string(sent)[0][0]) == len(ensemble_reference)


def test_flow_rejects_declared_unmodelled_residues():
    path = (
        Path(__file__).parents[1]
        / "data"
        / "boltzgen_entity_one_letter_missing_residues.cif"
    )
    protein = Complex.from_filepath(path, verbose=False).get_proteins()["B"]
    assert protein.get_structure_mask().any()
    with pytest.raises(ValueError, match="unmodelled residues"):
        _flow(protein, protein)


def test_flow_accepts_a_reference_trimmed_to_its_modelled_residues():
    path = Path(__file__).parents[1] / "data" / "1GCM.pdb"
    chain = Structure.from_filepath(path, verbose=False)[0].get_proteins()["A"]
    with pytest.raises(ValueError, match="trim"):
        _flow(chain, chain)
    modelled = chain[~chain.get_structure_mask()]
    assert _residues(_flow(modelled, modelled)) == len(modelled)


def test_flow_rejects_multiple_protein_chains(ensemble_reference):
    value = Complex({"A": ensemble_reference, "B": ensemble_reference})
    with pytest.raises(ValueError, match="one protein chain"):
        _flow(value, ensemble_reference)


def test_reference_requires_coordinates():
    with pytest.raises(ValueError, match="coordinates"):
        _flow(Protein(sequence="MKT"), "MKT")


@pytest.mark.parametrize("filename", ["1A3N.cif", "1MSW.cif"])
def test_flow_rejects_present_nonprotein_chains(filename):
    reference = Structure.from_filepath(DATA / filename, verbose=False)
    with pytest.raises(ValueError, match="non-protein"):
        _flow(reference, "MKT")


def test_flow_without_references_sends_none():
    assert _ensemble_inputs.flow_references([None, None], ["MKT", "MKT"]) is None


def test_flow_reads_pdb_bytes(ensemble_reference):
    pdb = ensemble_reference.to_string(format="pdb").encode()
    assert _residues(_flow(pdb, ensemble_reference)) == len(ensemble_reference)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (Structure([]), "contains no structure"),
        ("not a structure\n", "contains no protein chain"),
        ("  \n", "is empty"),
        (b"", "is empty"),
        (42, "must be structure text"),
    ],
)
def test_flow_rejects_unusable_structures(value, message):
    with pytest.raises((TypeError, ValueError), match=rf"{re.escape(FIELD)} {message}"):
        _flow(value, "MKT")


def test_flow_pdb_ligand_error_names_the_field(ensemble_reference):
    lines = ensemble_reference.to_string(format="pdb").splitlines()
    ligand = (
        "HETATM99990 MG    MG B 501      10.000  11.000  12.000"
        "  1.00 20.00          MG  "
    )
    with pytest.raises(ValueError, match=r"inputs\[0\]\.reference_structure .*ligand"):
        _flow("\n".join([*lines, ligand]) + "\n", ensemble_reference)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda s: s[:-1], "residues and the sequence has"),
        (lambda s: ("A" if s[0] != "A" else "G") + s[1:], "at position 1"),
    ],
)
def test_flow_reference_must_match_the_sequence(ensemble_reference, edit, message):
    sequence = edit(ensemble_reference.sequence.decode())
    with pytest.raises(ValueError, match=message):
        _flow(ensemble_reference, sequence)


def test_flow_reference_matches_an_unknown_residue(ensemble_reference):
    sequence = "X" + ensemble_reference.sequence.decode()[1:]
    assert _residues(_flow(ensemble_reference, sequence)) == len(ensemble_reference)


def test_a_protein_with_only_plddt_has_no_coordinates():
    protein = Protein(sequence="MKT")
    protein._plddt[:] = 90.0
    assert protein.has_structure
    with pytest.raises(ValueError, match="coordinates"):
        _flow(protein, "MKT")


def test_references_use_the_first_complex_of_a_structure(ensemble_reference):
    shorter = ensemble_reference[:50]
    structure = Structure([Complex({"A": ensemble_reference}), Complex({"A": shorter})])
    assert _residues(_flow(structure, ensemble_reference)) == len(ensemble_reference)


def test_references_leave_caller_molecules_unchanged(ensemble_reference):
    pdb = ensemble_reference.to_string(format="pdb")
    coordinates = ensemble_reference.coordinates.copy()

    _flow(ensemble_reference, ensemble_reference)

    assert ensemble_reference.to_string(format="pdb") == pdb
    np.testing.assert_array_equal(ensemble_reference.coordinates, coordinates)
