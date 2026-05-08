from unittest.mock import MagicMock

from openprotein.fold.common import serialize_input
from openprotein.molecules import Complex, Ligand, Protein


def test_serialize_input_emits_scalar_id_per_chain():
    session = MagicMock()
    seq = b"MVTPEG"
    complex = Complex({"A": Protein(seq), "B": Protein(b"GHIK")})
    [entries] = serialize_input(session, [complex], needs_msa=False)
    assert entries[0] == {"protein": {"id": "A", "sequence": "MVTPEG"}}
    assert entries[1] == {"protein": {"id": "B", "sequence": "GHIK"}}


def test_serialize_input_protein_and_ligand():
    session = MagicMock()
    seq = b"MVTPEG"
    complex = Complex({"A": Protein(seq), "C": Ligand(ccd="SAH")})
    [entries] = serialize_input(session, [complex], needs_msa=False)
    assert entries[0] == {"protein": {"id": "A", "sequence": "MVTPEG"}}
    assert entries[1] == {"ligand": {"id": "C", "ccd": "SAH"}}


def test_serialize_input_ligand_with_smiles():
    session = MagicMock()
    complex = Complex({"E": Ligand(smiles="CCO")})
    [entries] = serialize_input(session, [complex], needs_msa=False)
    assert entries[0] == {"ligand": {"id": "E", "smiles": "CCO"}}
