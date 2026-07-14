from unittest.mock import MagicMock

from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.molecules import Complex, Ligand, Protein


def test_normalize_inputs_splits_colon_delimited_into_one_complex():
    [complex] = normalize_inputs(["AAAA:BBBB"])
    chains = complex.get_chains()
    assert list(chains.keys()) == ["A", "B"]
    assert [chain.sequence.decode() for chain in chains.values()] == ["AAAA", "BBBB"]


def test_normalize_inputs_colon_delimited_bytes():
    [complex] = normalize_inputs([b"AAAA:BBBB"])
    assert [chain.sequence.decode() for chain in complex.get_chains().values()] == [
        "AAAA",
        "BBBB",
    ]


def test_normalize_inputs_colon_delimited_chains_default_to_null_msa():
    [complex] = normalize_inputs(["AAAA:BBBB"])
    for chain in complex.get_chains().values():
        assert chain.msa is Protein.NullMSA


def test_normalize_inputs_single_sequence_is_single_chain_complex():
    [complex] = normalize_inputs(["AAAA"])
    assert list(complex.get_chains().keys()) == ["A"]


def test_normalize_inputs_preserves_order_and_resets_ids_per_complex():
    complexes = normalize_inputs(["AAAA", "BBBB:CCCC", Protein(b"DDDD")])
    assert [list(c.get_chains().keys()) for c in complexes] == [["A"], ["A", "B"], ["A"]]


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
