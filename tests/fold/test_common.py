import inspect
from unittest.mock import MagicMock

import numpy as np
import pytest

import openprotein.common.residue_constants as RC
from openprotein.align import AlignAPI
from openprotein.base import APISession
from openprotein.codecs import MSA as DecodedMSA
from openprotein.embeddings import api as embeddings_api
from openprotein.fold.common import (
    _msa_id_of,
    normalize_inputs,
    normalize_templates,
    serialize_input,
)
from openprotein.molecules import Complex, Ligand, Protein
from openprotein.molecules.template import Template


def test_an_unknown_msa_type_names_the_accepted_and_the_given_types():
    with pytest.raises(
        TypeError,
        match=r"openprotein\.data\.MSA.*not openprotein\.codecs\.msa\.MSA",
    ):
        _msa_id_of(object.__new__(DecodedMSA))


def test_serializing_an_msa_id_reads_no_seed():
    session = MagicMock(spec=APISession)
    session.align = MagicMock(spec=AlignAPI)
    protein = Protein(sequence="MKT")
    protein.msa = "e0b1c3d4-0000-4000-8000-000000000001"

    serialize_input(session, [Complex({"A": protein})], needs_msa=True)

    session.align.get_seed.assert_not_called()


def test_the_embeddings_stream_documents_the_sdk_error():
    stream = embeddings_api.request_get_embeddings_stream
    assert "openprotein.errors.HTTPError" in (stream.__doc__ or "")
    assert "raise_for_status" not in inspect.getsource(stream)


def _protein_with_structure() -> Protein:
    """Build a Protein carrying coordinates so Template's validation accepts it."""
    amino_acids = list(RC.restype_1to3.keys()) + ["X"]
    p = Protein(sequence="".join(amino_acids[i] for i in range(6)))
    return p._set_coordinates(np.random.random(p.coordinates.shape).astype(np.float32))


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
    assert [list(c.get_chains().keys()) for c in complexes] == [
        ["A"],
        ["A", "B"],
        ["A"],
    ]


def test_normalize_inputs_lone_string_is_one_complex():
    [complex] = normalize_inputs("AAAA:BBBB")
    assert [chain.sequence.decode() for chain in complex.get_chains().values()] == [
        "AAAA",
        "BBBB",
    ]


def test_normalize_inputs_lone_bytes_is_one_complex():
    [complex] = normalize_inputs(b"AAAA")
    assert [chain.sequence.decode() for chain in complex.get_chains().values()] == [
        "AAAA"
    ]


def test_normalize_inputs_lone_protein_is_single_chain_complex():
    [complex] = normalize_inputs(Protein(b"DDDD"))
    assert list(complex.get_chains().keys()) == ["A"]


def test_normalize_inputs_lone_complex_passes_through():
    complex = Complex({"A": Protein(b"AAAA"), "B": Protein(b"BBBB")})
    assert normalize_inputs(complex) == [complex]


@pytest.mark.parametrize("chain_id", ["A", "B"])
def test_a_chain_with_different_templates_across_a_batch_is_refused(chain_id):
    first = Complex({"A": Protein("MKT")})
    later_chains = {"A": Protein("MKT")}
    later_chains[chain_id] = Protein("MKT").set_templates([_protein_with_structure()])
    later = Complex(later_chains)

    with pytest.raises(ValueError, match="same chain.*same templates"):
        normalize_templates(MagicMock(), [first, later])


@pytest.mark.parametrize(
    ("first_ids", "later_ids", "templated_id"),
    [("A", "B", "A"), ("AB", "A", "B")],
)
def test_a_shared_template_requires_its_chain_in_every_complex(
    first_ids, later_ids, templated_id
):
    first = Complex({chain: Protein("MKT") for chain in first_ids})
    first.get_protein(templated_id).templates = [_protein_with_structure()]
    later = Complex({chain: Protein("MKT") for chain in later_ids})

    with pytest.raises(ValueError, match="Template maps.*does not exist"):
        normalize_templates(MagicMock(), [first, later])


def test_shared_templates_allow_additional_untemplated_chains():
    template = _protein_with_structure()
    first = Complex({"A": Protein("MKT").set_templates([template])})
    later = Complex({"A": Protein("MKT").set_templates([template]), "B": Protein("AC")})

    [normalized] = normalize_templates(MagicMock(), [first, later])

    assert normalized.template is template
    assert normalized.mapping == "A"


def test_automatic_template_mapping_allows_different_chain_ids():
    template = Template(template=_protein_with_structure())
    first = Complex({"A": Protein("MKT").set_templates([template])})
    later = Complex({"B": Protein("AC")})

    assert normalize_templates(MagicMock(), [first, later]) == [template]


def test_normalize_templates_validates_lone_complex_against_its_own_chains():
    """A lone Complex is one target, so a chain mapping resolves against its chains."""
    session = MagicMock()
    target = Complex({"A": Protein(b"AAAA"), "B": Protein(b"BBBB")})
    template = Template(template=_protein_with_structure(), mapping="B")
    assert normalize_templates(session, target, [template]) == [template]


def test_normalize_templates_maps_a_template_to_chain_a_of_a_lone_protein():
    """A lone protein is chain A, so a template mapped to ``A`` is valid."""
    session = MagicMock()
    template = Template(template=_protein_with_structure(), mapping="A")
    assert normalize_templates(session, Protein(b"AAAA"), [template]) == [template]


def test_normalize_templates_refuses_another_chain_of_a_lone_protein():
    """A lone protein has no chain but ``A``."""
    session = MagicMock()
    template = Template(template=_protein_with_structure(), mapping="B")
    with pytest.raises(ValueError, match="does not exist in the target"):
        normalize_templates(session, Protein(b"AAAA"), [template])


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
