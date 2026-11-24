from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from openprotein.fold import _request as request
from openprotein.fold.common import serialize_input
from openprotein.fold.future import FoldResultFuture
from openprotein.fold.schemas import FoldJob, FoldMetadata
from openprotein.jobs import JobStatus, JobType
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein


def test_every_chain_wrapper_accepts_its_entities():
    msa_id = uuid4()
    chain = request.MsaChain.model_validate(
        {"protein": {"id": "A", "sequence": "ACDE", "msa_id": str(msa_id)}}
    )
    assert chain.protein is not None and chain.protein.msa_id == msa_id
    assert chain.entity is chain.protein
    assert request.NoMsaChain(ligand=request.Ligand(id="L", ccd="SAH")).ligand
    assert request.ProteinChain(
        protein=request.Protein(id="A", sequence="ACDE")
    ).protein
    assert request.MsaProteinChain(
        protein=request.ProteinWithMSA(id="A", sequence="ACDE")
    ).protein
    assert request.MsaProteinLigandChain(
        ligand=request.Ligand(id="L", smiles="CCO")
    ).ligand
    assert request.MsaChain(dna=request.DNA(id="D", sequence="ACGT")).dna
    assert request.MsaChain(rna=request.RNA(id="R", sequence="ACGU")).rna


@pytest.mark.parametrize("chain_id", ["A", "h1", "ABCDE", "12345", ["A", "B"]])
def test_chain_ids_are_one_to_five_letters_or_digits(chain_id):
    assert request.Protein(id=chain_id, sequence="ACDE").id == chain_id


@pytest.mark.parametrize(
    "chain_id", ["", "ABCDEF", "A B", "A-1", "α", [], ["A", ""]], ids=repr
)
def test_other_chain_ids_are_rejected(chain_id):
    with pytest.raises(ValidationError, match="chain id"):
        request.Protein(id=chain_id, sequence="ACDE")


@pytest.mark.parametrize(
    "wrapper, payload",
    [
        (request.MsaChain, {}),
        (
            request.MsaChain,
            {
                "protein": {"id": "A", "sequence": "ACDE"},
                "dna": {"id": "D", "sequence": "ACGT"},
            },
        ),
        (request.ProteinChain, {"ligand": {"id": "L", "ccd": "SAH"}}),
        (
            request.NoMsaChain,
            {"protein": {"id": "A", "sequence": "ACDE", "msa_id": None}},
        ),
    ],
    ids=["no-entity", "two-entities", "entity-outside-scope", "field-outside-scope"],
)
def test_a_chain_names_exactly_one_entity_the_wrapper_accepts(wrapper, payload):
    with pytest.raises(ValidationError):
        wrapper.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "L"},
        {"id": "L", "smiles": "CCO", "ccd": "SAH"},
        {"id": "L", "smiles": ""},
    ],
    ids=["neither", "both", "empty-string"],
)
def test_a_ligand_is_exactly_one_of_smiles_or_ccd(payload):
    with pytest.raises(ValidationError, match="smiles or ccd"):
        request.Ligand.model_validate(payload)


def test_an_empty_ccd_counts_as_absent():
    assert request.Ligand(id="L", smiles="CCO", ccd="").ccd is None


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="cyclic"):
        request.Protein.model_validate({"id": "A", "sequence": "ACDE", "cyclic": True})


def _complex() -> Complex:
    protein = Protein("MVTPEG")
    protein.msa = Protein.NullMSA
    return Complex(
        {
            "A": protein,
            "L": Ligand(smiles="CCO"),
            "D": DNA("ACGT"),
            "R": RNA("ACGU"),
        }
    )


def test_complex_chain_ids_expand_copies_in_request_order():
    chains = [
        request.ProteinChain(protein=request.Protein(id=["A", "B"], sequence="ACDE")),
        request.NoMsaChain(ligand=request.Ligand(id="L", ccd="SAH")),
    ]

    assert request.complex_chain_ids(chains) == ["A", "B", "L"]


def test_complex_chain_ids_refuse_a_repeated_id():
    chains = [
        request.ProteinChain(protein=request.Protein(id="A", sequence="ACDE")),
        request.ProteinChain(protein=request.Protein(id="A", sequence="ACDF")),
    ]

    with pytest.raises(ValueError, match="duplicate chain id 'A'"):
        request.complex_chain_ids(chains)


def test_serialize_input_dumps_validated_chains():
    [entries] = serialize_input(MagicMock(), [_complex()], needs_msa=True)
    assert entries == [
        {"protein": {"id": "A", "sequence": "MVTPEG", "msa_id": None}},
        {"ligand": {"id": "L", "smiles": "CCO"}},
        {"dna": {"id": "D", "sequence": "ACGT"}},
        {"rna": {"id": "R", "sequence": "ACGU"}},
    ]


def test_a_serialized_complex_reads_back_as_the_same_molecules():
    session = MagicMock()
    job = FoldJob(
        job_id="job",
        job_type=JobType.embeddings_fold,
        status=JobStatus.SUCCESS,
        created_date=datetime.now(timezone.utc),
    )
    sequences = serialize_input(session, [_complex()], needs_msa=True)
    future = FoldResultFuture(
        session,
        job=job,
        metadata=FoldMetadata(job_id="job", model_id="m", sequences=sequences),
        sequences=[],
    )
    [complex] = future.complexes
    assert list(complex.get_chains()) == ["A", "L", "D", "R"]
    assert complex.get_protein("A").sequence == b"MVTPEG"
    assert complex.get_ligand("L").smiles == "CCO"
    assert complex.get_dna("D").sequence == "ACGT"
    assert complex.get_rna("R").sequence == "ACGU"
