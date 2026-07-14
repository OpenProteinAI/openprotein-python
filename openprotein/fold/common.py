"""Common utilities for creating fold jobs."""

from typing import Any, Sequence

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.fold.complex import id_generator
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein
from openprotein.molecules.template import Template
from openprotein.prompt import PromptAPI


def normalize_inputs(
    proteins: Sequence[Complex | Protein | str | bytes],
):
    normalized_complexes: list[Complex] = []
    for protein in proteins:
        if isinstance(protein, Complex):
            normalized_complexes.append(protein)
        elif isinstance(protein, Protein):
            # a lone protein becomes a single-chain complex
            id_gen = id_generator()
            normalized_complexes.append(Complex(chains={next(id_gen): protein}))
        else:
            if isinstance(protein, bytes):
                protein = protein.decode()
            # a ':'-delimited string is one complex, one chain per split sequence.
            # chain ids only need to be unique within the complex, so use a fresh
            # generator per complex.
            id_gen = id_generator()
            chains: dict[str, Protein] = {}
            for seq in protein.split(":"):
                chain = Protein(sequence=seq)
                # protein strings default to null msa
                chain.msa = Protein.NullMSA
                chains[next(id_gen)] = chain
            normalized_complexes.append(Complex(chains=chains))

    return normalized_complexes


def msa_future_to_complex(session: APISession, msa: MSAFuture) -> Complex:
    """Convert an ``MSAFuture`` seed sequence set into a single multichain ``Complex``."""
    align_api = getattr(session, "align", None)
    assert isinstance(align_api, AlignAPI)
    seed = align_api.get_seed(job_id=msa.job.job_id)
    proteins: dict[str, Protein] = {}
    for chain_id, seq in zip(id_generator(), seed.split(":")):
        protein = Protein(sequence=seq)
        protein.msa = msa.id
        proteins[chain_id] = protein
    return Complex(chains=proteins)


def normalize_templates(
    session: APISession,
    sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
    templates: Sequence[Protein | Complex | Template] | None = None,
) -> list[Template]:
    """Normalize and validate template inputs for fold models."""
    normalized_templates: list[Template] = []

    # normalize templates attached to sequence/complex inputs
    if not isinstance(sequences, MSAFuture):
        first_chain_id_to_template = {}
        for batch_idx, seq in enumerate(sequences):
            if isinstance(seq, (str, bytes)):
                seq = Protein(seq)
            seq._assert_valid_templates()
            if isinstance(seq, Protein):
                complex = Complex({"A": seq})
            else:
                complex = seq

            for chain_id, protein in complex.get_proteins().items():
                if batch_idx == 0:
                    first_chain_id_to_template[chain_id] = protein.templates
                    for template in protein.templates:
                        normalized_templates.append(
                            Template.from_obj(template, chain_id=chain_id)
                        )
                elif first_chain_id_to_template[chain_id] != protein.templates:
                    raise ValueError(
                        "Expected same chain across batches to have the same templates"
                    )

            if batch_idx == 0:
                first_templates = complex.templates
                for template in complex.templates:
                    normalized_templates.append(Template.from_obj(template))
            elif first_templates != complex.templates:
                raise ValueError(
                    "Expected templates across complexes in batch to be the same"
                )

    # normalize method-level templates
    if templates is not None:
        if isinstance(sequences, MSAFuture):
            validation_sequences: Sequence[Complex | Protein | str | bytes] = [
                msa_future_to_complex(session=session, msa=sequences)
            ]
        else:
            validation_sequences = sequences
        for template in templates:
            template = Template.from_obj(template)
            for seq in validation_sequences:
                if isinstance(seq, (str, bytes)):
                    seq = Protein(seq)
                template.validate_for_target(seq)
            normalized_templates.append(template)

    return normalized_templates


def resolve_templates(session: APISession, templates: Sequence[Template]) -> list[dict]:
    """Resolve normalized ``Template`` objects into backend API payload dictionaries."""
    prompt_api = getattr(session, "prompt", None)
    assert isinstance(prompt_api, PromptAPI)

    template_dicts: list[dict] = []
    struct_id_to_query_id: dict[int, str] = {}

    for template in templates:
        struct_id = id(template.template)
        if struct_id not in struct_id_to_query_id:
            struct_id_to_query_id[struct_id] = prompt_api._resolve_query(
                query=template.template
            )

        template_dict: dict = {"query_id": struct_id_to_query_id[struct_id]}
        if template.mapping is not None:
            if isinstance(template.mapping, str):
                template_dict["chain_id"] = template.mapping
            else:
                template_dict["chain_id"] = list(template.mapping.values())
                template_dict["template_id"] = list(template.mapping.keys())
        if template.index is not None:
            template_dict["index"] = list(template.index)
        if template.index_intervals is not None:
            template_dict["index_intervals"] = [
                [int(start), int(end)] for start, end in template.index_intervals
            ]
        template_dicts.append(template_dict)

    return template_dicts


def serialize_input(session: APISession, complexes: list[Complex], needs_msa: bool):
    # build the serialized input
    _models: list[list[dict[str, Any]]] = []
    msa_to_seed: dict[str, set[str]] = dict()
    for complex in complexes:
        _complex: list[dict[str, Any]] = []
        for chain_id, chain in complex.get_chains().items():
            if isinstance(chain, Protein):
                # add the protein in the unified format
                p: dict = {
                    "id": chain_id,
                    "sequence": chain.sequence.decode(),
                }
                if needs_msa:
                    # check the msa
                    msa = chain.msa
                    if msa is None:
                        raise ValueError(
                            "Expected all protein sequences to have `.msa` set with an `MSAFuture` or `Protein.single_sequence_mode` for single sequence mode."
                        )
                    # convert to msa id or null for single sequence mode
                    msa_id = (
                        msa
                        if isinstance(msa, str)
                        else msa.id
                        if isinstance(msa, MSAFuture)
                        else None
                    )
                    # handle msa
                    this_msas: set[str] = set()
                    if msa_id is not None:
                        this_msas.add(msa_id)
                        if msa_id in msa_to_seed:
                            seeds = msa_to_seed[msa_id]
                        else:
                            align_api = getattr(session, "align", None)
                            assert isinstance(align_api, AlignAPI)
                            seed = align_api.get_seed(job_id=msa_id)
                            # need a counter so we can make sure later that the proteins make up the msa completely
                            seeds = set(seed.split(":"))
                            msa_to_seed[msa_id] = seeds
                    # make sure we only have one msa
                    # TODO: by my reading, this could never have size > 1 atm...
                    if len(this_msas) > 1:
                        raise ValueError("Expected only 1 unique msa")
                    p["msa_id"] = msa_id
                _complex.append({"protein": p})
            elif isinstance(chain, Ligand):
                ligand_payload: dict[str, Any] = {"id": chain_id}
                if chain.smiles is not None:
                    ligand_payload["smiles"] = chain.smiles
                if chain.ccd is not None:
                    ligand_payload["ccd"] = chain.ccd
                _complex.append({"ligand": ligand_payload})
            elif isinstance(chain, DNA):
                d: dict[str, Any] = {"id": chain_id, "sequence": chain.sequence}
                _complex.append({"dna": d})
            elif isinstance(chain, RNA):
                r: dict[str, Any] = {"id": chain_id, "sequence": chain.sequence}
                _complex.append({"rna": r})
            else:
                raise ValueError(f"Unexpected chain type: {chain}")
        _models.append(_complex)

    return _models
