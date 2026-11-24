"""Common utilities for creating fold jobs."""

from collections.abc import Sequence
from typing import Any

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.data.msa import MSA
from openprotein.fold import _request as request
from openprotein.fold.complex import id_generator
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein
from openprotein.molecules.template import Template
from openprotein.prompt import PromptAPI

__all__ = [
    "FoldEntity",
    "FoldInput",
    "msa_future_to_complex",
    "normalize_inputs",
    "normalize_templates",
    "resolve_templates",
    "serialize_input",
]


FoldEntity = Complex | Protein | str | bytes

FoldInput = Sequence[FoldEntity] | FoldEntity
"""A batch of foldable entities, or a lone entity folded on its own."""


def _as_entities(entities: FoldInput) -> Sequence[FoldEntity]:
    """Wrap a lone entity into a one-element batch."""
    if isinstance(entities, (Complex, Protein, str, bytes)):
        return [entities]
    return entities


def normalize_inputs(proteins: FoldInput):
    entities = _as_entities(proteins)
    normalized_complexes: list[Complex] = []
    for entity in entities:
        if isinstance(entity, Complex):
            normalized_complexes.append(entity)
        elif isinstance(entity, Protein):
            # a lone protein becomes a single-chain complex
            id_gen = id_generator()
            normalized_complexes.append(Complex(chains={next(id_gen): entity}))
        else:
            if isinstance(entity, bytes):
                entity = entity.decode()
            # a ':'-delimited string is one complex, one chain per split sequence.
            # chain ids only need to be unique within the complex, so use a fresh
            # generator per complex.
            id_gen = id_generator()
            chains: dict[str, Protein] = {}
            for seq in entity.split(":"):
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


def _msa_id_of(msa: Any) -> str | None:
    """The MSA id a protein's ``.msa`` sends, or None in single sequence mode."""
    if msa is Protein.single_sequence_mode:
        return None
    if isinstance(msa, str):
        return msa
    if isinstance(msa, (MSAFuture, MSA)):
        return msa.id
    raise TypeError(
        "`.msa` must be an MSA id, an MSAFuture, an `openprotein.data.MSA` or "
        "`Protein.single_sequence_mode`, not "
        f"{type(msa).__module__}.{type(msa).__qualname__}"
    )


def normalize_templates(
    session: APISession,
    sequences: FoldInput | MSAFuture,
    templates: Sequence[Complex | Protein | Template] | None = None,
) -> list[Template]:
    """Normalize and validate template inputs for fold models."""
    normalized_templates: list[Template] = []

    # normalize templates attached to sequence/complex inputs
    if not isinstance(sequences, MSAFuture):
        entities = _as_entities(sequences)
        first_chain_id_to_template = {}
        for batch_idx, entity in enumerate(entities):
            if isinstance(entity, (str, bytes)):
                entity = normalize_inputs(entity)[0]
            entity._assert_valid_templates()
            if isinstance(entity, Protein):
                complex = Complex({"A": entity})
            else:
                complex = entity

            for chain_id, protein in complex.get_proteins().items():
                if batch_idx == 0:
                    first_chain_id_to_template[chain_id] = protein.templates
                    for template in protein.templates:
                        normalized_templates.append(
                            Template.from_obj(template, chain_id=chain_id)
                        )
                elif list(first_chain_id_to_template.get(chain_id, ())) != list(
                    protein.templates
                ):
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

            for template in normalized_templates:
                if template.index is None and template.index_intervals is None:
                    template.validate_for_target(complex)

    # normalize method-level templates
    if templates is not None:
        if isinstance(sequences, MSAFuture):
            validation_sequences: Sequence[FoldEntity] = [
                msa_future_to_complex(session=session, msa=sequences)
            ]
        else:
            validation_sequences = _as_entities(sequences)
        for template in templates:
            template = Template.from_obj(template)
            for entity in validation_sequences:
                if isinstance(entity, (str, bytes)):
                    entity = normalize_inputs(entity)[0]
                if isinstance(entity, Protein):
                    entity = Complex({"A": entity})
                template.validate_for_target(entity)
            normalized_templates.append(template)

    return normalized_templates


def resolve_templates(session: APISession, templates: Sequence[Template]) -> list[dict]:
    """Resolve normalized ``Template`` objects into API payload dictionaries."""
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
    """Return the ``complexes`` of a fold request, one validated ``MsaChain`` dump per chain.

    ``msa_id`` is sent for every protein when ``needs_msa``, null in single
    sequence mode, and left out otherwise.
    """
    _models: list[list[dict[str, Any]]] = []
    for complex in complexes:
        _complex: list[dict[str, Any]] = []
        for chain_id, chain in complex.get_chains().items():
            if isinstance(chain, Protein):
                p: dict[str, Any] = {
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
                    msa_id = _msa_id_of(msa)
                    p["msa_id"] = msa_id
                entry = request.MsaChain(protein=request.ProteinWithMSA(**p))
            elif isinstance(chain, Ligand):
                ligand = {"smiles": chain.smiles, "ccd": chain.ccd}
                entry = request.MsaChain(
                    ligand=request.Ligand(
                        id=chain_id,
                        **{k: v for k, v in ligand.items() if v is not None},
                    )
                )
            elif isinstance(chain, DNA):
                entry = request.MsaChain(
                    dna=request.DNA(id=chain_id, sequence=chain.sequence)
                )
            elif isinstance(chain, RNA):
                entry = request.MsaChain(
                    rna=request.RNA(id=chain_id, sequence=chain.sequence)
                )
            else:
                raise ValueError(f"Unexpected chain type: {chain}")
            _complex.append(entry.model_dump(mode="json", exclude_unset=True))
        _models.append(_complex)

    return _models
