"""Common utilities for creating fold jobs."""

from typing import Any, Sequence

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.fold.complex import id_generator
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein


def normalize_inputs(
    proteins: Sequence[Complex | Protein | str | bytes],
):
    # collate the id's used
    used_ids = []
    normalized_complexes: list[Complex] = []
    remaining_proteins: list[Protein] = []
    remaining_protein_strings: list[str] = []
    if isinstance(proteins, list):
        for protein in proteins:
            if isinstance(protein, Protein):
                # collate these to init with id_gen
                remaining_proteins.append(protein)
            elif isinstance(protein, Complex):
                used_ids.extend(list(protein.get_chains().keys()))
                normalized_complexes.append(protein)
            else:
                if isinstance(protein, bytes):
                    protein = protein.decode()
                # handle ':'-delimited
                for seq in protein.split(":"):
                    # collate these to init with id_gen
                    remaining_protein_strings.append(seq)

    # auto generate the chain ids
    id_gen = id_generator(used_ids)

    # add the remaining proteins with id gen
    for protein in remaining_proteins:
        id = next(id_gen)
        normalized_complexes.append(Complex(chains={id: protein}))

    for protein_str in remaining_protein_strings:
        id = next(id_gen)
        protein = Protein(sequence=protein_str)
        # protein strings default to null msa
        protein.msa = Protein.NullMSA
        normalized_complexes.append(Complex(chains={id: protein}))

    return normalized_complexes


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
                        else msa.id if isinstance(msa, MSAFuture) else None
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
                l = {
                    "id": chain_id,
                }
                if chain.smiles is not None:
                    l["smiles"] = chain.smiles
                if chain.ccd is not None:
                    l["ccd"] = chain.ccd
                _complex.append({"ligand": l})
            elif isinstance(chain, DNA):
                d = {"id": chain_id, "sequence": chain.sequence}
                _complex.append({"dna": d})
            elif isinstance(chain, RNA):
                r = {"id": chain_id, "sequence": chain.sequence}
                _complex.append({"rna": r})
            else:
                raise ValueError(f"Unexpected chain type: {chain}")
        _models.append(_complex)

    return _models
