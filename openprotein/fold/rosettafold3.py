"""Community-based RosettaFold3 models for complex structure prediction with ligands/dna/rna."""

from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.chains import Ligand
from openprotein.common import ModelMetadata
from openprotein.protein import Protein

from . import api
from .complex import id_generator
from .future import FoldComplexResultFuture
from .models import FoldModel


class RosettaFold3Model(FoldModel):
    """
    Class providing inference endpoints for RosettaFold-3 structure prediction model.
    """

    model_id: str = "rosettafold-3"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session, model_id, metadata)

    def fold(
        self,
        proteins: list[Protein] | MSAFuture | None = None,
        ligands: list[Ligand] | None = None,
        diffusion_samples: int = 1,
        num_recycles: int = 10,
        num_steps: int = 50,
        **kwargs,
    ) -> FoldComplexResultFuture:
        """
        Request structure prediction with RosettaFold-3 model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture | None
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        ligands : List[Ligand] | None
            List of ligands to include in folded output.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use

        Returns
        -------
        FoldComplexResultFuture
            Future for the folding complex result.
        """
        # collate the id's used
        used_ids = []
        if isinstance(proteins, list):
            for protein in proteins:
                if isinstance(protein, Protein) and protein.chain_id is not None:
                    if isinstance(protein.chain_id, str):
                        used_ids.append(protein.chain_id)
                    elif isinstance(protein.chain_id, list):
                        used_ids.extend(protein.chain_id)
        for ligand in ligands or []:
            if isinstance(ligand.chain_id, str):
                used_ids.append(ligand.chain_id)
            elif isinstance(ligand.chain_id, list):
                used_ids.extend(ligand.chain_id)
        id_gen = id_generator(used_ids)
        # build the proteins from msa
        if isinstance(proteins, MSAFuture):
            align_api = getattr(self.session, "align", None)
            assert isinstance(align_api, AlignAPI)
            msa = proteins  # rename
            proteins = []  # convert back to list of proteins
            seed = align_api.get_seed(job_id=msa.job.job_id)
            query_seqs_cardinality: dict[str, int] = dict()
            for seq in seed.split(":"):
                query_seqs_cardinality[seq] = query_seqs_cardinality.get(seq, 0) + 1
            for seq, card in query_seqs_cardinality.items():
                protein = Protein(sequence=seq)
                if card == 1:
                    id = next(id_gen)
                else:
                    id = [next(id_gen) for _ in range(card)]
                protein.chain_id = id
                protein.msa = msa
                proteins.append(protein)

        # build the sequences input
        sequences: list[dict[str, Any]] = []
        for protein in proteins or []:
            # check the msa
            msa = protein.msa
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
            # add the protein in the expected format
            p = {
                "id": protein.chain_id or next(id_gen),
                "msa_id": msa_id,
                "sequence": protein.sequence.decode(),
            }
            if protein.cyclic:
                p["cyclic"] = protein.cyclic
            sequences.append({"protein": p})
        for ligand in ligands or []:
            ligand_: dict = {"id": ligand.chain_id or next(id_gen)}
            if ligand.ccd:
                ligand_["ccd"] = ligand.ccd
            if ligand.smiles:
                ligand_["smiles"] = ligand.smiles
            sequences.append({"ligand": ligand_})

        if len(sequences) == 0:
            raise ValueError("Expected proteins or ligands")

        return FoldComplexResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=sequences,
                diffusion_samples=diffusion_samples,
                num_recycles=num_recycles,
                num_steps=num_steps,
                **kwargs,
            ),
            model_id=self.model_id,
            proteins=proteins,
            ligands=ligands,
        )
