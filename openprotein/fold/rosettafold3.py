"""Community-based RosettaFold3 models for complex structure prediction with ligands/dna/rna."""

import warnings
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.fold.future import FoldResultFuture
from openprotein.molecules import Protein, DNA, RNA, Ligand, Complex

from . import api
from .complex import id_generator
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
        sequences: list[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 10,
        num_steps: int = 50,
        **kwargs,
    ) -> FoldResultFuture:
        """
        Request structure prediction with RosettaFold-3 model.

        Parameters
        ----------
        sequences: list[Complex | Protein | str | bytes] | MSAFuture,
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use

        Returns
        -------
        FoldResultFuture
            Future for the folding results.
        """

        # build the normalized_models from msa
        if isinstance(sequences, MSAFuture):
            id_gen = id_generator()
            align_api = getattr(self.session, "align", None)
            assert isinstance(align_api, AlignAPI)
            msa = sequences  # rename
            seed = align_api.get_seed(job_id=msa.job.job_id)
            _proteins: dict[str, Protein] = {}
            for seq in seed.split(":"):
                protein = Protein(sequence=seq)
                id = next(id_gen)
                protein.msa = msa.id
                _proteins[id] = protein
            normalized_complexes = [Complex(chains=_proteins)]

        else:
            normalized_complexes = normalize_inputs(sequences)

        for complex in normalized_complexes:
            for id, chain in complex.get_chains().items():
                if isinstance(chain, DNA) or isinstance(chain, RNA):
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")  # Force warning to always show
                        warnings.warn(
                            "RosettaFold-3 does not support DNA/RNA input. These extra chains will be ignored in the output."
                        )
                    del complex._chains[id]

        _complexes = serialize_input(self.session, normalized_complexes, needs_msa=True)

        if len(_complexes) == 0:
            raise ValueError("Expected proteins or ligands")

        return FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                diffusion_samples=diffusion_samples,
                num_recycles=num_recycles,
                num_steps=num_steps,
                **kwargs,
            ),
            complexes=normalized_complexes,
        )
