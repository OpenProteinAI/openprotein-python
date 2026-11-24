"""Community-based RosettaFold3 models for complex structure prediction with ligands/dna/rna."""

import warnings

from openprotein.align import MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import (
    FoldInput,
    msa_future_to_complex,
    normalize_inputs,
    serialize_input,
)
from openprotein.fold.future import FoldResultFuture
from openprotein.molecules import DNA, RNA

from . import api
from .models import FoldModel

__all__ = ["RosettaFold3Model"]


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
        sequences: FoldInput | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 10,
        num_steps: int = 50,
        seed: int | None = None,
        **kwargs,
    ) -> FoldResultFuture:
        """
        Request structure prediction with RosettaFold-3 model.

        Parameters
        ----------
        sequences: FoldInput | MSAFuture,
            Complexes to fold, or a lone complex/protein/sequence. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use
        seed : int | None
            Seed for the prediction, from 1 to 2**32 - 1. None lets the model
            pick its own, so every call is a fresh prediction. Folding the
            same input with the same seed returns the same structures.

        Returns
        -------
        FoldResultFuture
            Future for the folding results.
        """

        if isinstance(sequences, MSAFuture):
            normalized_complexes = [msa_future_to_complex(self.session, sequences)]

        else:
            normalized_complexes = normalize_inputs(sequences)

        for complex in normalized_complexes:
            for id, chain in complex.get_chains().items():
                if isinstance(chain, DNA) or isinstance(chain, RNA):
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")
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
                seed=seed,
                **kwargs,
            ),
            complexes=normalized_complexes,
        )
