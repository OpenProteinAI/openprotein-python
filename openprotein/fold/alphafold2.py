"""Community-based AlphaFold 2 model running using ColabFold."""

import io
import warnings
from typing import Any, Sequence

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.fold.complex import id_generator
from openprotein.molecules import Protein, DNA, RNA, Ligand, Complex

from . import api
from .future import FoldResultFuture
from .models import FoldModel


class AlphaFold2Model(FoldModel):
    """
    Class providing inference endpoints for AlphaFold2 structure prediction models, based on the implementation by ColabFold.
    """

    model_id: str = "alphafold2"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str] | MSAFuture | None = None,
        num_recycles: int | None = None,
        num_models: int = 1,
        num_relax: int = 0,
        **kwargs,
    ) -> FoldResultFuture:
        """
        Post sequences to alphafold model.

        Parameters
        ----------
        sequences : List[Complex | Protein | str] | MSAFuture
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        num_recycles : int
            number of times to recycle models
        num_models : int
            number of models to train - best model will be used
        num_relax : int
            maximum number of iterations for relax

        Returns
        -------
        job : Job
        """
        from openprotein.align import AlignAPI

        if "msa" in kwargs:
            warnings.warn(
                "Inputs to AlphaFold 2 have been updated. 'msa' should be supplied as 'proteins' argument. Support will be dropped in the future."
            )
            sequences = kwargs["msa"]
            assert isinstance(sequences, MSAFuture), "Expected msa to be an MSAFuture"

        if sequences is None:
            raise TypeError("Expected 'proteins' argument")

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
                if (
                    isinstance(chain, DNA)
                    or isinstance(chain, RNA)
                    or isinstance(chain, Ligand)
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("always")  # Force warning to always show
                        warnings.warn(
                            "AlphaFold-2 does not support ligand/DNA/RNA input. These extra chains will be ignored in the output."
                        )
                    del complex._chains[id]

        _complexes = serialize_input(self.session, normalized_complexes, needs_msa=True)

        if len(_complexes) == 0:
            raise TypeError(
                "Expected either non-empty list of proteins/models/sequences or MSAFuture"
            )

        result = FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                num_recycles=num_recycles,
                num_models=num_models,
                num_relax=num_relax,
            ),
            complexes=normalized_complexes,
        )
        return result
