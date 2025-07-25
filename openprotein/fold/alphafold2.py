"""Community-based AlphaFold 2 model running using ColabFold."""

import warnings
from collections import Counter

from openprotein.align import MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.protein import Protein

from . import api
from .future import FoldComplexResultFuture
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
        proteins: list[Protein] | MSAFuture | None = None,
        num_recycles: int | None = None,
        num_models: int = 1,
        num_relax: int = 0,
        **kwargs,
    ) -> FoldComplexResultFuture:
        """
        Post sequences to alphafold model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture
            List of protein sequences to fold. `Protein` objects must be tagged with an `msa`. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        num_recycles : int
            number of times to recycle models
        num_models : int
            number of models to train - best model will be used
        max_msa : Union[str, int]
            maximum number of sequences in the msa to use.
        relax_max_iterations : int
            maximum number of iterations

        Returns
        -------
        job : Job
        """
        if "msa" in kwargs:
            warnings.warn(
                "Inputs to AlphaFold 2 have been updated. 'msa' should be supplied as 'proteins' argument. Support will be dropped in the future."
            )
            proteins = kwargs["msa"]
        if "ligands" in kwargs or "dnas" in kwargs or "rnas" in kwargs:
            with warnings.catch_warnings():
                warnings.simplefilter("always")  # Force warning to always show
                warnings.warn(
                    "Alphafold 2 only supports proteins. All other chains will be ignored"
                )
        if proteins is None:
            raise TypeError("Expected 'proteins' argument")
        if isinstance(proteins, list):
            msa_to_seed: dict[str, Counter] = dict()
            for protein in proteins:
                if (msa := protein.msa) is not None:
                    msa_id = msa.id if isinstance(msa, MSAFuture) else msa
                    if msa_id in msa_to_seed:
                        seeds = msa_to_seed[msa_id]
                    else:
                        from openprotein.align import AlignAPI

                        align_api = getattr(self.session, "align", None)
                        assert isinstance(align_api, AlignAPI)
                        seed = align_api.get_seed(job_id=msa_id)
                        # need a counter so we can make sure later that the proteins make up the msa completely
                        seeds = Counter(seed.split(":"))
                        msa_to_seed[msa_id] = seeds
                    # check that this protein is in the seed
                    if protein.sequence.decode() not in seeds:
                        raise ValueError(
                            f"Expected specified msa_id {msa_id} for protein {protein.sequence} to contain the sequence as part of its seed/query"
                        )
                else:
                    raise ValueError("Expected msa for protein when using AlphaFold 2")
            # now make sure we only have one msa
            if len(msa_to_seed) > 1:
                raise ValueError("Expected only 1 unique msa when using AlphaFold 2")
            # now check that the list of proteins completely make up the msa
            seeds = list(msa_to_seed.values())[0]  # should have just 1
            for protein in proteins:
                # make sure to account for multimers
                seeds[protein.sequence.decode()] -= (
                    len(protein.chain_id) if isinstance(protein.chain_id, list) else 1
                )
                # handle when too many of a sequence in the list of proteins
                if seeds[protein.sequence.decode()] < 0:
                    raise ValueError(
                        "List of proteins does not completely make up the MSA seed"
                    )
            if seeds.total() != 0:
                # handle when overall mismatch - 1 and -1 case is handled above
                raise ValueError(
                    "List of proteins does not completely make up the MSA seed"
                )
            msa_id = list(msa_to_seed.keys())[0]
        elif isinstance(proteins, MSAFuture):
            msa_id = proteins.id
        else:
            raise TypeError("Expected either list of Proteins or MSAFuture")

        return FoldComplexResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                self.session,
                model_id=self.model_id,
                msa_id=msa_id,
                num_recycles=num_recycles,
                num_models=num_models,
                num_relax=num_relax,
            ),
        )
