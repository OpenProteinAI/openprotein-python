"""Community-based ESMFold model."""

import warnings
from collections.abc import Sequence
from typing import Sequence

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.molecules import DNA, RNA, Ligand, Protein, Complex

from . import api
from .future import FoldResultFuture
from .models import FoldModel


class ESMFoldModel(FoldModel):
    """
    Class providing inference endpoints for Facebook's ESMFold structure prediction models.
    """

    model_id: str = "esmfold"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes],
        num_recycles: int | None = None,
    ) -> FoldResultFuture:
        """
        Fold sequences using this model.

        Parameters
        ----------
        sequences : Sequence[bytes | str]
            sequences to fold
        num_recycles : int | None
            number of times to recycle models
        Returns
        -------
            FoldResultFuture
        """
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
                            "ESMFold does not support ligand/DNA/RNA input. These extra chains will be ignored in the output."
                        )
                    del complex._chains[id]

        _complexes = serialize_input(
            self.session, normalized_complexes, needs_msa=False
        )
        result = FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                num_recycles=num_recycles,
            ),
            complexes=normalized_complexes,
        )
        return result
