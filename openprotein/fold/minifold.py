import warnings
from collections.abc import Sequence

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.molecules import DNA, RNA, Ligand

from . import api
from .future import FoldResultFuture
from .models import FoldModel


class MiniFoldModel(FoldModel):
    """
    Class providing inference endpoints for MiniFold.
    """

    model_id: str = "minifold"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def fold(
        self, sequences: Sequence[bytes | str], num_recycles: int | None = None
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
            if len(complex.get_proteins()) > 1:
                raise ValueError("MiniFold only supports monomers")
            if len(complex.get_chains()) != len(complex.get_proteins()):
                raise ValueError("MiniFold only supports proteins")

        _models = serialize_input(self.session, normalized_complexes, needs_msa=False)
        result = FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_models,
                num_recycles=num_recycles,
            ),
            complexes=normalized_complexes,
        )
        return result
