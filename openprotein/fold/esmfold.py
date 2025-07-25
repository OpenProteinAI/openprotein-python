"""Community-based ESMFold model."""

from collections.abc import Sequence

from openprotein.base import APISession
from openprotein.common import ModelMetadata

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
        result = FoldResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=sequences,
                num_recycles=num_recycles,
            ),
        )
        assert isinstance(result, FoldResultFuture)
        return result
