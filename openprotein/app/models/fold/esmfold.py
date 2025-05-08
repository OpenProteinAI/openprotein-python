from collections.abc import Sequence

from openprotein.api import fold

from .base import FoldModel
from .future import FoldResultFuture


class ESMFoldModel(FoldModel):

    model_id = "esmfold"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

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
        return FoldResultFuture.create(
            session=self.session,
            job=fold.fold_models_esmfold_post(
                self.session, sequences, num_recycles=num_recycles
            ),
        )
