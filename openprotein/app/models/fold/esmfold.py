from openprotein.api import fold

from .base import FoldModel
from .future import FoldResultFuture


class ESMFoldModel(FoldModel):

    model_id = "esmfold"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

    def fold(self, sequences: list[bytes], num_recycles: int = 1) -> FoldResultFuture:
        """
        Fold sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to fold
        num_recycles : int
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
