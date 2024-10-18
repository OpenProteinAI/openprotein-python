from openprotein.api import fold

from ..align import MSAFuture
from .base import FoldModel
from .future import FoldResultFuture


class AlphaFold2Model(FoldModel):

    model_id = "alphafold2"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

    def fold(
        self,
        msa: str | MSAFuture,
        num_recycles: int | None = None,
        num_models: int = 1,
        num_relax: int = 0,
    ) -> FoldResultFuture:
        """
        Post sequences to alphafold model.

        Parameters
        ----------
        msa : Union[str, MSAFuture]
            msa
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
        msa_id = msa.id if isinstance(msa, MSAFuture) else msa

        return FoldResultFuture.create(
            session=self.session,
            job=fold.fold_models_alphafold2_post(
                self.session,
                msa_id=msa_id,
                num_recycles=num_recycles,
                num_models=num_models,
                num_relax=num_relax,
            ),
        )
