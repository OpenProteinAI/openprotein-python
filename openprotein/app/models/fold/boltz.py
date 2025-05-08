from openprotein.api import fold

from ..align import MSAFuture
from .base import FoldModel
from .future import FoldResultFuture


class BoltzModel(FoldModel):

    model_id = "boltz"

    def __init__(self, session, model_id, metadata=None):
        super().__init__(session, model_id, metadata)
        self.id = self.model_id

    def fold(
        self,
        msa: str | MSAFuture,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        step_scale: float = 1.638,
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
            job=fold.fold_models_post(
                session=self.session,
                model_id="boltz",
                msa_id=msa_id,
                diffusion_samples=diffusion_samples,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                step_scale=step_scale,
            ),
        )
