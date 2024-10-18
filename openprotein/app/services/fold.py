"""Application services for Fold."""

from openprotein.api import fold
from openprotein.app.models import (
    AlphaFold2Model,
    ESMFoldModel,
    FoldModel,
    FoldResultFuture,
)
from openprotein.base import APISession


class FoldAPI:
    """
    This class defines a high level interface for accessing the fold API.
    """

    esmfold: ESMFoldModel
    alphafold2: AlphaFold2Model

    def __init__(self, session: APISession):
        self.session = session
        self._load_models()

    @property
    def af2(self):
        """Alias for AlphaFold2"""
        return self.alphafold2

    def _load_models(self):
        # Dynamically add model instances as attributes - precludes any drift
        models = self.list_models()
        for model in models:
            model_name = model.id.replace("-", "_")  # hyphens out
            setattr(self, model_name, model)

    def list_models(self) -> list[FoldModel]:
        """list models available for creating folds of your sequences"""
        models = []
        for model_id in fold.fold_models_list_get(self.session):
            models.append(
                FoldModel.create(
                    session=self.session, model_id=model_id, default=FoldModel
                )
            )
        return models

    def get_model(self, model_id: str) -> FoldModel:
        """
        Get model by model_id. 

        FoldModel allows all the usual job manipulation: \
            e.g. making POST and GET requests for this model specifically. 


        Parameters
        ----------
        model_id : str
            the model identifier

        Returns
        -------
        FoldModel
            The model

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return FoldModel.create(
            session=self.session, model_id=model_id, default=FoldModel
        )

    def get_results(self, job) -> FoldResultFuture:
        """
        Retrieves the results of a fold job.

        Parameters
        ----------
        job : Job
            The fold job whose results are to be retrieved.

        Returns
        -------
        FoldResultFuture
            An instance of FoldResultFuture
        """
        return FoldResultFuture.create(job=job, session=self.session)
