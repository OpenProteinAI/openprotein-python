"""Fold API interface for making structure prediction requests."""

from openprotein.base import APISession

from . import api
from .alphafold2 import AlphaFold2Model
from .boltz import Boltz1Model, Boltz1xModel, Boltz2Model
from .esmfold import ESMFoldModel
from .future import FoldComplexResultFuture, FoldResultFuture
from .models import (
    FoldModel,
)


class FoldAPI:
    """
    Fold API provides a high level interface for making protein structure predictions.
    """

    #: Boltz-2 model
    boltz2: Boltz2Model
    boltz_2: Boltz2Model
    #: Boltz-1x model
    boltz1x: Boltz1xModel
    boltz_1x: Boltz1xModel
    #: Boltz-1 model
    boltz1: Boltz1Model
    boltz_1: Boltz1Model
    af2: AlphaFold2Model
    #: AlphaFold-2 model
    alphafold2: AlphaFold2Model
    #: ESMFold model
    esmfold: ESMFoldModel

    def __init__(self, session: APISession):
        self.session = session
        self._load_models()

    def _load_models(self):
        # Dynamically add model instances as attributes - precludes any drift
        models = self.list_models()
        for model in models:
            model_name = model.id.replace("-", "_")  # hyphens out
            setattr(self, model_name, model)
        # Setup aliases safely
        if getattr(self, "alphafold2", None):
            self.af2 = self.alphafold2
        if getattr(self, "boltz_1", None):
            self.boltz1 = self.boltz_1
        if getattr(self, "boltz_1x", None):
            self.boltz1x = self.boltz_1x
        if getattr(self, "boltz_2", None):
            self.boltz2 = self.boltz_2

    def list_models(self) -> list[FoldModel]:
        """list models available for creating folds of your sequences"""
        models = []
        for model_id in api.fold_models_list_get(self.session):
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

    def get_results(self, job) -> FoldResultFuture | FoldComplexResultFuture:
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
