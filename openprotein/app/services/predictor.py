from openprotein.api import predictor
from openprotein.app.models import (
    AssayDataset,
    AssayMetadata,
    EmbeddingModel,
    PredictorModel,
    SVDModel,
)
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import FeatureType, ReductionType

from .embeddings import EmbeddingsAPI
from .svd import SVDAPI


class PredictorAPI:
    """
    This class defines a high level interface for accessing the predictors API.
    """

    def __init__(self, session: APISession, embeddings: EmbeddingsAPI, svd: SVDAPI):
        self.session = session
        self.embeddings = embeddings
        self.svd = svd

    def get_predictor(self, predictor_id: str) -> PredictorModel:
        """
        Get predictor by model_id. 

        PredictorModel allows all the usual prediction job manipulation: \
            e.g. making POST and GET requests for this predictor specifically. 


        Parameters
        ----------
        predictor_id : str
            the model identifier

        Returns
        -------
        PredictorModel
            The predictor model to inspect and make predictions with.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return PredictorModel(
            session=self.session,
            metadata=predictor.predictor_get(
                session=self.session, predictor_id=predictor_id
            ),
        )

    def list_predictors(self) -> list[PredictorModel]:
        """
        List predictors available.

        Returns
        -------
        list[PredictorModel}
            List of predictor models to inspect and make predictions with.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return [
            PredictorModel(
                session=self.session,
                metadata=m,
            )
            for m in predictor.predictor_list(session=self.session)
        ]

    def fit_gp(
        self,
        assay: AssayDataset | AssayMetadata | str,
        properties: list[str],
        model: EmbeddingModel | SVDModel | str,
        feature_type: FeatureType | None = None,
        reduction: ReductionType | None = None,
        name: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> PredictorModel:
        """
        Fit a GP on an assay with the specified feature model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata | str
            Assay to fit GP on.
        properties: list[str]
            Properties in the assay to fit the gp on.
        feature_type: str
            Type of features to use for encoding sequences. "SVD" or "PLM".
        model : str
            Protembed/SVD model to use depending on feature type.
        reduction : str | None
            Type of embedding reduction to use for computing features. default = None
        prompt: PromptFuture | str | None
            Prompt if using PoET-based models.

        Returns
        -------
        PredictorModel
            The GP model being fit.
        """
        # extract feature type
        feature_type = (
            FeatureType.PLM
            if isinstance(model, EmbeddingModel)
            else FeatureType.SVD if isinstance(model, SVDModel) else feature_type
        )
        if feature_type is None:
            raise InvalidParameterError(
                "Expected feature_type to be provided if passing str model_id as model"
            )
        # get model if model_id
        if feature_type == FeatureType.PLM:
            if reduction is None:
                raise InvalidParameterError(
                    "Expected reduction if using EmbeddingModel"
                )
            if isinstance(model, str):
                model = self.embeddings.get_model(model)
            assert isinstance(model, EmbeddingModel), "Expected EmbeddingModel"
            return model.fit_gp(
                assay=assay,
                properties=properties,
                reduction=reduction,
                name=name,
                description=description,
                **kwargs,
            )
        elif feature_type == FeatureType.SVD:
            if isinstance(model, str):
                model = self.svd.get_svd(model)
            assert isinstance(model, SVDModel), "Expected SVDModel"
            return model.fit_gp(
                assay=assay,
                properties=properties,
                name=name,
                description=description,
                **kwargs,
            )

    def __delete_predictor(self, predictor_id: str) -> bool:
        """
        Delete predictor model.

        Parameters
        ----------
        predictor_id : str
            The ID of the predictor.
        Returns
        -------
        bool
            True: successful deletion

        """
        return predictor.predictor_delete(
            session=self.session, predictor_id=predictor_id
        )
