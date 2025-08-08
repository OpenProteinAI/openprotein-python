"""Predictor API providing the interface to train and predict predictors."""

from openprotein.base import APISession
from openprotein.common import FeatureType, ReductionType
from openprotein.data import (
    AssayDataset,
    AssayMetadata,
)
from openprotein.embeddings import EmbeddingModel, EmbeddingsAPI
from openprotein.errors import InvalidParameterError
from openprotein.svd import SVDAPI, SVDModel

from . import api
from .models import PredictorModel


class PredictorAPI:
    """Predictor API providing the interface to train and predict predictors."""

    def __init__(
        self,
        session: APISession,
    ):
        self.session = session

    def get_predictor(
        self,
        predictor_id: str,
        include_stats: bool = False,
        include_calibration_curves: bool = False,
    ) -> PredictorModel:
        """
        Get predictor by model_id.

        PredictorModel allows all the usual prediction job manipulation:
        e.g. making POST and GET requests for this predictor specifically.

        Parameters
        ----------
        predictor_id : str
            The model identifier.
        include_stats : bool
            Whether to include stats of the predictor from the latest evaluation
            (pearson, spearman, ece). Default is False.
        include_calibration_curves : bool
            Whether to include calibration curves of the predictor from the latest
            evaluation. Default is False.

        Returns
        -------
        PredictorModel
            The predictor model to inspect and make predictions with.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        metadata = api.predictor_get(
            session=self.session,
            predictor_id=predictor_id,
            include_stats=include_stats,
            include_calibration_curves=include_calibration_curves,
        )
        return PredictorModel(
            session=self.session,
            metadata=metadata,
        )

    def list_predictors(
        self,
        limit: int = 100,
        offset: int = 0,
        include_stats: bool = False,
        include_calibration_curves: bool = False,
    ) -> list[PredictorModel]:
        """
        List predictors available.

        Parameters
        ----------
        limit : int
            Limit of the number of predictors to return in list. Default is 100.
        offset : int
            Offset to the predictors to query for paged queries. Default is 0.
        include_stats : bool
            Whether to include stats of each predictor from the latest evaluation
            (pearson, spearman, ece). Default is False.
        include_calibration_curves : bool
            Whether to include calibration curves of each predictor from the latest
            evaluation. Default is False.

        Returns
        -------
        list[PredictorModel]
            List of predictor models to inspect and make predictions with.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        metadatas = api.predictor_list(
            session=self.session,
            limit=limit,
            offset=offset,
            include_stats=include_stats,
            include_calibration_curves=include_calibration_curves,
        )
        return [
            PredictorModel(
                session=self.session,
                metadata=m,
            )
            for m in metadatas
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
        assay : AssayMetadata or AssayDataset or str
            Assay to fit GP on.
        properties : list of str
            Properties in the assay to fit the gp on.
        model : EmbeddingModel or SVDModel or str
            Instance of either EmbeddingModel or SVDModel to use depending
            on feature type. Can also be a str specifying the model id,
            but then feature_type would have to be specified.
        feature_type : FeatureType or None
            Type of features to use for encoding sequences. "SVD" or "PLM".
            None would require model to be EmbeddingModel or SVDModel.
        reduction  : str or None, optional
            Type of embedding reduction to use for computing features.
            E.g. "MEAN" or "SUM". Used only if using EmbeddingModel, and
            must be non-nil if using an EmbeddingModel. Defaults to None.
        kwargs :
            Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

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
                embeddings_api = getattr(self.session, "embedding", None)
                assert isinstance(embeddings_api, EmbeddingsAPI)
                model = embeddings_api.get_model(model)
            assert isinstance(model, EmbeddingModel), "Expected EmbeddingModel"
            model_id = model.id
        elif feature_type == FeatureType.SVD:
            if isinstance(model, str):
                svd_api = getattr(self.session, "svd", None)
                assert isinstance(svd_api, SVDAPI)
                model = svd_api.get_svd(model)
            assert isinstance(model, SVDModel), "Expected SVDModel"
            model_id = model.id
        # get assay_id
        assay_id = (
            assay.assay_id
            if isinstance(assay, AssayMetadata)
            else assay.id if isinstance(assay, AssayDataset) else assay
        )
        return PredictorModel(
            session=self.session,
            job=api.predictor_fit_gp_post(
                session=self.session,
                assay_id=assay_id,
                properties=properties,
                feature_type=feature_type,
                model_id=model_id,
                reduction=reduction,
                name=name,
                description=description,
                **kwargs,
            ),
        )

    def delete_predictor(self, predictor_id: str) -> bool:
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
        return api.predictor_delete(session=self.session, predictor_id=predictor_id)

    def ensemble(self, predictors: list[PredictorModel]) -> PredictorModel:
        """
        Ensemble predictor models together.

        Parameters
        __________
        predictors: list[PredictorModel]
            List of predictors to ensemble together.
        Returns
        -------
        PredictorModel
            Ensembled predictor model
        """
        return PredictorModel(
            session=self.session,
            metadata=api.predictor_ensemble(
                session=self.session,
                predictor_ids=[predictor.id for predictor in predictors],
            ),
        )
