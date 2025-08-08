"""UMAP API providing the interface to fit and run UMAP visualizations."""

from openprotein.base import APISession
from openprotein.common import FeatureType, ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.embeddings import EmbeddingModel, EmbeddingsAPI
from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobsAPI
from openprotein.svd import SVDAPI, SVDModel

from . import api
from .models import UMAPModel


class UMAPAPI:
    """UMAP API providing the interface to fit and run UMAP visualizations."""

    def __init__(
        self,
        session: APISession,
    ):
        self.session = session

    def fit_umap(
        self,
        model: EmbeddingModel | SVDModel | str,
        feature_type: FeatureType | None = None,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayMetadata | AssayDataset | str | None = None,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> UMAPModel:
        """
        Fit an UMAP on the sequences with the specified model_id and hyperparameters (n_components).

        Parameters
        ----------
        sequences: list of bytes or None, optional
            Optional sequences to fit UMAP with. Either use sequences or
            assay_id. sequences is preferred.
        assay : AssayMetadata or AssayDataset or str or None, optional
            Optional assay containing sequences to fit SVD with.
            Or its assay_id. Either use sequences or assay.
            Ignored if sequences are provided.
        model : EmbeddingModel or SVDModel or str
            Instance of either EmbeddingModel or SVDModel to use depending
            on feature type. Can also be a str specifying the model id,
            but then feature_type would have to be specified.
        feature_type : FeatureType or None, optional
            Type of features to use for encoding sequences. "SVD" or "PLM".
            None would require model to be EmbeddingModel or SVDModel.
        n_components : int, optional
            Number of UMAP components to fit. Defaults to 2.
        n_neighbors : int, optional
            Number of neighbors to use for fitting. Defaults to 15.
        min_dist : float, optional
            Minimum distance in UMAP fitting. Defaults to 0.1.
        reduction : str or None, optional
            Type of embedding reduction to use for computing features.
            E.g. "MEAN" or "SUM". Useful when dealing with variable length
            sequence. Defaults to None.
        kwargs :
            Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        UMAPModel
            The UMAP model being fit.
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
        return UMAPModel(
            session=self.session,
            job=api.umap_fit_post(
                session=self.session,
                model_id=model_id,
                feature_type=feature_type,
                sequences=sequences,
                assay_id=assay_id,
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                reduction=reduction,
                **kwargs,
            ),
        )

    def get_umap(self, umap_id: str) -> UMAPModel:
        """
        Get UMAP job results. Including UMAP dimension and sequence lengths.

        Requires a successful UMAP job from fit_umap.

        Parameters
        ----------
        umap_id : str
            The ID of the UMAP  job.
        Returns
        -------
        UMAPModel
            The model with the UMAP fit.
        """
        metadata = api.umap_get(self.session, umap_id)
        return UMAPModel(session=self.session, metadata=metadata)

    def __delete_umap(self, umap_id: str) -> bool:
        """
        Delete UMAP model.

        Parameters
        ----------
        umap_id : str
            The ID of the UMAP  job.
        Returns
        -------
        bool
            True: successful deletion

        """
        return api.umap_delete(self.session, umap_id)

    def list_umap(self) -> list[UMAPModel]:
        """
        List UMAP models made by user.

        Takes no args.

        Returns
        -------
        list[UMAPModel]
            UMAPModels

        """
        jobs_api = getattr(self.session, "jobs", None)
        assert isinstance(jobs_api, JobsAPI)
        return [
            UMAPModel(session=self.session, metadata=metadata)
            for metadata in api.umap_list_get(self.session)
        ]
