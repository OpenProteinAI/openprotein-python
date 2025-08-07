"""UMAP API providing the interface to fit and run UMAP visualizations."""

from openprotein.base import APISession
from openprotein.common import FeatureType, ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.embeddings import EmbeddingModel
from openprotein.errors import InvalidParameterError
from openprotein.jobs import JobsAPI
from openprotein.svd import SVDModel

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
        model_id : str
            The ID of the model to fit the UMAP on.
        sequences: list of bytes or None, optional
            Optional sequences to fit UMAP with. Either use sequences or
            assay_id. sequences is preferred.
        assay : AssayMetadata or AssayDataset or str or None, optional
            Optional assay containing sequences to fit SVD with.
            Or its assay_id. Either use sequences or assay.
            Ignored if sequences are provided.
        n_components: int
            Number of UMAP components to fit. Defaults to 2.
        n_neighbors: int
            Number of neighbors to use for fitting. Defaults to 15.
        min_dist: float
            Minimum distance in UMAP fitting. Defaults to 0.1.
        reduction: str or None, optional
            Type of embedding reduction to use for computing features.
            E.g. "MEAN" or "SUM". Useful when dealing with variable length
            sequence. Defaults to None.
        kwargs:
            Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        UMAPModel
            The UMAP model being fit.
        """
        if isinstance(model, str):
            if feature_type is None:
                raise InvalidParameterError(
                    "Expected feature_type to be specified if using a string identifier as model"
                )
            model_id = model
        else:
            model_id = model.id  # for embeddings / svd model
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
