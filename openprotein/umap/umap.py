"""UMAP API providing the interface to fit and run UMAP visualizations."""

from openprotein.base import APISession
from openprotein.common import FeatureType, ReductionType
from openprotein.data import AssayDataset
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
        assay: AssayDataset | None = None,
        n_components: int = 2,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> UMAPModel:
        """
        Fit an UMAP on the sequences with the specified model_id and hyperparameters (n_components).

        Parameters
        ----------
        model_id : str
            The ID of the model to fit the UMAP on.
        sequences : list[bytes]
            The list of sequences to use for the UMAP fitting.
        n_components : int, optional
            The number of components for the UMAP, by default 2.
        reduction : str, optional
            The reduction method to apply to the embeddings, by default None.

        Returns
        -------
        UMAPModel
            The model with the UMAP fit.
        """
        if isinstance(model, str):
            if feature_type is None:
                raise InvalidParameterError(
                    "Expected feature_type to be specified if using a string identifier as model"
                )
            model_id = model
        else:
            model_id = model.id  # for embeddings / svd model
        return UMAPModel.create(
            session=self.session,
            job=api.umap_fit_post(
                model_id=model_id,
                sequences=sequences,
                assay=assay,
                n_components=n_components,
                reduction=reduction,
                **kwargs,
            ),
        )

    def get_umap(self, umap_id: str) -> UMAPModel:
        """
        Get UMAP job results. Including UMAP dimension and sequence lengths.

        Requires a successful UMAP job from fit_umap

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
