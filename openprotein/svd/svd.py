"""SVD API providing the interface for creating and using SVD models."""

from openprotein.base import APISession
from openprotein.common import ReductionType
from openprotein.data import AssayDataset
from openprotein.embeddings import EmbeddingsAPI

from . import api
from .models import SVDModel


class SVDAPI:
    """SVD API providing the interface for creating and using SVD models."""

    def __init__(
        self,
        session: APISession,
    ):
        self.session = session

    def fit_svd(
        self,
        model_id: str,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> SVDModel:
        """
        Fit an SVD on the sequences with the specified model_id and hyperparameters (n_components).

        Parameters
        ----------
        model_id : str
            The ID of the model to fit the SVD on.
        sequences : list[bytes]
            The list of sequences to use for the SVD fitting.
        n_components : int, optional
            The number of components for the SVD, by default 1024.
        reduction : str, optional
            The reduction method to apply to the embeddings, by default None.

        Returns
        -------
        SVDModel
            The model with the SVD fit.
        """
        embeddings_api = getattr(self.session, "embedding", None)
        assert isinstance(embeddings_api, EmbeddingsAPI)
        model = embeddings_api.get_model(model_id)
        return model.fit_svd(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
        )

    def get_svd(self, svd_id: str) -> SVDModel:
        """
        Get SVD job results. Including SVD dimension and sequence lengths.

        Requires a successful SVD job from fit_svd

        Parameters
        ----------
        svd_id : str
            The ID of the SVD  job.
        Returns
        -------
        SVDModel
            The model with the SVD fit.
        """
        metadata = api.svd_get(self.session, svd_id)
        return SVDModel(
            session=self.session,
            metadata=metadata,
        )

    def __delete_svd(self, svd_id: str) -> bool:
        """
        Delete SVD model.

        Parameters
        ----------
        svd_id : str
            The ID of the SVD  job.
        Returns
        -------
        bool
            True: successful deletion

        """
        return api.svd_delete(self.session, svd_id)

    def list_svd(self) -> list[SVDModel]:
        """
        List SVD models made by user.

        Takes no args.

        Returns
        -------
        list[SVDModel]
            SVDModels

        """
        return [
            SVDModel(
                session=self.session,
                metadata=metadata,
            )
            for metadata in api.svd_list_get(self.session)
        ]
