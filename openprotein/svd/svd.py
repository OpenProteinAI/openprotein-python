"""SVD API providing the interface for creating and using SVD models."""

from openprotein.base import APISession
from openprotein.common import ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.embeddings import EmbeddingModel, EmbeddingsAPI

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
        assay: AssayMetadata | AssayDataset | str | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> SVDModel:
        """
        Fit an SVD on the sequences with the specified model_id and hyperparameters (n_components).

        Parameters
        ----------
        model_id : str
            ID of embeddings model to use.
        sequences : list of bytes or None, optional
            Optional sequences to fit SVD with. Either use sequences or
            assay_id. sequences is preferred.
        assay : AssayMetadata or AssayDataset or str or None, optional
            Optional assay containing sequences to fit SVD with.
            Or its assay_id. Either use sequences or assay.
            Ignored if sequences are provided.
        n_components : int, optional
            The number of components for the SVD. Defaults to 1024.
        reduction : str or None, optional
            Type of embedding reduction to use for computing features.
            E.g. "MEAN" or "SUM". Useful when dealing with variable length
            sequence. Defaults to None.
        kwargs :
            Additional keyword arguments to be passed to foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        SVDModel
            The SVD model being fit.
        """
        embeddings_api = getattr(self.session, "embedding", None)
        assert isinstance(embeddings_api, EmbeddingsAPI)
        model = embeddings_api.get_model(model_id)
        assert isinstance(model, EmbeddingModel), "Expected EmbeddingModel"
        # get assay_id
        assay_id = (
            assay.assay_id
            if isinstance(assay, AssayMetadata)
            else assay.id if isinstance(assay, AssayDataset) else assay
        )
        return SVDModel(
            session=self.session,
            job=api.svd_fit_post(
                session=self.session,
                model_id=model.id,
                sequences=sequences,
                assay_id=assay_id,
                n_components=n_components,
                reduction=reduction,
                **kwargs,
            ),
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
            The ID of the SVD job.
        Returns
        -------
        bool
            Whether or not the SVD was successfully deleted.

        """
        return api.svd_delete(self.session, svd_id)

    def list_svd(self) -> list[SVDModel]:
        """
        List SVD models made by user.

        Returns
        -------
        list of SVDModel
            List of SVDs that the user has access to.

        """
        return [
            SVDModel(
                session=self.session,
                metadata=metadata,
            )
            for metadata in api.svd_list_get(self.session)
        ]
