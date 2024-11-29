from openprotein.api import umap
from openprotein.app.models import AssayDataset, EmbeddingModel, SVDModel, UMAPModel
from openprotein.base import APISession
from openprotein.schemas import ReductionType


class UMAPAPI:

    def __init__(self, session: APISession):
        self.session = session

    def fit_umap(
        self,
        model: EmbeddingModel | SVDModel,
        sequences: list[bytes] | None = None,
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
            The number of components for the UMAP, by default 1024.
        reduction : str, optional
            The reduction method to apply to the embeddings, by default None.

        Returns
        -------
        UMAPModel
            The model with the UMAP fit.
        """
        return model.fit_umap(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
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
        metadata = umap.umap_get(self.session, umap_id)
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
        return umap.umap_delete(self.session, umap_id)

    def list_umap(self) -> list[UMAPModel]:
        """
        List UMAP models made by user.

        Takes no args.

        Returns
        -------
        list[UMAPModel]
            UMAPModels

        """
        return [
            UMAPModel(session=self.session, metadata=metadata)
            for metadata in umap.umap_list_get(self.session)
        ]
