"""Embeddings model representations which can be used directly for creating embeddings."""

from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.common import (
    Feature,
    FeatureType,
    ModelMetadata,
    Reduction,
    ReductionType,
)
from openprotein.data import AssayDataset, AssayMetadata, DataAPI
from openprotein.errors import InvalidParameterError

from . import api
from .future import EmbeddingsResultFuture

if TYPE_CHECKING:
    from openprotein.predictor import PredictorModel
    from openprotein.svd import SVDModel
    from openprotein.umap import UMAPModel


class EmbeddingModel:
    """Base embeddings model used to understand and provide embeddings from sequences."""

    # overridden by subclasses
    # used to get correct emb model during factory create
    model_id: list[str] | str = "protembed"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        self.__doc__ = self.__fmt_doc()

    def __fmt_doc(self):
        summary = str(self.metadata.description.summary)
        return f"""\t{summary}
        \t max_sequence_length = {self.metadata.max_sequence_length}
        \t supported outputs = {self.metadata.output_types}
        \t supported tokens = {self.metadata.input_tokens} 
        """

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    @classmethod
    def get_model(cls):
        """
        Get the model_id(s) for this EmbeddingModel subclass.

        Returns
        -------
        list of str
            List of model_id strings associated with this class.
        """
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id

    @classmethod
    def create(
        cls,
        session: APISession,
        model_id: str,
        default: type["EmbeddingModel"] | None = None,
        **kwargs,
    ):
        """
        Create and return an instance of the appropriate EmbeddingModel subclass based on the model_id.

        Parameters
        ----------
        session : APISession
            The API session to use.
        model_id : str
            The model identifier.
        default : type variable of EmbeddingModel or None, optional
            Default EmbeddingModel subclass to use if no match is found.
        kwargs :
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        EmbeddingModel
            An instance of the appropriate EmbeddingModel subclass.

        Raises
        ------
        ValueError
            If no suitable EmbeddingModel subclass is found and no default is provided.
        """
        # Dynamically discover all subclasses of EmbeddingModel
        model_classes = EmbeddingModel.__subclasses__()

        # Find the EmbeddingModel class that matches the model_id
        for model_class in model_classes:
            if model_id in model_class.get_model():
                return model_class(session=session, model_id=model_id, **kwargs)
        # default to ProtembedModel
        if default is not None:
            try:
                return default(session=session, model_id=model_id, **kwargs)
            except:
                # continue to throw error as unsupported
                pass
        raise ValueError(f"Unsupported model_id type: {model_id}")

    @property
    def metadata(self):
        """
        ModelMetadata for this model.

        Returns
        -------
        ModelMetadata
            The metadata associated with this model.
        """
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for this model.

        Returns
        -------
        ModelMetadata
            The metadata associated with this model.
        """
        return api.get_model(self.session, self.id)

    def embed(
        self,
        sequences: list[bytes] | list[str],
        reduction: ReductionType | None = ReductionType.MEAN,
        **kwargs,
    ) -> EmbeddingsResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        sequences : list of bytes or list of str
            Sequences to embed.
        reduction : ReductionType or None, optional
            Reduction to use (e.g. mean). Defaults to mean embedding.
        kwargs:
            Additional keyword arguments to be used from foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        EmbeddingsResultFuture
            Future object representing the embedding result.
        """
        return EmbeddingsResultFuture.create(
            session=self.session,
            job=api.request_post(
                session=self.session,
                model_id=self.id,
                sequences=sequences,
                reduction=reduction,
                **kwargs,
            ),
            sequences=sequences,
        )

    def logits(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingsResultFuture:
        """
        Compute logit embeddings for sequences using this model.

        Parameters
        ----------
        sequences : list of bytes or list of str
            Sequences to compute logits for.
        kwargs :
            Additional keyword arguments to be used from foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        EmbeddingsResultFuture
            Future object representing the logits result.
        """
        return EmbeddingsResultFuture.create(
            session=self.session,
            job=api.request_logits_post(
                session=self.session, model_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )

    def fit_svd(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | AssayMetadata | None = None,
        n_components: int = 1024,
        reduction: Reduction | ReductionType | None = None,
        **kwargs,
    ) -> "SVDModel":
        """
        Fit an SVD on the embedding results of this model.

        This function will create an SVDModel based on the embeddings from this model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        sequences : list of bytes or list of str or None, optional
            Sequences to fit SVD on.
        assay : AssayDataset or None, optional
            Assay containing sequences to fit SVD on.
        n_components : int, optional
            Number of components in SVD. Determines output shapes. Default is 1024.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g. mean).
        kwargs :
            Additional keyword arguments to be used from foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        SVDModel
            The fitted SVD model.

        Raises
        ------
        InvalidParameterError
            If neither or both of `assay` and `sequences` are provided.
        """
        # local import for cyclic dep
        from openprotein.svd import SVDAPI

        # runtime check on value
        if isinstance(reduction, str):
            reduction = ReductionType(reduction)
            reduction = reduction.value

        svd_api = getattr(self.session, "svd", None)
        assert isinstance(svd_api, SVDAPI)

        # Ensure either or
        if (assay is None and sequences is None) or (
            assay is not None and sequences is not None
        ):
            raise InvalidParameterError(
                "Expected either assay or sequences to fit SVD on!"
            )
        return svd_api.fit_svd(
            model=self,
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
        )

    def fit_umap(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | AssayMetadata | None = None,
        n_components: int = 2,
        reduction: Reduction | ReductionType = "MEAN",
        **kwargs,
    ) -> "UMAPModel":
        """
        Fit a UMAP on the embedding results of this model.

        This function will create a UMAPModel based on the embeddings from this model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        sequences : list of bytes or list of str or None, optional
            Optional sequences to fit UMAP with. Either use sequences or assay. Sequences is preferred.
        assay : AssayDataset or AssayMetadata or None, optional
            Optional assay containing sequences to fit UMAP with. Either use sequences or assay. Ignored if sequences are provided.
        n_components : int, optional
            Number of components in UMAP fit. Determines output shapes. Default is 2.
        reduction : Reduction or ReductionType or None, optional
            Embeddings reduction to use (e.g. mean). Defaults to MEAN.
        kwargs :
            Additional keyword arguments to be used from foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        UMAPModel
            The fitted UMAP model.

        Raises
        ------
        InvalidParameterError
            If neither or both of `assay` and `sequences` are provided.
        """
        # local import for cyclic dep
        from openprotein.umap import UMAPAPI

        if reduction is None:
            raise InvalidParameterError(
                "Expected reduction if using EmbeddingModel to fit UMAP"
            )

        # runtime check on value
        if isinstance(reduction, str):
            reduction = ReductionType(reduction)
            reduction = reduction.value

        umap_api = getattr(self.session, "umap", None)
        assert isinstance(umap_api, UMAPAPI)

        # Ensure either or
        if (assay is None and sequences is None) or (
            assay is not None and sequences is not None
        ):
            raise InvalidParameterError(
                "Expected either assay or sequences to fit UMAP on!"
            )
        # get assay_id
        assay_id = (
            assay.assay_id
            if isinstance(assay, AssayMetadata)
            else assay.id if isinstance(assay, AssayDataset) else assay
        )
        model_id = self.id
        return umap_api.fit_umap(
            model_id=model_id,
            feature_type=FeatureType.PLM,
            sequences=sequences,
            assay_id=assay_id,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
        )

    def fit_gp(
        self,
        assay: AssayDataset | AssayMetadata | str,
        properties: list[str],
        reduction: ReductionType,
        name: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> "PredictorModel":
        """
        Fit a Gaussian Process (GP) on an assay using this embedding model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata, AssayDataset, or str
            Assay to fit GP on.
        properties : list of str
            Properties in the assay to fit the GP on.
        reduction : ReductionType
            Type of embedding reduction to use for computing features. PLM must use reduction.
        name : str or None, optional
            Optional name for the predictor model.
        description : str or None, optional
            Optional description for the predictor model.
        kwargs :
            Additional keyword arguments to be used from foundational models, e.g. prompt_id for PoET models.

        Returns
        -------
        PredictorModel
            The fitted predictor model.

        Raises
        ------
        InvalidParameterError
            If no properties are provided, properties are not a subset of assay measurements,
            or multitask GP is requested.
        """
        # local import to resolve cyclic
        from openprotein.predictor import PredictorAPI

        predictor_api = getattr(self.session, "predictor", None)
        assert isinstance(predictor_api, PredictorAPI)

        # inject into predictor api
        return predictor_api.fit_gp(
            assay=assay,
            properties=properties,
            feature_type=FeatureType.PLM,
            model=self,
            reduction=reduction,
            name=name,
            description=description,
            **kwargs,
        )


class AttnModel(EmbeddingModel):
    """Embeddings model that provides attention computation."""

    def attn(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingsResultFuture:
        """
        Compute attention embeddings for sequences using this model.

        Parameters
        ----------
        sequences : list of bytes or list of str
            Sequences to compute attention embeddings for.
        kwargs :
            Additional keyword arguments to be used from foundational models.

        Returns
        -------
        EmbeddingsResultFuture
            Future object representing the attention result.
        """
        return EmbeddingsResultFuture.create(
            session=self.session,
            job=api.request_attn_post(
                session=self.session, model_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )
