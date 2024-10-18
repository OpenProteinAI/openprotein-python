from typing import TYPE_CHECKING

from openprotein.api import assaydata, embedding, predictor, svd
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import FeatureType, ModelMetadata, ReductionType

from ..assaydata import AssayDataset, AssayMetadata
from .future import EmbeddingResultFuture

if TYPE_CHECKING:
    from ..predictor import PredictorModel
    from ..svd import SVDModel


class EmbeddingModel:

    # overridden by subclasses
    # get correct emb model
    model_id: list[str] | str = "protembed"

    def __init__(
        self, session: APISession, model_id: str, metadata: ModelMetadata | None = None
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
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id

    @classmethod
    def create(
        cls,
        session: APISession,
        model_id: str,
        default: type["EmbeddingModel"] | None = None,
    ):
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Returns:
        - An instance of the appropriate Future class.
        """
        # Dynamically discover all subclasses of EmbeddingModel
        model_classes = EmbeddingModel.__subclasses__()

        # Find the EmbeddingModel class that matches the model_id
        for model_class in model_classes:
            if model_id in model_class.get_model():
                return model_class(session=session, model_id=model_id)
        # default to ProtembedModel
        if default is not None:
            try:
                return default(session=session, model_id=model_id)
            except:
                # continue to throw error as unsupported
                pass
        raise ValueError(f"Unsupported model_id type: {model_id}")

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for this model.

        Returns
        -------
            ModelMetadata
        """
        if self._metadata is not None:
            return self._metadata
        self._metadata = embedding.get_model(self.session, self.id)
        return self._metadata

    def embed(
        self,
        sequences: list[bytes] | list[str],
        reduction: ReductionType | None = ReductionType.MEAN,
        **kwargs,
    ) -> EmbeddingResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD
        reduction: ReductionType | None, Optional
            embeddings reduction to use (e.g. mean)

        Returns
        -------
            EmbeddingResultFuture
        """
        return EmbeddingResultFuture.create(
            session=self.session,
            job=embedding.request_post(
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
    ) -> EmbeddingResultFuture:
        """
        logit embeddings for sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD

        Returns
        -------
            EmbeddingResultFuture
        """
        return EmbeddingResultFuture.create(
            session=self.session,
            job=embedding.request_logits_post(
                session=self.session, model_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )

    def attn(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingResultFuture:
        """
        Attention embeddings for sequences using this model.

        Parameters
        ----------
        sequences : List[bytes]
            sequences to SVD

        Returns
        -------
            EmbeddingResultFuture
        """
        return EmbeddingResultFuture.create(
            session=self.session,
            job=embedding.request_attn_post(
                session=self.session, model_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )

    def fit_svd(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> "SVDModel":
        """
        Fit an SVD on the embedding results of this model. 

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        sequences : List[bytes] 
            sequences to SVD
        n_components: int 
            number of components in SVD. Will determine output shapes
        reduction: ReductionType | None
            embeddings reduction to use (e.g. mean)

        Returns
        -------
            SVDModel
        """
        # local import for cyclic dep
        from ..svd import SVDModel

        # Ensure either or
        if (assay is None and sequences is None) or (
            assay is not None and sequences is not None
        ):
            raise InvalidParameterError(
                "Expected either assay or sequences to fit SVD on!"
            )
        model_id = self.id
        job = svd.svd_fit_post(
            session=self.session,
            model_id=model_id,
            sequences=sequences,
            assay_id=assay.id if assay is not None else None,
            n_components=n_components,
            reduction=reduction,
            **kwargs,
        )
        return SVDModel.create(session=self.session, job=job)

    def fit_gp(
        self,
        assay: AssayMetadata | AssayDataset | str,
        properties: list[str],
        reduction: ReductionType,
        name: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> "PredictorModel":
        """
        Fit a GP on assay using this embedding model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata | str
            Assay to fit GP on.
        properties: list[str]
            Properties in the assay to fit the gp on.
        reduction : str
            Type of embedding reduction to use for computing features. PLM must use reduction.

        Returns
        -------
            PredictorModel
        """
        # local import to resolve cyclic
        from ..predictor import PredictorModel

        model_id = self.id
        # get assay if str
        assay = (
            assaydata.get_assay_metadata(session=self.session, assay_id=assay)
            if isinstance(assay, str)
            else assay
        )
        # extract assay_id
        assay_id = assay.assay_id if isinstance(assay, AssayMetadata) else assay.id
        if len(properties) == 0:
            raise InvalidParameterError("Expected (at-least) 1 property to train")
        if not set(properties) <= set(assay.measurement_names):
            raise InvalidParameterError(
                f"Expected all provided properties to be a subset of assay's measurements: {assay.measurement_names}"
            )
        # TODO - support multitask
        if len(properties) > 1:
            raise InvalidParameterError(
                "Training a multitask GP is not yet supported (i.e. number of properties should only be 1 for now)"
            )
        job = predictor.predictor_fit_gp_post(
            session=self.session,
            assay_id=assay_id,
            properties=properties,
            feature_type=FeatureType.PLM,
            model_id=model_id,
            reduction=reduction,
            name=name,
            description=description,
            **kwargs,
        )
        return PredictorModel.create(session=self.session, job=job)
