from typing import TYPE_CHECKING

import numpy as np
from openprotein.api import assaydata
from openprotein.api import job as job_api
from openprotein.api import predictor, svd
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import FeatureType, FitJob, SVDEmbeddingsJob, SVDMetadata

from .assaydata import AssayDataset, AssayMetadata
from .embeddings import EmbeddingModel, EmbeddingResultFuture
from .futures import Future

if TYPE_CHECKING:
    from .predictor import PredictorModel


class SVDModel(Future):
    """
    Class providing embedding endpoint for SVD models. \
        Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    Implements a Future to allow waiting for a fit job.
    """

    job: FitJob

    def __init__(
        self,
        session: APISession,
        job: FitJob | None = None,
        metadata: SVDMetadata | None = None,
    ):
        """Initializes with either job get or svd metadata get."""
        if metadata is None:
            # use job to fetch metadata
            if job is None:
                raise ValueError("Expected svd metadata or job")
            metadata = svd.svd_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            job = FitJob.create(job_api.job_get(session=session, job_id=metadata.id))
        # getter initializes job if not provided
        super().__init__(session, job)

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self):
        return self._metadata.id

    @property
    def n_components(self):
        return self._metadata.n_components

    @property
    def sequence_length(self):
        return self._metadata.sequence_length

    @property
    def reduction(self):
        return self._metadata.reduction

    @property
    def metadata(self):
        self._refresh_metadata()
        return self._metadata

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = svd.svd_get(self.session, self._metadata.id)

    def get_model(self) -> EmbeddingModel:
        """Fetch embeddings model"""
        model = EmbeddingModel.create(session=self.session, model_id=self._metadata.id)
        return model

    @property
    def model(self) -> EmbeddingModel:
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this SVD model.
        """
        return svd.svd_delete(self.session, self.id)

    def get(self, verbose: bool = False):
        # overload for AsyncJobFuture
        return self

    def get_inputs(self) -> list[bytes]:
        """
        Get sequences used for svd job.

        Returns
        -------
            List[bytes]: list of sequences
        """
        return svd.svd_get_sequences(session=self.session, svd_id=self.id)

    def embed(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingResultFuture:
        """
        Use this SVD model to get reduced embeddings from input sequences.

        Parameters
        ----------
        sequences : List[bytes]
            List of protein sequences.

        Returns
        -------
        EmbeddingResultFuture
            Class for further job manipulation.
        """
        return EmbeddingResultFuture.create(
            session=self.session,
            job=svd.svd_embed_post(
                session=self.session, svd_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )

    def fit_gp(
        self,
        assay: AssayMetadata | AssayDataset | str,
        properties: list[str],
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

        Returns
        -------
            PredictorModel
        """
        # local import to resolve cyclic
        from .predictor import PredictorModel

        model_id = self.id
        # get assay if str
        assay = (
            assaydata.get_assay_metadata(session=self.session, assay_id=assay)
            if isinstance(assay, str)
            else assay
        )
        # extract assay_id
        assay_id = assay.assay_id if isinstance(assay, AssayMetadata) else assay.id
        if (
            self.sequence_length is not None
            and assay.sequence_length != self.sequence_length
        ):
            raise InvalidParameterError(
                f"Expected dataset to be of sequence length {self.sequence_length} due to svd fitted constraints"
            )
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
            feature_type=FeatureType.SVD,
            model_id=model_id,
            name=name,
            description=description,
            **kwargs,
        )
        return PredictorModel.create(session=self.session, job=job)


class SVDEmbeddingResultFuture(EmbeddingResultFuture, Future):
    """Future for manipulating results for embeddings-related requests."""

    job: SVDEmbeddingsJob

    def get_item(self, sequence: bytes) -> np.ndarray:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: embeddings
        """
        data = svd.embed_get_sequence_result(self.session, self.job.job_id, sequence)
        return svd.embed_decode(data)
