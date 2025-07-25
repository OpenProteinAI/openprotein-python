"""SVD model representations which can be used for creating reduced embeddings."""

from typing import TYPE_CHECKING

import numpy as np

from openprotein.base import APISession
from openprotein.common import FeatureType
from openprotein.data import AssayDataset, AssayMetadata, DataAPI
from openprotein.embeddings import EmbeddingModel, EmbeddingsResultFuture
from openprotein.errors import InvalidParameterError
from openprotein.jobs import Future, JobsAPI

from . import api
from .schemas import SVDEmbeddingsJob, SVDFitJob, SVDMetadata

if TYPE_CHECKING:
    from openprotein.predictor import PredictorModel
    from openprotein.umap import UMAPModel


class SVDModel(Future):
    """
    Class providing embedding endpoint for SVD models. \
        Also allows retrieving embeddings of sequences used to fit the SVD with `get`.
    Implements a Future to allow waiting for a fit job.
    """

    job: SVDFitJob

    def __init__(
        self,
        session: APISession,
        job: SVDFitJob | None = None,
        metadata: SVDMetadata | None = None,
    ):
        """Construct the SVD model using either job get or svd metadata get."""
        # initialize the metadata
        if metadata is None:
            # use job to fetch metadata
            if job is None:
                raise ValueError("Expected svd metadata or job")
            metadata = api.svd_get(session=session, svd_id=job.job_id)
        self._metadata = metadata
        if job is None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = SVDFitJob.create(jobs_api.get_job(job_id=metadata.id))
        # getter initializes job if not provided
        super().__init__(session=session, job=job)

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
            self._metadata = api.svd_get(session=self.session, svd_id=self._metadata.id)

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
        return api.svd_delete(self.session, self.id)

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
        return api.svd_get_sequences(session=self.session, svd_id=self.id)

    def embed(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingsResultFuture:
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
        return EmbeddingsResultFuture.create(
            session=self.session,
            job=api.svd_embed_post(
                session=self.session, svd_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )

    def fit_umap(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 2,
        **kwargs,
    ) -> "UMAPModel":
        """
        Fit an UMAP on the embedding results of this model. 

        This function will create an UMAPModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        sequences : List[bytes] 
            sequences to UMAP
        n_components: int 
            number of components in UMAP. Will determine output shapes
        reduction: ReductionType | None
            embeddings reduction to use (e.g. mean)

        Returns
        -------
            UMAPModel
        """
        # local import for cyclic dep
        from openprotein.umap import UMAPAPI

        umap_api = getattr(self.session, "umap", None)
        assert isinstance(umap_api, UMAPAPI)

        # Ensure either or
        if (assay is None and sequences is None) or (
            assay is not None and sequences is not None
        ):
            raise InvalidParameterError(
                "Expected either assay or sequences to fit UMAP on!"
            )
        model_id = self.id
        return umap_api.fit_umap(
            model=model_id,
            feature_type=FeatureType.SVD,
            sequences=sequences,
            assay_id=assay.id if assay is not None else None,
            n_components=n_components,
            **kwargs,
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
        from openprotein.predictor import PredictorAPI

        data_api = getattr(self.session, "jobs", None)
        assert isinstance(data_api, DataAPI)

        predictor_api = getattr(self.session, "predictor", None)
        assert isinstance(predictor_api, PredictorAPI)

        model_id = self.id
        # get assay if str
        assay = data_api.get(assay_id=assay) if isinstance(assay, str) else assay
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
        return predictor_api.fit_gp(
            assay_id=assay_id,
            properties=properties,
            feature_type=FeatureType.SVD,
            model_id=model_id,
            name=name,
            description=description,
            **kwargs,
        )


class SVDEmbeddingResultFuture(EmbeddingsResultFuture, Future):
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
        data = api.embed_get_sequence_result(self.session, self.job.job_id, sequence)
        return api.embed_decode(data)
