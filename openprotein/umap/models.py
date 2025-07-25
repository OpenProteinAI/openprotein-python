"""UMAP models on the OpenProtein system which can be used directly to create projected embeddings useful for visualization."""

from openprotein.base import APISession
from openprotein.embeddings import EmbeddingModel, EmbeddingsResultFuture
from openprotein.jobs import Future, JobsAPI

from . import api
from .schemas import UMAPEmbeddingsJob, UMAPFitJob, UMAPMetadata


class UMAPModel(Future):
    """
    Class providing embedding endpoint for UMAP models. \
        Also allows retrieving embeddings of sequences used to fit the UMAP with `get`.
    Implements a Future to allow waiting for a fit job.
    """

    job: UMAPFitJob

    def __init__(
        self,
        session: APISession,
        job: UMAPFitJob | None = None,
        metadata: UMAPMetadata | None = None,
    ):
        """Initializes with either job get or umap metadata get."""
        if metadata is None:
            # use job to fetch metadata
            if job is None:
                raise ValueError("Expected umap metadata or job")
            metadata = api.umap_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = UMAPFitJob.create(jobs_api.get_job(job_id=metadata.id))
        self._sequences = None
        self._embeddings = None
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
    def n_neighbors(self):
        return self._metadata.n_neighbors

    @property
    def min_dist(self):
        return self._metadata.min_dist

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

    @property
    def sequences(self):
        if self._sequences is not None:
            return self._sequences
        self._sequences = self.get_inputs()
        return self._sequences

    @property
    def embeddings(self):
        if self._embeddings is not None:
            return self._embeddings
        data = api.embed_get_batch_result(session=self.session, job_id=self.id)
        embeddings = [
            (seq, umap)
            for seq, umap in zip(self.sequences, api.embed_batch_decode(data))
        ]
        self._embeddings = embeddings
        return self._embeddings

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = api.umap_get(self.session, self._metadata.id)

    def get_model(self) -> EmbeddingModel:
        """Fetch embeddings model"""
        model = EmbeddingModel.create(session=self.session, model_id=self._metadata.id)
        return model

    @property
    def model(self) -> EmbeddingModel:
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this UMAP model.
        """
        return api.umap_delete(self.session, self.id)

    def get(self, verbose: bool = False):
        return self.embeddings

    def get_inputs(self) -> list[bytes]:
        """
        Get sequences used for umap job.

        Returns
        -------
            List[bytes]: list of sequences
        """
        return api.umap_get_sequences(session=self.session, umap_id=self.id)

    def embed(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> EmbeddingsResultFuture:
        """
        Use this UMAP model to get reduced embeddings from input sequences.

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
            job=api.umap_embed_post(
                session=self.session, umap_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )


class UMAPEmbeddingResultFuture(EmbeddingsResultFuture, Future):
    """Future for manipulating results for embeddings-related requests."""

    job: UMAPEmbeddingsJob
