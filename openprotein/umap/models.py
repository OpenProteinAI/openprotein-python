"""UMAP models on the OpenProtein system which can be used directly to create projected embeddings useful for visualization."""

import numpy as np

from openprotein import config
from openprotein.base import APISession
from openprotein.embeddings import EmbeddingModel, EmbeddingsResultFuture
from openprotein.jobs import Future, JobsAPI

from . import api
from .schemas import UMAPEmbeddingsJob, UMAPFitJob, UMAPMetadata


class UMAPModel(Future):
    """
    UMAP model that can be used to create projected embeddings.

    The model is also implemented as a `Future` to allow waiting for a fit job.
    The projected embeddings of the sequences used to fit the UMAP can be
    accessed using `embeddings`.
    """

    job: UMAPFitJob

    def __init__(
        self,
        session: APISession,
        job: UMAPFitJob | None = None,
        metadata: UMAPMetadata | None = None,
    ):
        # Initializes with either job get or umap metadata get.
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
        """UMAP unique identifier."""

        return self._metadata.id

    @property
    def n_components(self):
        """Number of components specified for the UMAP."""

        return self._metadata.n_components

    @property
    def n_neighbors(self):
        """Number of neighbors specified for the UMAP."""

        return self._metadata.n_neighbors

    @property
    def min_dist(self):
        """Minimum distance specified for the UMAP."""

        return self._metadata.min_dist

    @property
    def sequence_length(self):
        """Sequence length constraint of the UMAP."""

        return self._metadata.sequence_length

    @property
    def reduction(self):
        """Reduction used to fit the UMAP."""

        return self._metadata.reduction

    @property
    def metadata(self):
        """Metadata of the UMAP."""

        self._refresh_metadata()
        return self._metadata

    @property
    def sequences(self):
        """The sequences used to fit the UMAP."""

        if self._sequences is not None:
            return self._sequences
        self._sequences = self.get_inputs()
        return self._sequences

    @property
    def embeddings(self):
        """The projected embeddings of the sequences used to fit the UMAP."""

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
        model = EmbeddingModel.create(session=self.session, model_id=self._metadata.id)
        return model

    @property
    def model(self) -> EmbeddingModel:
        """Base embeddings model used for the UMAP."""
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this UMAP model.
        """
        return api.umap_delete(self.session, self.id)

    def get(self, verbose: bool = False):
        """Retrieve this UMAP model itself."""
        return self

    def get_inputs(self) -> list[bytes]:
        """
        Get sequences used for umap job.

        Returns
        -------
        list[bytes]
            list of sequences
        """
        return api.umap_get_sequences(session=self.session, umap_id=self.id)

    def embed(
        self, sequences: list[bytes] | list[str], **kwargs
    ) -> "UMAPEmbeddingsResultFuture":
        """
        Use this UMAP model to get projected embeddings from input sequences.

        Parameters
        ----------
        sequences : List[bytes]
            List of protein sequences.

        Returns
        -------
        UMAPEmbeddingsResultFuture
            Future result containing the projected embeddings.
        """
        return UMAPEmbeddingsResultFuture.create(
            session=self.session,
            job=api.umap_embed_post(
                session=self.session, umap_id=self.id, sequences=sequences, **kwargs
            ),
            sequences=sequences,
        )


class UMAPEmbeddingsResultFuture(EmbeddingsResultFuture, Future):
    """UMAP embeddings results represented as a future."""

    job: UMAPEmbeddingsJob

    def wait(
        self,
        interval: int = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ) -> list[np.ndarray]:
        """Wait for the UMAP embeddings job and retrieve the embeddings."""
        return super().wait(interval, timeout, verbose)

    def get(self, verbose=False) -> list[np.ndarray]:
        """Get all the UMAP projected embeddings from the job."""
        return super().get(verbose)

    def get_item(self, sequence: bytes) -> np.ndarray:
        """
        Get UMAP embeddings for specified sequence.

        Parameters
        ----------
        sequence: bytes
            Sequence to fetch UMAP embeddings for.

        Returns
        -------
        np.ndarray
            UMAP embeddings represented a numpy array.
        """
        data = api.embed_get_sequence_result(self.session, self.job.job_id, sequence)
        return api.embed_decode(data)
