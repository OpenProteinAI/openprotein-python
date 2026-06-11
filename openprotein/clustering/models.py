"""HierarchicalClusteringFuture — the future type returned by ClusteringAPI calls."""

from openprotein.base import APISession
from openprotein.jobs import Future, JobsAPI

from . import api
from .schemas import ClusteringMetadata, HierarchicalClusteringResult, HierarchicalFitJob


class HierarchicalClusteringFuture(Future["HierarchicalClusteringResult"]):
    """Future for a hierarchical clustering job. Waits for the fit job to complete, then
    returns a HierarchicalClusteringResult (linkage + leaf_order + sequences)."""

    job: HierarchicalFitJob

    def __init__(
        self,
        session: APISession,
        job: HierarchicalFitJob | None = None,
        metadata: ClusteringMetadata | None = None,
    ):
        if metadata is None:
            if job is None:
                raise ValueError("Expected clustering metadata or job")
            metadata = api.clustering_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = HierarchicalFitJob.create(
                jobs_api.get_job(job_id=metadata.id)
            )
        self._sequences: list[bytes] | None = None
        super().__init__(session, job)

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self) -> str:
        return self._metadata.id

    @property
    def metadata(self) -> ClusteringMetadata:
        """Metadata — lazily refreshes until the job reaches a terminal state."""
        if not self._metadata.is_done():
            self._metadata = api.clustering_get(self.session, self._metadata.id)
        return self._metadata

    @property
    def sequences(self) -> list[bytes]:
        """Input sequences used to fit the clustering job (cached)."""
        if self._sequences is None:
            self._sequences = api.clustering_get_sequences(
                session=self.session, clustering_id=self.id
            )
        return self._sequences

    def _get(self, verbose: bool = False) -> HierarchicalClusteringResult:
        """Fetch the clustering result. Called by Future.get after SUCCESS."""
        result = api.clustering_get_result(self.session, self.id)
        result.sequences = self.sequences
        return result

    def delete(self) -> bool:
        """Delete this clustering job."""
        return api.clustering_delete(self.session, self.id)

    def redispatch(self) -> "HierarchicalClusteringFuture":
        """Redispatch this clustering job."""
        job = api.clustering_redispatch_post(self.session, self.id)
        return HierarchicalClusteringFuture(session=self.session, job=job)
