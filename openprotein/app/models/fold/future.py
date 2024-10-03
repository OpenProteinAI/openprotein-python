from openprotein import config
from openprotein.api import fold
from openprotein.base import APISession
from openprotein.schemas import FoldJob

from ..futures import Future, MappedFuture


class FoldResultFuture(MappedFuture, Future):
    """Future Job for manipulating results"""

    job: FoldJob

    def __init__(
        self,
        session: APISession,
        job: FoldJob,
        sequences: list[bytes] | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session, job, max_workers)
        if sequences is None:
            sequences = fold.fold_get_sequences(self.session, job_id=job.job_id)
        self._sequences = sequences

    @property
    def sequences(self) -> list[bytes]:
        if self._sequences is None:
            self._sequences = fold.fold_get_sequences(self.session, self.job.job_id)
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def keys(self):
        return self.sequences

    def get(self, verbose=False) -> list[tuple[str, str]]:
        return super().get(verbose=verbose)

    def get_item(self, sequence: bytes) -> bytes:
        """
        Get fold results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: fold
        """
        data = fold.fold_get_sequence_result(self.session, self.job.job_id, sequence)
        return data
