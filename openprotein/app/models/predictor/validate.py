import numpy as np
from openprotein.api import predictor
from openprotein.base import APISession
from openprotein.schemas import CVJob

from ..futures import Future


class CVResultFuture(Future):
    """Future Job for manipulating results"""

    job: CVJob

    def __init__(
        self,
        session: APISession,
        job: CVJob,
    ):
        super().__init__(session, job)

    @property
    def id(self):
        return self.job.job_id

    def get(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            mu (np.ndarray): means of predictions
            var (np.ndarray): variances of predictions
        """
        data = predictor.predictor_crossvalidate_get(self.session, self.job.job_id)
        return predictor.decode_crossvalidate(data)
