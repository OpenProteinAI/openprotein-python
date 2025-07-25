"""Predictor validation results represented as futures."""

import numpy as np

from openprotein.base import APISession
from openprotein.jobs import Future

from . import api
from .schemas import PredictorCVJob


class CVResultFuture(Future):
    """Future Job for manipulating results"""

    job: PredictorCVJob

    def __init__(
        self,
        session: APISession,
        job: PredictorCVJob,
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
        data = api.predictor_crossvalidate_get(self.session, self.job.job_id)
        return api.decode_crossvalidate(data)
