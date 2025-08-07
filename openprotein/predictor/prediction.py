"""Prediction results represented as futures."""

import numpy as np

from openprotein.base import APISession
from openprotein.jobs import Future

from . import api
from .schemas import (
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictSingleSiteJob,
)


class PredictionResultFuture(Future):
    """Prediction results represented as a future."""

    job: PredictJob | PredictSingleSiteJob | PredictMultiJob | PredictMultiSingleSiteJob

    def __init__(
        self,
        session: APISession,
        job: (
            PredictJob
            | PredictSingleSiteJob
            | PredictMultiJob
            | PredictMultiSingleSiteJob
        ),
        sequences: list[bytes] | None = None,
    ):
        super().__init__(session, job)
        self._sequences = sequences

    @property
    def sequences(self):
        if self._sequences is None:
            self._sequences = api.predictor_predict_get_sequences(
                self.session, self.job.job_id
            )
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def __keys__(self):
        return self.sequences

    def get_item(self, sequence: bytes) -> tuple[np.ndarray, np.ndarray]:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            mu (np.ndarray): means of sequence prediction
            var (np.ndarray): variances of sequence prediction
        """
        data = api.predictor_predict_get_sequence_result(
            self.session, self.job.job_id, sequence
        )
        return api.decode_predict(data)

    def get(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            mu (np.ndarray): means of predictions
            var (np.ndarray): variances of predictions
        """
        data = api.predictor_predict_get_batched_result(self.session, self.job.job_id)
        return api.decode_predict(data, batched=True)
