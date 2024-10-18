import numpy as np
from openprotein.api import predictor
from openprotein.base import APISession
from openprotein.schemas import (
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictSingleSiteJob,
)

from ..futures import Future


class PredictionResultFuture(Future):
    """Future Job for manipulating results"""

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
            self._sequences = predictor.predictor_predict_get_sequences(
                self.session, self.job.job_id
            )
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def keys(self):
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
        data = predictor.predictor_predict_get_sequence_result(
            self.session, self.job.job_id, sequence
        )
        return predictor.decode_predict(data)

    def get(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            mu (np.ndarray): means of predictions
            var (np.ndarray): variances of predictions
        """
        data = predictor.predictor_predict_get_batched_result(
            self.session, self.job.job_id
        )
        return predictor.decode_predict(data, batched=True)
