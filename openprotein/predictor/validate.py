"""Predictor validation results represented as futures."""

import numpy as np

from openprotein.jobs import Future

from . import api
from .schemas import PredictorCVJob

__all__ = ["CVResultFuture"]


class CVResultFuture(Future):
    """Future Job for manipulating results"""

    job: PredictorCVJob

    @property
    def id(self):
        return self.job.job_id

    def _get(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the cross-validation results for this job.

        Parameters
        ----------
        verbose : bool, optional
            Flag to enable verbose output, by default False.

        Returns
        -------
        tuple of np.ndarray
            The observed ``y`` and the cross-validated ``y_mu`` and ``y_var`` of
            each evaluated row.
        """
        data = api.predictor_crossvalidate_get(self.session, self.job.job_id)
        return api.decode_crossvalidate(data)
