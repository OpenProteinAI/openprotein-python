"""Align results represented as futures."""

from openprotein.base import APISession
from openprotein.jobs import Job

from . import api
from .schemas import AlignType


class AlignFuture:
    """A future object representing an alignment job, providing methods to retrieve job inputs and seed sequences."""

    session: APISession
    job: Job

    def get_input(self, input_type: AlignType):
        """
        Retrieve input data for this alignment job.

        Parameters
        ----------
        input_type : AlignType
            The type of input data to retrieve.

        Returns
        -------
        Iterator[list[str]]
            An iterator over the input data rows.
        """
        return api.get_input(
            session=self.session, job_id=self.job.job_id, input_type=input_type
        )

    def get_seed(self):
        """
        Retrieve the seed sequence for this alignment job.

        Returns
        -------
        str
            The seed sequence.
        """
        return api.get_seed(session=self.session, job_id=self.job.job_id)

    @property
    def id(self):
        """
        The job ID for this alignment job.

        Returns
        -------
        str
            The job ID.
        """
        return self.job.job_id
