"""Align results represented as futures."""

from openprotein import _deprecation
from openprotein.base import APISession
from openprotein.jobs import Job

from . import api
from .schemas import AlignType

__all__ = ["AlignFuture"]


class AlignFuture:
    """A future object representing an alignment job, providing methods to retrieve job inputs and seed sequences."""

    session: APISession
    job: Job

    def get_input(self, input_type: AlignType):
        """
        Retrieve input data for this alignment job.

        .. deprecated:: 0.18.0
            For ``AlignType.MSA``, use `MSAFuture.wait()` for the generated MSA, or
            `session.data.MSA(id=...).to_string()`. For ``AlignType.PROMPT``, use
            `Prompt.get()` for the replicates of a sampled prompt, indexed by
            replicate. For ``AlignType.INPUT``, use
            `MSAFuture.get_seed()` for the seed.

        Parameters
        ----------
        input_type : AlignType
            The type of input data to retrieve.

        Returns
        -------
        Iterator[list[str]]
            An iterator over the input data rows.
        """
        _deprecation.warn_deprecated(
            "openprotein.align.future.AlignFuture.get_input",
            api._inputs_replacement(input_type, on_future=True),
            since="0.18.0",
            removed_in="0.19.0",
        )
        return api._get_input(
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
