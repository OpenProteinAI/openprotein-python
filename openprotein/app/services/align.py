from typing import BinaryIO, Iterator

from openprotein.api import align
from openprotein.app.models import MSAFuture, PromptFuture
from openprotein.base import APISession
from openprotein.schemas import Job, PoetInputType


class AlignAPI:
    """API interface for calling Poet and Align endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def upload_msa(self, msa_file) -> MSAFuture:
        """
        Upload an MSA from file.

        Parameters
        ----------
        msa_file : str, optional
            Ready-made MSA. If not provided, default value is None.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        return MSAFuture.create(
            session=self.session, job=align.msa_post(self.session, msa_file=msa_file)
        )

    def create_msa(self, seed: bytes) -> MSAFuture:
        """
        Construct an MSA via homology search with the seed sequence.

        Parameters
        ----------
        seed : bytes
            Seed sequence for the MSA construction.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAJob
            Job object containing the details of the MSA construction.
        """
        return MSAFuture.create(
            session=self.session, job=align.msa_post(self.session, seed=seed)
        )

    def upload_prompt(self, prompt_file: BinaryIO) -> PromptFuture:
        """
        Directly upload a prompt.

        Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
        You can specify multiple prompts (one per replicate) with an <END_PROMPT> and newline between CSVs.

        Parameters
        ----------
        prompt_file : BinaryIO
            Binary I/O object representing the prompt file.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        PromptJob
            An object representing the status and results of the prompt job.
        """
        return PromptFuture.create(
            session=self.session,
            job=align.upload_prompt_post(session=self.session, prompt_file=prompt_file),
        )

    def get_prompt(
        self, job: Job, prompt_index: int | None = None
    ) -> Iterator[list[str]]:
        """
        Get prompts for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.
        prompt_index : Optional[int]
            The replicate number for the prompt (input_type=-PROMPT only)

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return align.get_input(
            session=self.session,
            job=job,
            input_type=PoetInputType.PROMPT,
            prompt_index=prompt_index,
        )

    def get_seed(self, job: Job) -> Iterator[list[str]]:
        """
        Get input data for a given msa job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return align.get_input(
            session=self.session, job=job, input_type=PoetInputType.INPUT
        )

    def get_msa(self, job: Job) -> Iterator[list[str]]:
        """
        Get generated MSA for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return align.get_input(
            session=self.session, job=job, input_type=PoetInputType.MSA
        )
