from typing import Iterator

from openprotein import config
from openprotein.api import align
from openprotein.base import APISession
from openprotein.schemas import JobType, MSAJob, MSASamplingMethod

from ..futures import Future, InvalidFutureError
from .base import AlignFuture
from .prompt import PromptFuture


class MSAFuture(AlignFuture, Future):
    """
    Represents a result of a MSA job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.
    page_size : int
        The number of results to fetch in a single page.

    Methods
    -------
    get(verbose=False)
        Get the final results of the PoET scoring job.

    Returns
    -------
    List[PoetScoreResult]
        The list of results from the PoET scoring job.
    """

    job: MSAJob

    def __init__(
        self, session: APISession, job: MSAJob, page_size: int = config.POET_PAGE_SIZE
    ):
        """
        init a PoetScoreFuture instance.

        Parameters
        ----------
        session : APISession
            An instance of APISession for API interactions.
        job : Job
            The PoET scoring job.
        page_size : int
            The number of results to fetch in a single page.

        """
        super().__init__(session, job)
        self.page_size = page_size
        self.msa_id = self.job.job_id

    # def wait(self, verbose: bool = False):
    #     _ = self.job.wait(
    #         self.session,
    #         interval=config.POLLING_INTERVAL,
    #         timeout=config.POLLING_TIMEOUT,
    #         verbose=verbose,
    #     )  # no progress to track
    #     return self.get()

    def get(self, verbose: bool = False) -> Iterator[list[str]]:
        return align.get_msa(self.session, self.job)

    def sample_prompt(
        self,
        num_sequences: int | None = None,
        num_residues: int | None = None,
        method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
        homology_level: float = 0.8,
        max_similarity: float = 1.0,
        min_similarity: float = 0.0,
        always_include_seed_sequence: bool = False,
        num_ensemble_prompts: int = 1,
        random_seed: int | None = None,
    ) -> PromptFuture:
        """
        Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.

        Parameters
        ----------
        num_sequences : int, optional
            Maximum number of sequences in the prompt. Must be  <100.
        num_residues : int, optional
            Maximum number of residues (tokens) in the prompt. Must be less than 24577.
        method : MSASamplingMethod, optional
            Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
        homology_level : float, optional
            Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
        max_similarity : float, optional
            Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
        min_similarity : float, optional
            Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
        always_include_seed_sequence : bool, optional
            Whether to always include the seed sequence in the MSA. Defaults to False.
        num_ensemble_prompts : int, optional
            Number of ensemble jobs to run. Defaults to 1.
        random_seed : int, optional
            Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

        Raises
        ------
        InvalidParameterError
            If provided parameter values are not in the allowed range.
        MissingParameterError
            If both or none of 'num_sequences', 'num_residues' is specified.

        Returns
        -------
        PromptJob
        """
        msa_id = self.msa_id
        job = align.prompt_post(
            self.session,
            msa_id=msa_id,
            num_sequences=num_sequences,
            num_residues=num_residues,
            method=method,
            homology_level=homology_level,
            max_similarity=max_similarity,
            min_similarity=min_similarity,
            always_include_seed_sequence=always_include_seed_sequence,
            num_ensemble_prompts=num_ensemble_prompts,
            random_seed=random_seed,
        )
        future = PromptFuture.create(session=self.session, job=job)
        return future
