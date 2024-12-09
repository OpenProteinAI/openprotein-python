from typing import Collection, Iterator

import numpy as np
from openprotein import config
from openprotein.api import align
from openprotein.api.deprecated import poet
from openprotein.base import APISession
from openprotein.csv import csv_stream
from openprotein.errors import APIError
from openprotein.schemas import AlignType
from openprotein.schemas.deprecated.poet import (
    PoetGenerateJob,
    PoetScoreJob,
    PoetScoreResult,
    PoetSSPJob,
    PoetSSPResult,
)

from ..futures import Future, PagedFuture, StreamingFuture


class AlignPrompted(Future):
    """Requests conditioned on an MSA prompt."""

    def get_input(self, input_type: AlignType):
        """Get prompt's input - can be INPUT (seed), GENERATED (aligned msa) or PROMPT."""
        return align.get_input(self.session, self.job, input_type)

    def get_prompt(self, prompt_index: int | None = None):
        """Get sampled prompt."""
        return align.get_prompt(self.session, self.job, prompt_index=prompt_index)

    def get_seed(self):
        """Get seed sequence to MSA."""
        return align.get_seed(self.session, self.job)

    def get_msa(self):
        """Get aligned MSA."""
        return align.get_msa(self.session, self.job)


class PoetScoreFuture(PagedFuture, AlignPrompted, Future):
    """
    Represents a result of a PoET scoring job.

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
        Get the final results of the PoET  job.

    """

    job: PoetScoreJob

    def __init__(
        self,
        session: APISession,
        job: PoetScoreJob,
        page_size=config.POET_PAGE_SIZE,
    ):
        """
        init a PoetScoreFuture instance.

        Parameters
        ----------
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET scoring job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

    def _fmt_results(self, results: list[PoetScoreResult]):
        # Format results after getting is complete
        return [(p.name, p.sequence, np.asarray(p.score)) for p in results]

    def get_slice(self, start: int, end: int) -> Collection:
        results = poet.poet_score_get(
            self.session,
            self.id,
            page_offset=start,
            page_size=end - start,
        ).result
        if results is not None:
            return self._fmt_results(results)
        return []


class PoetSingleSiteFuture(PagedFuture, AlignPrompted, Future):
    """
    Represents a result of a PoET single-site analysis job.

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
        Get the final results of the PoET  job.

    """

    job: PoetSSPJob

    def __init__(
        self,
        session: APISession,
        job: PoetSSPJob,
        page_size=config.POET_PAGE_SIZE,
    ):
        """
        init a PoetSingleSiteFuture instance.

        Parameters
        ----------
            session (APISession): An instance of APISession for API interactions.
            job (Job): The PoET single-site analysis job.
            page_size (int, optional): The number of results to fetch in a single page. Defaults to config.POET_PAGE_SIZE.

        """
        super().__init__(session, job)
        self.page_size = page_size

    def _fmt_results(self, results: list[PoetSSPResult]):
        # Format results after getting is complete
        return {p.sequence: np.asarray(p.score) for p in results}

    def get_slice(self, start: int, end: int) -> Collection:
        results = poet.poet_single_site_get(
            self.session,
            self.id,
            page_offset=start,
            page_size=end - start,
        ).result
        if results is not None:
            return self._fmt_results(results)
        return []


class PoetGenerateFuture(StreamingFuture, AlignPrompted, Future):
    """
    Represents a result of a PoET generation job.

    Attributes
    ----------
    session : APISession
        An instance of APISession for API interactions.
    job : Job
        The PoET scoring job.

    Methods:
        stream() -> Iterator[PoetScoreResult]:
            Stream the results of the PoET generation job.

    """

    job: PoetGenerateJob

    def stream(self) -> Iterator[PoetScoreResult]:
        """
        Stream the results from the response.

        Returns
        ------
        PoetScoreResult: Yield
            A result object containing the sequence, score, and name.

        Raises
        ------
        APIError
            If the request fails.
        """
        try:
            response = poet.poet_generate_get(self.session, self.job.job_id)
            for tokens in csv_stream(response):
                try:
                    name, sequence = tokens[:2]
                    score = [float(s) for s in tokens[2:]]
                    sequence = sequence.encode()
                    sample = PoetScoreResult(sequence=sequence, score=score, name=name)
                    yield sample
                except (IndexError, ValueError) as exc:
                    # Skip malformed or incomplete tokens
                    print(
                        f"Skipping malformed or incomplete tokens: {tokens} with {exc}"
                    )
        except APIError as exc:
            print(f"Failed to stream PoET generation results: {exc}")
