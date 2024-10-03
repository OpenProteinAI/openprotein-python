"""Future for embeddings-related jobs."""

from collections import namedtuple
from typing import Generator

import numpy as np

from openprotein import config
from openprotein.api import embedding
from openprotein.base import APISession
from openprotein.schemas import (
    AttnJob,
    EmbeddingsJob,
    GenerateJob,
    JobType,
    LogitsJob,
    ScoreJob,
    ScoreSingleSiteJob,
)

from ..futures import Future, MappedFuture, StreamingFuture


class EmbeddingResultFuture(MappedFuture, Future):
    """Future for manipulating results for embeddings-related requests."""

    job: EmbeddingsJob | AttnJob | LogitsJob

    def __init__(
        self,
        session: APISession,
        job: EmbeddingsJob | AttnJob | LogitsJob,
        sequences: list[bytes] | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session=session, job=job, max_workers=max_workers)
        self._sequences = sequences

    def get(self, verbose=False) -> list:
        return super().get(verbose=verbose)

    @property
    def sequences(self) -> list[bytes]:
        if self._sequences is None:
            self._sequences = embedding.get_request_sequences(
                self.session, self.job.job_id
            )
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def keys(self):
        return self.sequences

    def get_item(self, sequence: bytes) -> np.ndarray:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: embeddings
        """
        data = embedding.request_get_sequence_result(
            self.session, self.job.job_id, sequence
        )
        return embedding.result_decode(data)


class EmbeddingsScoreResultFuture(StreamingFuture, Future):
    """Future for manipulating results for embeddings score-related requests."""

    job: ScoreJob | ScoreSingleSiteJob | GenerateJob

    def __init__(
        self,
        session: APISession,
        job: ScoreJob | ScoreSingleSiteJob | GenerateJob,
    ):
        super().__init__(session=session, job=job)

    def stream(self) -> Generator:
        if self.job_type == JobType.poet_generate:
            stream = embedding.request_get_generate_result(
                session=self.session, job_id=self.id
            )
        else:
            stream = embedding.request_get_score_result(
                session=self.session, job_id=self.id
            )
        header = next(stream)
        Score = namedtuple("Score", header)
        for line in stream:
            output = Score(*line)
            yield output
