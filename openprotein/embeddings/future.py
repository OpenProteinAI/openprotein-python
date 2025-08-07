"""Future for embeddings-related jobs."""

from collections import namedtuple
from typing import Generator

import numpy as np

from openprotein import config

"""Embeddings results represented as futures."""

from openprotein.base import APISession
from openprotein.jobs import Future, MappedFuture, StreamingFuture

from . import api
from .schemas import (
    AttnJob,
    EmbeddingsJob,
    GenerateJob,
    JobType,
    LogitsJob,
    ScoreIndelJob,
    ScoreJob,
    ScoreSingleSiteJob,
)


class EmbeddingsResultFuture(MappedFuture, Future):
    """Future for manipulating results for embeddings-related requests."""

    job: EmbeddingsJob | AttnJob | LogitsJob

    def __init__(
        self,
        session: APISession,
        job: EmbeddingsJob | AttnJob | LogitsJob,
        sequences: list[bytes] | list[str] | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        super().__init__(session=session, job=job, max_workers=max_workers)
        # ensure all list[bytes]
        self._sequences = (
            [seq.encode() if isinstance(seq, str) else seq for seq in sequences]
            if sequences is not None
            else sequences
        )

    def stream(self):
        return api.request_get_embeddings_stream(session=self.session, job_id=self.id)

    def get(self, verbose=False) -> list[np.ndarray]:
        return super().get(verbose=verbose)

    @property
    def sequences(self) -> list[bytes] | list[str]:
        if self._sequences is None:
            self._sequences = api.get_request_sequences(
                session=self.session, job_id=self.job.job_id, job_type=self.job.job_type
            )
        return self._sequences

    @property
    def id(self):
        return self.job.job_id

    def __keys__(self):
        """
        Get the list of sequences submitted for the embed request.

        Returns
        -------
        list of bytes
            List of sequences.
        """
        return self.sequences

    def get_item(self, sequence: bytes) -> np.ndarray:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (bytes): sequence to fetch results for

        Returns:
            np.ndarray: embeddings
        """
        data = api.request_get_sequence_result(
            session=self.session,
            job_id=self.job.job_id,
            sequence=sequence,
            job_type=self.job.job_type,
        )
        return api.result_decode(data)


class EmbeddingsScoreFuture(StreamingFuture, Future):
    """Future for manipulating results for embeddings score-related requests."""

    job: ScoreJob | ScoreIndelJob | ScoreSingleSiteJob

    def __init__(
        self,
        session: APISession,
        job: ScoreJob | ScoreSingleSiteJob | GenerateJob,
        sequences: list[bytes] | list[str] | None = None,
    ):
        super().__init__(session=session, job=job)
        self._sequences = sequences

    @property
    def sequences(self) -> list[bytes] | list[str]:
        if self._sequences is None:
            self._sequences = api.get_request_sequences(self.session, self.job.job_id)
        return self._sequences

    def stream(self) -> Generator:
        if self.job_type == JobType.poet_generate:
            stream = api.request_get_generate_result(
                session=self.session, job_id=self.id
            )
        else:
            stream = api.request_get_score_result(session=self.session, job_id=self.id)
        # mut_code, ... for ssp
        # name, sequence, ... for score
        header = next(stream)
        score_start_index = 0
        for i, col_name in enumerate(header):
            if col_name.startswith("score"):
                score_start_index = i
                break
        Score = namedtuple("Score", header[:score_start_index] + ["score"])
        for line in stream:
            # combine scores into numpy array
            scores = np.array([float(s) for s in line[score_start_index:]])
            output = Score(*line[:score_start_index], scores)
            yield output


class EmbeddingsGenerateFuture(EmbeddingsScoreFuture, StreamingFuture, Future):
    """Future for manipulating results for embeddings generate-related requests."""

    job: GenerateJob

    @property
    def sequences(self):
        raise Exception("generate job does not support listing sequences")
