"""Future for embeddings-related jobs."""

from collections import namedtuple
from typing import Any, Generator, Iterator, TypeVar, Union

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


class EmbeddingsResultFuture(MappedFuture[bytes, np.ndarray]):
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

    def stream_sync(self):
        """
        Stream the embeddings results synchronously using the streaming endpoint.
        """
        for i, array in enumerate(
            api.request_get_embeddings_stream(session=self.session, job_id=self.id)
        ):
            yield self.sequences[i], array

    @property
    def sequences(self):
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

    def get_item(self, sequence: str | bytes) -> np.ndarray:
        """
        Get embedding results for specified sequence.

        Args:
            sequence (str | bytes): sequence to fetch results for

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


Score = namedtuple("Score", ["name", "sequence", "score", "query_id"])
Score.__new__.__defaults__ = (None,)
SingleSiteScore = namedtuple("SingleSiteScore", ["mut_code", "score", "query_id"])
SingleSiteScore.__new__.__defaults__ = (None,)
S = TypeVar("S", bound=Union[Score, SingleSiteScore])


class BaseScoreFuture(StreamingFuture[S]):
    """Future for manipulating results for embeddings score-related requests."""

    def __init__(
        self,
        session: APISession,
        job: ScoreJob | ScoreSingleSiteJob | GenerateJob,
        sequences: list[bytes] | list[str] | None = None,
    ):
        super().__init__(session=session, job=job)
        # ensure all list[bytes]
        self._sequences = (
            [seq.encode() if isinstance(seq, str) else seq for seq in sequences]
            if sequences is not None
            else sequences
        )

    @property
    def sequences(self) -> list[bytes]:
        if self._sequences is None:
            self._sequences = api.get_request_sequences(self.session, self.job.job_id)
        return self._sequences


class EmbeddingsScoreFuture(BaseScoreFuture[Score]):
    """Future for manipulating results for embeddings score-related requests."""

    job: ScoreJob | ScoreIndelJob

    def stream(self) -> Iterator[Score]:
        stream = api.request_get_score_result(session=self.session, job_id=self.id)
        header = next(stream)
        has_query_id = len(header) > 0 and header[0].strip().lower() == "query_id"
        for line in stream:
            if has_query_id:
                query_id = line[0] if line[0] else None
                name, sequence = line[1], line[2]
                scores = np.array([float(s) for s in line[3:]])
            else:
                query_id = None
                name, sequence = line[0], line[1]
                scores = np.array([float(s) for s in line[2:]])
            yield Score(name=name, sequence=sequence, score=scores, query_id=query_id)


class EmbeddingsScoreSingleSiteFuture(BaseScoreFuture[SingleSiteScore]):
    """Future for manipulating results for embeddings score-related requests."""

    job: ScoreSingleSiteJob

    def stream(self) -> Iterator[SingleSiteScore]:
        stream = api.request_get_score_result(session=self.session, job_id=self.id)
        header = next(stream)
        has_query_id = len(header) > 0 and header[0].strip().lower() == "query_id"
        for line in stream:
            if has_query_id:
                query_id = line[0] if line[0] else None
                mut_code = line[1]
                scores = np.array([float(s) for s in line[2:]])
            else:
                query_id = None
                mut_code = line[0]
                scores = np.array([float(s) for s in line[1:]])
            yield SingleSiteScore(mut_code=mut_code, score=scores, query_id=query_id)


class EmbeddingsGenerateFuture(BaseScoreFuture[Score]):
    """Future for manipulating results for embeddings generate-related requests."""

    job: GenerateJob

    def stream(self) -> Iterator[Score]:
        stream = api.request_get_generate_result(session=self.session, job_id=self.id)
        header = next(stream)
        has_query_id = len(header) > 0 and header[0].strip().lower() == "query_id"
        for line in stream:
            if has_query_id:
                query_id = line[0] if line[0] else None
                name, sequence = line[1], line[2]
                scores = np.array([float(s) for s in line[3:]])
            else:
                query_id = None
                name, sequence = line[0], line[1]
                scores = np.array([float(s) for s in line[2:]])
            yield Score(name=name, sequence=sequence, score=scores, query_id=query_id)

    @property
    def sequences(self):
        raise Exception("generate job does not support listing sequences")
