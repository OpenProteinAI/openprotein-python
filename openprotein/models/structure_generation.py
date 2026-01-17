"""Utilities for structure generation models."""

from typing import Iterator

from openprotein.base import APISession
from openprotein.jobs import Job, JobsAPI, MappedFuture
from openprotein.molecules import Complex


class StructureGenerationFuture(MappedFuture[int, Complex]):
    """Future for handling the results of an RFdiffusion job."""

    def __init__(self, session: APISession, job: Job, N: int | None = None, **kwargs):
        super().__init__(session, job, **kwargs)
        num_designs = N
        if num_designs is None:
            jobs_api = getattr(self.session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            num_designs = jobs_api.get_job_args(self.job.job_id).get("n") or 1
        self.n = num_designs

    def __keys__(self) -> list[int]:
        return list(range(self.n))

    def stream(self, **kwargs) -> Iterator[Complex]:
        for _, v in super().stream(**kwargs):
            yield v

    def get(self, **kwargs) -> list[Complex]:
        return [v for v in self.stream(**kwargs)]
