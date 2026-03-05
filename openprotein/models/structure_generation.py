"""Utilities for structure generation models."""

from typing import Iterator, Literal

from openprotein.base import APISession
from openprotein.jobs import Job, JobsAPI, MappedFuture
from openprotein.molecules import Complex


class StructureGenerationJob(Job):
    """Job schema for an RFdiffusion request."""

    job_type: Literal["/models/design"]


class StructureGenerationFuture(MappedFuture[int, Complex]):
    """Future for handling structure-generation model results."""

    job: StructureGenerationJob

    def __init__(
        self,
        session: APISession,
        job: Job,
        N: int | None = None,
        result_format: Literal["pdb", "cif"] = "pdb",
        **kwargs,
    ):
        super().__init__(session, job, **kwargs)
        num_designs = N
        if num_designs is None:
            jobs_api = getattr(self.session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            num_designs = jobs_api.get_job_args(self.job.job_id).get("n") or 1
        self.n = num_designs
        self.result_format = result_format

    def __keys__(self) -> list[int]:
        return list(range(self.n))

    def get_item(self, replicate: int = 0) -> Complex:
        response = self.session.get(
            f"v1/design/{self.id}/results", params={"replicate": replicate}
        )
        return Complex.from_string(response.text, format=self.result_format)

    def stream(self, **kwargs) -> Iterator[Complex]:
        for _, v in super().stream(**kwargs):
            yield v

    def get(self, **kwargs) -> list[Complex]:
        return [v for v in self.stream(**kwargs)]
