"""Utilities for structure generation models."""

from typing import Iterator, Literal

from openprotein import _deprecation
from openprotein.base import APISession
from openprotein.jobs import Job, JobsAPI, MappedFuture
from openprotein.molecules import Complex

__all__ = [
    "BoltzGenFuture",
    "RFdiffusionFuture",
    "StructureGenerationFuture",
    "StructureGenerationJob",
]


class StructureGenerationJob(Job):
    """A compatibility structure-generation job.

    Attributes
    ----------
    job_type : str
        The ``/models/design`` job type read by ``StructureGenerationFuture``.
    """

    job_type: Literal["/models/design"]


class StructureGenerationFuture(MappedFuture[int, Complex]):
    """
    Reader for compatibility design jobs with job type ``/models/design``.

    Results are ``Complex`` objects indexed by replicate. Generation methods on
    ``session.models`` return ``StructureDesignFuture``.
    """

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

    def _get(self, verbose: bool = False, **kwargs) -> list[Complex]:
        return [v for v in self.stream(**kwargs)]


class RFdiffusionFuture(StructureGenerationFuture):
    """
    Alias for :class:`StructureGenerationFuture`.

    .. deprecated:: 0.18.0
        Submit designs with ``session.models.rfdiffusion.generate(...)``,
        which returns :class:`openprotein.jobs.StructureDesignFuture`.
        This alias reads compatibility design jobs.
    """

    def __init__(self, *args, **kwargs):
        _deprecation.warn_deprecated(
            "openprotein.models.RFdiffusionFuture",
            "`session.models.rfdiffusion.generate(...)` "
            "(returns `openprotein.jobs.StructureDesignFuture`)",
            since="0.18.0",
            removed_in="0.19.0",
        )
        super().__init__(*args, **kwargs)


class BoltzGenFuture(StructureGenerationFuture):
    """
    Alias for :class:`StructureGenerationFuture`.

    .. deprecated:: 0.18.0
        Submit designs with ``session.models.boltzgen.generate(...)``,
        which returns :class:`openprotein.jobs.StructureDesignFuture`.
        This alias reads compatibility design jobs.
    """

    def __init__(self, *args, **kwargs):
        _deprecation.warn_deprecated(
            "openprotein.models.BoltzGenFuture",
            "`session.models.boltzgen.generate(...)` "
            "(returns `openprotein.jobs.StructureDesignFuture`)",
            since="0.18.0",
            removed_in="0.19.0",
        )
        super().__init__(*args, **kwargs)
