"""Select supported models-task result contracts."""

import re
from functools import cache
from typing import TYPE_CHECKING, Any

from openprotein.base import APISession

from . import futures
from .futures import Future, _Submission
from .schemas import Job

if TYPE_CHECKING:
    from openprotein.models.schemas import MethodDescriptor

_TASKS: dict[str, type[Future]] = {
    "batch/embeddings": futures.EmbeddingFuture,
    "batch/logits": futures.LogitsFuture,
    "batch/attn": futures.AttentionFuture,
    "batch/loglikelihood": futures.LogLikelihoodFuture,
    "single-site/scores": futures.SingleSiteScoreFuture,
    "indel/scores": futures.IndelScoreFuture,
    "generate/sequences": futures.SequenceGenerationFuture,
    "generate/structures": futures.StructureDesignFuture,
    "batch/structures": futures.StructurePredictionFuture,
    "batch/predictions": futures.PredictionFuture,
    "single-site/predictions": futures.SingleSitePredictionFuture,
}


@cache
def _fit_futures() -> dict[str, type[Future]]:
    from .futures import GPFitFuture, SVDFitFuture, UMAPFitFuture

    return {cls._family: cls for cls in (GPFitFuture, SVDFitFuture, UMAPFitFuture)}


_METHOD_SUFFIXES = tuple(
    sorted((*_TASKS, "fit", "crossvalidate"), key=len, reverse=True)
)


def _encoded_identity(job: Job) -> tuple[str | None, str | None]:
    if not job.job_type.startswith("/models/"):
        return None, None
    encoded = job.job_type.removeprefix("/models/")
    for path in _METHOD_SUFFIXES:
        suffix = "/" + path
        if encoded.endswith(suffix):
            provider = encoded[: -len(suffix)]
            if provider:
                return provider, path
    return None, None


def identity(job: Job) -> tuple[str | None, str | None]:
    """The job's provider and path, from its job type."""
    return _encoded_identity(job)


def _supported(provider: str | None, path: str | None) -> bool:
    family = provider.partition("/")[0] if provider else None
    return path in _TASKS or (path == "fit" and family in _fit_futures())


def recognizes(job: Job) -> bool:
    """Whether the job type names a supported models task."""
    return _supported(*identity(job))


def create(
    session: APISession,
    job: Job,
    *,
    descriptor: "MethodDescriptor | None" = None,
    submission: _Submission | None = None,
    **kwargs: Any,
) -> Future:
    """Create the future for a models task, or a plain ``Future``.

    The submission, else the job type, selects the contract; an unresolved
    identity is tracked through ``Future``.
    """
    if submission is not None:
        provider: str | None = submission.model_id
        path: str | None = submission.path.strip("/")
        usable = bool(provider and path)
    else:
        provider, path = identity(job)
        usable = bool(provider and path)

    future_type: type[Future] | None = None
    if usable and provider is not None and path is not None:
        if path == "fit":
            future_type = _fit_futures().get(provider.partition("/")[0])
        else:
            future_type = _TASKS.get(path)

    selected = future_type or Future
    future = selected(session, job, **kwargs)
    future._resolved_descriptor = descriptor
    future._submission = submission
    return future


_MODELS_JOB_TYPE = re.compile(r"^/models(/[^/]+/[^/]+(/[^/]+)?)?$")


def is_models_job(job: Job) -> bool:
    """Whether the job uses task-output reads, by its job type."""
    return (
        bool(_MODELS_JOB_TYPE.match(job.job_type)) or job.job_type == "/search/mmseqs2"
    )


_PROMPT_SAMPLING_JOB_TYPE = "/prompt/sample_msa"


def serves_outputs(job: Job) -> bool:
    """Whether the job's outputs are served by the data routes, by its job type."""
    return is_models_job(job) or recognizes(job) or is_prompt_sampling(job)


def is_prompt_sampling(job: Job) -> bool:
    """Whether the job is a prompt sampling, by its job type."""
    return job.job_type == _PROMPT_SAMPLING_JOB_TYPE
