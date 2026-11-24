"""`Future.outputs` lists names only for jobs whose outputs the reads serve."""

from typing import Any

import pytest

from openprotein.errors import APIError
from openprotein.jobs import Future, Job, JobsAPI, SVDFitFuture
from openprotein.models import ModelsAPI
from openprotein.prompt.models import Prompt
from openprotein.prompt.schemas import PromptMetadata
from tests.models.support import SUBMITTED_JOB, ModelsSession

NO_OUTPUTS = "has no stored outputs to read"


def _session_with_sampling_job() -> tuple[ModelsSession, dict[str, Any]]:
    client = ModelsSession()
    session = client.session
    session.jobs = JobsAPI(session)
    job = SUBMITTED_JOB | {"job_type": "/prompt/sample_msa"}
    job_id = job["job_id"]
    client.get_routes[f"v1/jobs/{job_id}"] = job
    client.get_routes[f"v1/prompt/{job_id}"] = {
        "id": job_id,
        "job_id": job_id,
        "name": "p",
        "description": None,
        "created_date": job["created_date"],
        "num_replicates": 1,
        "status": "SUCCESS",
    }
    client.outputs[(job_id, "prompt")] = ["value"]
    return client, job


def test_a_sampling_job_lists_its_output_without_a_model_lookup():
    client, job = _session_with_sampling_job()

    future = Future(client.session, Job.model_validate(job))

    assert future.outputs == ["prompt"]
    assert not any(
        call.args[0].startswith("v1/models/")
        for call in client.session.get.call_args_list
    )


def test_a_reloaded_prompt_lists_its_output_without_a_model_lookup():
    client, job = _session_with_sampling_job()
    metadata = PromptMetadata.model_validate(
        client.get_routes[f"v1/prompt/{job['job_id']}"]
    )

    prompt = Prompt(client.session, metadata=metadata)

    assert prompt.outputs == ["prompt"]
    assert not any(
        call.args[0].startswith("v1/models/")
        for call in client.session.get.call_args_list
    )


def test_a_submitted_projection_without_stored_outputs_refuses_to_list_outputs():
    client = ModelsSession()
    models = ModelsAPI(client.session)
    job = SUBMITTED_JOB | {"job_type": "/svd/embed"}
    client.post_routes["v1/models/svd/s-1/batch/embeddings"] = job

    future = models.get_model("svd/s-1").transform(["AC"])

    with pytest.raises(APIError, match=NO_OUTPUTS):
        _ = future.outputs
    with pytest.raises(APIError, match=NO_OUTPUTS):
        future.get()


def test_a_fit_without_stored_outputs_refuses_to_list_outputs():
    client = ModelsSession()
    job = Job.create(SUBMITTED_JOB | {"job_type": "/svd/fit"})

    with pytest.raises(APIError, match=NO_OUTPUTS):
        _ = SVDFitFuture(client.session, job).outputs


def test_a_fit_with_stored_outputs_lists_them():
    client = ModelsSession()
    job = Job.create(SUBMITTED_JOB | {"job_type": "/models/svd/s-1/fit"})

    assert SVDFitFuture(client.session, job).outputs == ["sequences"]
