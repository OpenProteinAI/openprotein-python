from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from openprotein.data.prompts import Prompt as StoredPrompt
from openprotein.errors import HTTPError
from openprotein.jobs import Future, Job, JobsAPI, JobStatus, _dispatch
from openprotein.jobs import api as job_api
from openprotein.jobs.futures import (
    CrossvalidateFuture,
    CrossvalidateResults,
    GPFitFuture,
)
from openprotein.models import gp as models_gp
from openprotein.models.gp import GPModel
from openprotein.molecules import Protein
from openprotein.predictor import api as predictor_api
from openprotein.predictor.models import PredictorModel
from openprotein.predictor.validate import CVResultFuture
from openprotein.prompt import api as prompt_api
from openprotein.prompt.models import Prompt
from openprotein.prompt.schemas import PromptMetadata
from tests.models.test_task_dispatch import CASES, task_job


@pytest.mark.parametrize(("path", "expected"), CASES)
def test_jobs_reconstruct_identified_typed_futures(
    mock_session, monkeypatch, path, expected
):
    job = task_job(path)
    monkeypatch.setattr(job_api, "job_get", lambda session, job_id: job)
    future = JobsAPI(mock_session).get(job.job_id)
    assert type(future) is expected


def test_gp_training_keeps_compatibility_reconstruction(mock_session, monkeypatch):
    payload = {
        "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
        "job_type": "/predictor/train",
        "status": "SUCCESS",
        "created_date": "2026-10-21T08:00:00Z",
    }
    job = Job.create(payload)
    curves = [{"measurement_name": "fitness", "loss": [1.0]}]
    metadata = SimpleNamespace(traingraphs=curves, is_done=lambda: True)
    fitted = GPModel(mock_session, f"gp/{job.job_id}")
    monkeypatch.setattr(
        predictor_api, "predictor_get", lambda session, model_id: metadata
    )
    monkeypatch.setattr(models_gp, "_load", lambda session, model_id: fitted)

    submitted = GPFitFuture(mock_session, job)
    reloaded = Future.create(mock_session, job=job)

    assert submitted.get() is fitted
    assert type(reloaded) is PredictorModel
    assert reloaded.get() == curves


def test_crossvalidation_keeps_compatibility_reconstruction(mock_session, monkeypatch):
    payload = {
        "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
        "job_type": "/predictor/crossvalidate",
        "status": "SUCCESS",
        "created_date": "2026-10-21T08:00:00Z",
    }
    job = Job.create(payload)
    evaluation = {
        "job_id": job.job_id,
        "rows": [
            {
                "row_index": 0,
                "sequence": "MKT",
                "property": "activity",
                "y": 1.0,
                "mean": 0.9,
                "variance": 0.1,
                "fold": None,
            }
        ],
    }
    scores = (np.array([1.0]), np.array([0.9]), np.array([0.1]))
    monkeypatch.setattr(
        GPModel, "evaluations", lambda self, evaluation_id=None: evaluation
    )
    monkeypatch.setattr(
        predictor_api,
        "predictor_crossvalidate_get",
        lambda session, job_id: b"result",
    )
    monkeypatch.setattr(predictor_api, "decode_crossvalidate", lambda data: scores)

    submitted = CrossvalidateFuture(mock_session, job, GPModel(mock_session, "gp/p-1"))
    reloaded = Future.create(mock_session, job=job)

    results = submitted.get()
    assert isinstance(results, CrossvalidateResults)
    assert (results.row_index.tolist(), results.measurement_name.tolist()) == (
        [0],
        ["activity"],
    )
    assert results.fold is None
    assert type(reloaded) is CVResultFuture
    assert reloaded.get() is scores


def test_an_align_prompt_job_keeps_compatibility_reconstruction(
    mock_session, monkeypatch
):
    payload = {
        "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
        "job_type": "/align/prompt",
        "status": "SUCCESS",
        "created_date": "2026-10-21T08:00:00Z",
    }
    job = Job.create(payload)
    contexts = [[Protein(sequence="MKT")]]
    metadata = PromptMetadata(
        id=job.job_id,
        job_id=job.job_id,
        name="sample",
        description=None,
        created_date=job.created_date,
        num_replicates=1,
        status=job.status,
    )
    mock_session.jobs = JobsAPI(mock_session)
    monkeypatch.setattr(job_api, "job_get", lambda session, job_id: job)
    monkeypatch.setattr(
        prompt_api, "get_prompt_metadata", lambda session, prompt_id: metadata
    )
    monkeypatch.setattr(prompt_api, "get_prompt", lambda session, prompt_id: contexts)

    reloaded = Future.create(mock_session, job=job)
    assert type(reloaded) is Prompt
    assert reloaded.get() == contexts


SAMPLING_JOB = {
    "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
    "job_type": "/prompt/sample_msa",
    "status": "RUNNING",
    "created_date": "2026-10-21T08:00:00Z",
}


def test_a_sampling_job_reloads_as_the_stored_prompt_it_samples(mock_session):
    stored = {
        "id": SAMPLING_JOB["job_id"],
        "status": "SUCCESS",
        "job_id": SAMPLING_JOB["job_id"],
        "name": "sampled",
    }
    mock_session.get.return_value.json.return_value = stored

    reloaded = Future.create(mock_session, job=Job.create(SAMPLING_JOB))

    assert isinstance(reloaded, StoredPrompt)
    assert (reloaded.id, reloaded.name) == (SAMPLING_JOB["job_id"], "sampled")
    assert reloaded.status is JobStatus.SUCCESS
    mock_session.get.assert_called_once_with(
        f"v1/data/prompts/{SAMPLING_JOB['job_id']}"
    )


def test_a_sampling_job_whose_prompt_is_not_stored_yet_reports_the_job(
    mock_session,
):
    response = MagicMock(status_code=404)
    mock_session.get.side_effect = HTTPError(response)

    reloaded = Future.create(mock_session, job=Job.create(SAMPLING_JOB))

    assert isinstance(reloaded, StoredPrompt)
    assert (reloaded.id, reloaded.job_id) == (SAMPLING_JOB["job_id"],) * 2
    assert reloaded.status is JobStatus.RUNNING
    mock_session.get.assert_called_once_with(
        f"v1/data/prompts/{SAMPLING_JOB['job_id']}"
    )


def test_fitted_encoded_identity_preserves_the_instance_id(mock_session):
    job = Job.create(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": "/models/gp/p-1/crossvalidate",
            "status": "PENDING",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )

    assert _dispatch.identity(job) == ("gp/p-1", "crossvalidate")
    assert type(Future.create(mock_session, job=job)) is Future
    mock_session.get.assert_not_called()


def test_models_records_without_identity_remain_trackable(mock_session):
    job = Job.create(
        {
            "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
            "job_type": "/models",
            "status": "PENDING",
            "created_date": "2026-10-21T08:00:00Z",
        }
    )
    future = Future.create(mock_session, job=job)
    assert type(future) is Future
    assert future.id == job.job_id
    mock_session.get.assert_not_called()
