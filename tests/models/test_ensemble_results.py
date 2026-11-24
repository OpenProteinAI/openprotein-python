import numpy as np
import pytest

from openprotein.fold import api as fold_api
from openprotein.fold.future import FoldResultFuture
from openprotein.fold.schemas import FoldMetadata
from openprotein.jobs import Future, Job, JobsAPI, _dispatch
from openprotein.jobs import api as job_api
from openprotein.jobs.futures import StructurePredictionFuture
from openprotein.jobs.schemas import JobType
from openprotein.models import (
    AlphaFlowInput,
    BioEmuInput,
    ESMFlowInput,
    ModelsAPI,
)
from openprotein.molecules import Complex, Protein, Structure
from tests.models.support import SUBMITTED_JOB, Stored

IDS = (
    "bioemu-v1-1",
    "alphaflow-md-base-v1",
    "alphaflow-md-distilled-v1",
    "esmflow-md-base-v1",
    "esmflow-md-distilled-v1",
)
MSA_ID = "3f1c2a7e-9b4d-4e2a-8c1f-0d5e6a7b8c9d"
PROMPT_ID = "5b2d8e4a-1c7f-4a3e-9d6b-2e8f0a1c3b5d"


#: x-offsets of each position's conformations, distinct so order is checked
SHIFTS = [[0.0, 10.0], [100.0, 110.0]]


def _ensemble(reference: Protein, shifts: list[float]) -> str:
    conformations = [
        Complex({"A": reference.copy()}).transform(t=np.array([shift, 0.0, 0.0]))
        for shift in shifts
    ]
    return Structure(conformations).to_string(format="cif")


def _shifts(value: object, reference: Protein) -> list[float]:
    assert isinstance(value, Structure)
    origin = np.nanmean(reference.coordinates[..., 0])
    return [
        round(
            float(
                np.nanmean(conformation.get_proteins()["A"].coordinates[..., 0])
                - origin
            ),
            1,
        )
        for conformation in value
    ]


def _submit(models, model_id, reference):
    handle = models.get_model(model_id)
    sequence = reference.sequence
    if model_id == "bioemu-v1-1":
        entry = BioEmuInput(sequence, msa=MSA_ID)
    elif model_id.startswith("alphaflow-"):
        entry = AlphaFlowInput(sequence, msa=MSA_ID)
    else:
        entry = ESMFlowInput(sequence)
    return handle.sample([entry, entry], num_samples=2, random_seed=7)


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize(
    ("job_type", "operation"),
    [
        ("/models", "submitted"),
        ("/models", "get"),
        ("/models", "explicit"),
    ],
)
def test_ensemble_results_preserve_positions_and_conformations(
    aligned_ensemble_session,
    ensemble_reference,
    monkeypatch,
    model_id,
    operation,
    job_type,
):
    monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)
    models_session = aligned_ensemble_session
    is_flow = model_id.startswith(("alphaflow-", "esmflow-"))
    outputs = ["structures", "plddt", "confidence"] if is_flow else ["structures"]
    payload = {
        **SUBMITTED_JOB,
        "job_type": (
            f"/models/{model_id}/batch/structures"
            if job_type == "/models"
            else job_type
        ),
        "model_id": model_id,
        "args": {"random_seed": 7},
    }
    route = f"v1/models/{model_id}/batch/structures"
    models_session.post_routes[route] = payload
    submitted = _submit(ModelsAPI(models_session.session), model_id, ensemble_reference)
    assert type(submitted) is StructurePredictionFuture

    job_id = submitted.id
    models_session.outputs[(job_id, "structures")] = [
        Stored(None, _ensemble(ensemble_reference, shifts).encode())
        for shifts in SHIFTS
    ]
    if is_flow:
        models_session.outputs[(job_id, "plddt")] = [
            [[80.0 + shift] * len(ensemble_reference) for shift in shifts]
            for shifts in SHIFTS
        ]
        models_session.outputs[(job_id, "confidence")] = [0.8, 0.81]

    for output in outputs:
        for position in range(2):
            models_session.get_routes[
                f"v1/data/results/{job_id}/{output}/{position}"
            ] = {
                "position": position,
                "unit_id": f"u{position}",
                "value": models_session._value(job_id, output, position),
            }

    job = Job.create(payload)
    monkeypatch.setattr(job_api, "job_get", lambda session, job_id: job)
    jobs = JobsAPI(models_session.session)
    future = (
        jobs.get(job_id)
        if operation == "get"
        else _dispatch.create(models_session.session, job)
        if operation == "explicit"
        else submitted
    )
    assert isinstance(future, StructurePredictionFuture)
    assert type(future) is StructurePredictionFuture
    assert future.args == payload["args"]

    for values in (
        future.get(),
        future.wait(),
        list(future.stream()),
        list(future),
    ):
        assert [_shifts(value, ensemble_reference) for value in values] == SHIFTS

    for position, shifts in enumerate(SHIFTS):
        for value in (
            future.get_item(position),
            future[position],
            future.unit(position).get(),
        ):
            assert _shifts(value, ensemble_reference) == shifts

    if is_flow:
        assert future.get(output="confidence") == [0.8, 0.81]
        assert future.unit(1, output="confidence").get() == 0.81
        plddt = np.asarray(future.get_item(1, output="plddt"))
        assert plddt[:, 0].tolist() == [80.0 + shift for shift in SHIFTS[1]]


@pytest.mark.parametrize("model_id", IDS)
def test_fold_records_keep_their_compatibility_reader(
    models_session,
    ensemble_reference,
    monkeypatch,
    model_id,
):
    payload = {
        **SUBMITTED_JOB,
        "job_type": JobType.embeddings_fold.value,
    }
    job = Job.create(payload)
    monkeypatch.setattr(job_api, "job_get", lambda session, job_id: job)
    monkeypatch.setattr(
        fold_api,
        "fold_get",
        lambda session, job_id: FoldMetadata(job_id=job_id, model_id=model_id),
    )
    monkeypatch.setattr(
        fold_api,
        "fold_get_sequences",
        lambda session, job_id: [ensemble_reference.sequence],
    )
    future = JobsAPI(models_session.session).get(job.job_id)
    assert type(future) is FoldResultFuture


@pytest.mark.parametrize("job_type", ["/models", JobType.embeddings_fold.value])
def test_explicit_task_reconstruction_needs_method_identity(mock_session, job_type):
    job = Job.create({**SUBMITTED_JOB, "job_type": job_type})
    future = _dispatch.create(mock_session, job)
    assert type(future) is Future
    assert future.id == job.job_id
    mock_session.get.assert_not_called()
