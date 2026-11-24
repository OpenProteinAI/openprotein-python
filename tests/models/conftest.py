"""Fixtures serving cards, params and outputs from ``support.ModelsSession``."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from openprotein.align import AlignAPI
from openprotein.jobs import Future, Job, JobsAPI, JobStatus
from openprotein.jobs import _api as jobs_api
from openprotein.jobs._storage import DirectArtifactStorage
from openprotein.models.schemas import MethodDescriptor
from openprotein.molecules import Complex
from tests.models.support import ModelsSession, card


@pytest.fixture
def models_session(monkeypatch) -> ModelsSession:
    session = ModelsSession()
    monkeypatch.setattr(
        DirectArtifactStorage,
        "iter_bytes",
        lambda self, reference: iter([session.blobs[reference["uri"]]]),
    )
    return session


@pytest.fixture
def model_card():
    return card


@pytest.fixture
def task_outputs(mock_session, monkeypatch):
    def make(
        future_type,
        *,
        path="generate/structures",
        provider="rfdiffusion",
        values=None,
    ):
        values = values or {
            "structures": [
                (Path(__file__).parents[1] / "data" / "test_protein.cif").read_text()
            ]
        }
        primary = next(iter(values))
        job = Job.model_validate(
            {
                "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
                "job_type": f"/models/{provider}/{path}",
                "status": "SUCCESS",
                "created_date": "2026-10-21T08:00:00Z",
            }
        )

        def entries(self, job_id, output, *, follow=False, interval=5):
            name = output or primary
            return iter(
                SimpleNamespace(position=i, unit_id="shared", value=value)
                for i, value in enumerate(values[name])
            )

        def item(self, job_id, position, output):
            return SimpleNamespace(
                position=position,
                unit_id="shared",
                value=values[output][position],
            )

        monkeypatch.setattr(JobsAPI, "_iter_job_outputs", entries)
        monkeypatch.setattr(JobsAPI, "_get_job_output", item)
        monkeypatch.setattr(
            JobsAPI,
            "_iter_job_units",
            lambda self, job_id: iter(
                SimpleNamespace(
                    position=i, unit_id="shared", status=JobStatus.SUCCESS, error=None
                )
                for i in range(len(values[primary]))
            ),
        )
        monkeypatch.setattr(
            jobs_api, "job_default_output", lambda session, job_id: primary
        )
        monkeypatch.setattr(
            jobs_api,
            "get_unit",
            lambda session, job_id, position: SimpleNamespace(
                status=JobStatus.SUCCESS,
                unit_id="shared",
                error=None,
            ),
        )
        monkeypatch.setattr(Future, "_wait_job", lambda self, **kwargs: self.job)
        monkeypatch.setattr("openprotein.jobs.futures.time.sleep", lambda seconds: None)

        future = future_type(mock_session, job)
        future._resolved_descriptor = MethodDescriptor(path=path, outputs=list(values))
        return future, values

    return make


@pytest.fixture
def ensemble_reference():
    path = (
        Path(__file__).parents[1]
        / "data"
        / "boltzgen_entity_one_letter_missing_residues.cif"
    )
    return Complex.from_filepath(path, verbose=False).get_proteins()["A"]


@pytest.fixture
def aligned_ensemble_session(models_session, ensemble_reference, monkeypatch):
    align = AlignAPI(models_session.session)
    monkeypatch.setattr(
        align,
        "get_seed",
        lambda job_id: ensemble_reference.sequence.decode(),
    )
    models_session.session.align = align
    return models_session
