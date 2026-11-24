"""A verb takes a query or a design, not both, and refuses an empty query."""

import pytest

from openprotein.jobs import Future, Job
from openprotein.models import ModelsAPI
from tests.models.support import SUBMITTED_JOB
from tests.models.test_handles import DESIGN_UUID, QUERY_UUID


def _design(session) -> Future:
    job = Job.create(
        SUBMITTED_JOB
        | {"job_id": DESIGN_UUID, "job_type": "/models/rfdiffusion/generate/structures"}
    )
    return Future.create(session.session, job=job)


class TestStructureSources:
    def test_a_query_and_a_design_are_refused_together(self, models_session):
        models = ModelsAPI(models_session.session)

        with pytest.raises(TypeError, match="pass query or design, not both"):
            models.poet2.generate(query=QUERY_UUID, design=_design(models_session))

        assert models_session.queries == []
        assert models_session.submissions == []

    def test_an_empty_query_is_refused(self, models_session):
        models = ModelsAPI(models_session.session)

        with pytest.raises(ValueError, match="query has no entries; pass query=None"):
            models.poet2.generate(query=[])

        assert models_session.submissions == []
