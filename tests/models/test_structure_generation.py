"""Tests for shared structure-generation futures."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.jobs import JobStatus
from openprotein.models.structure_generation import (
    BoltzGenFuture,
    RFdiffusionFuture,
    StructureGenerationFuture,
    StructureGenerationJob,
)


def _make_job() -> StructureGenerationJob:
    return StructureGenerationJob(
        job_id="design-job-1",
        job_type="/models/design",
        status=JobStatus.SUCCESS,
        created_date="2026-01-01T00:00:00",
    )


def test_structure_generation_future_get_item_uses_result_format():
    """The shared future fetches a replicate and parses it with configured format."""
    session = MagicMock()
    session.get.return_value.text = "MOCK_STRUCTURE_TEXT"
    future = StructureGenerationFuture(
        session=session, job=_make_job(), N=1, result_format="cif"
    )

    with patch(
        "openprotein.models.structure_generation.Complex.from_string",
        return_value=MagicMock(),
    ) as mock_from_string:
        future.get_item(0)

    session.get.assert_called_once_with(
        "v1/design/design-job-1/results", params={"replicate": 0}
    )
    mock_from_string.assert_called_once_with("MOCK_STRUCTURE_TEXT", format="cif")


@pytest.mark.parametrize(
    ("future_class", "method"),
    [(RFdiffusionFuture, "rfdiffusion"), (BoltzGenFuture, "boltzgen")],
)
def test_deprecated_future_aliases_warn_on_construction(future_class, method):
    name = future_class.__name__
    with pytest.warns(FutureWarning) as record:
        future_class(session=MagicMock(), job=_make_job(), N=1, result_format="pdb")
    expected = (
        f"`openprotein.models.{name}` is deprecated since 0.18.0 and will be "
        f"removed in 0.19.0. Use `session.models.{method}.generate(...)` "
        "(returns `openprotein.jobs.StructureDesignFuture`)."
    )
    assert [str(w.message) for w in record] == [expected]
    assert record[0].filename == __file__


@pytest.mark.parametrize("model", ["rfdiffusion", "boltzgen"])
def test_a_design_query_with_a_length_range_is_refused_before_any_upload(
    models_session, model
):
    """A design query needs a fixed length, so a length range is refused up front."""
    from openprotein.models import ModelsAPI
    from openprotein.molecules import Protein

    models = ModelsAPI(models_session.session)

    with pytest.raises(ValueError, match=r"length range.*Protein\.from_expr\(15\)"):
        getattr(models, model).generate(Protein.from_expr("10..20"))

    assert models_session.queries == []
    assert models_session.submissions == []


def test_a_model_without_generate_is_refused_with_its_missing_method(models_session):
    """A model whose card lacks `generate` raises the missing-method error."""
    from openprotein.models import ModelsAPI
    from openprotein.molecules import Protein
    from tests.models.support import card

    listing = card("rfdiffusion")
    listing["methods"] = [
        m for m in listing["methods"] if m["path"] != "generate/structures"
    ]
    models_session.get_routes["v1/models/rfdiffusion"] = listing
    models = ModelsAPI(models_session.session)

    with pytest.raises(ValueError, match="has no generate"):
        models.rfdiffusion.generate(Protein.from_expr(15))

    assert models_session.queries == []
    assert models_session.submissions == []
    assert not any(
        call.args[0].endswith("/params")
        for call in models_session.session.get.call_args_list
    )
