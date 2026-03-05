"""Tests for shared structure-generation futures."""

from unittest.mock import MagicMock, patch

import pytest

from openprotein.jobs import JobStatus
from openprotein.models.foundation.boltzgen import BoltzGenFuture
from openprotein.models.foundation.rfdiffusion import RFdiffusionFuture
from openprotein.models.structure_generation import (
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


def test_rfdiffusion_future_emits_deprecation_warning():
    """RFdiffusionFuture remains as a deprecated alias."""
    session = MagicMock()
    with pytest.warns(DeprecationWarning, match="RFdiffusionFuture is deprecated"):
        RFdiffusionFuture(session=session, job=_make_job(), N=1, result_format="pdb")


def test_boltzgen_future_emits_deprecation_warning():
    """BoltzGenFuture remains as a deprecated alias."""
    session = MagicMock()
    with pytest.warns(DeprecationWarning, match="BoltzGenFuture is deprecated"):
        BoltzGenFuture(session=session, job=_make_job(), N=1, result_format="cif")
