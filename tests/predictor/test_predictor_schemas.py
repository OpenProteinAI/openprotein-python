"""Test the schemas for the predictor domain."""

from datetime import datetime

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType
from openprotein.predictor.schemas import (
    Dataset,
    Features,
    ModelSpec,
    PredictJob,
    PredictorCVJob,
    PredictorMetadata,
    PredictorTrainJob,
    PredictorType,
    PredictSingleSiteJob,
)


def test_predictor_metadata_is_done():
    """Test the is_done() method on PredictorMetadata."""
    spec = ModelSpec(type=PredictorType.GP)
    dataset = Dataset(assay_id="a1", properties=["p1"])
    metadata_done = PredictorMetadata(
        id="p1",
        name="p1",
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
        model_spec=spec,
        training_dataset=dataset,
    )
    metadata_not_done = metadata_done.model_copy(update={"status": JobStatus.PENDING})
    assert metadata_done.is_done() is True
    assert metadata_not_done.is_done() is False


@pytest.mark.parametrize(
    "job_class, expected_job_type",
    [
        (PredictorTrainJob, JobType.predictor_train),
        (PredictJob, JobType.predictor_predict),
        (PredictSingleSiteJob, JobType.predictor_predict_single_site),
        (PredictorCVJob, JobType.predictor_crossvalidate),
    ],
)
def test_job_schemas_job_type(job_class, expected_job_type):
    """Test that each job schema has the correct job_type."""
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": expected_job_type.value,
    }
    job = job_class(**job_data)
    assert job.job_type == expected_job_type


def test_features_schema():
    """Test the Features schema validation."""
    # This should pass
    Features(type=FeatureType.PLM, model_id="model-1", reduction="mean")
    Features(type=FeatureType.SVD, model_id="model-1", reduction="mean")
