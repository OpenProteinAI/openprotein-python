"""Test the schemas for the predictor domain."""

from datetime import datetime

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobStatus, JobType
from openprotein.models.kernels import (
    LinearKernel,
    _GPHyperparameters,
    _trained_kernel,
)
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


@pytest.mark.parametrize(
    "kernel,served",
    [
        ("linear", {"outputscale": [2.0, 3.0]}),
        ("linear", {"lengthscale": [None, 0.8]}),
        ("linear", {"alpha": [None, 1.2]}),
        ("rbf", {"lengthscale": [0.8], "period": [2.0, 3.0]}),
        ("rbf", {"lengthscale": [0.8, 0.9]}),
        ("rbf", {"lengthscale": []}),
    ],
)
def test_trained_kernel_takes_exactly_one_kernel(kernel, served):
    hyperparameters = _GPHyperparameters(**({"outputscale": [2.0]} | served))

    with pytest.raises(ValueError, match="expected one"):
        _trained_kernel(kernel, hyperparameters)


@pytest.mark.parametrize(
    "served", [{}, {"lengthscale": [None], "period": [None], "alpha": [None]}]
)
def test_inapplicable_hyperparameters_may_be_empty_or_null(served):
    hyperparameters = _GPHyperparameters(outputscale=[2.0], **served)

    assert _trained_kernel("linear", hyperparameters) == LinearKernel(outputscale=2.0)


def test_predictor_metadata_is_done():
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
    job_data = {
        "job_id": "job-123",
        "status": JobStatus.SUCCESS,
        "created_date": datetime.now(),
        "job_type": expected_job_type.value,
    }
    job = job_class(**job_data)
    assert job.job_type == expected_job_type


def test_features_schema():
    # This should pass
    Features(type=FeatureType.PLM, model_id="model-1", reduction="mean")
    Features(type=FeatureType.SVD, model_id="model-1", reduction="mean")
