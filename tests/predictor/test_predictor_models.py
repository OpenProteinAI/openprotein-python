"""Test the model logic for the predictor domain."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from openprotein.common import FeatureType
from openprotein.jobs import JobsAPI, JobStatus, JobType
from openprotein.models.kernels import (
    LinearKernel,
    PeriodicKernel,
    RationalQuadraticKernel,
    RBFKernel,
)
from openprotein.predictor.models import PredictorModel, PredictorModelGroup
from openprotein.predictor.schemas import (
    Dataset,
    Features,
    Kernel,
    ModelSpec,
    PredictorMetadata,
    PredictorTrainJob,
    PredictorType,
)


@pytest.fixture
def mock_predictor_metadata():
    return PredictorMetadata(
        id="p1",
        name="p1",
        status=JobStatus.SUCCESS,
        created_date=datetime.now(),
        model_spec=ModelSpec(
            type=PredictorType.GP,
            features=Features(type=FeatureType.PLM, model_id="m1", reduction="mean"),
        ),
        training_dataset=Dataset(assay_id="a1", properties=["p1"]),
    )


@pytest.fixture
def mock_predictor_train_job():
    return PredictorTrainJob(
        job_id="p1",
        status=JobStatus.SUCCESS,
        job_type=JobType.predictor_train,
        created_date=datetime.now(),
    )


def test_predictor_model_init(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    # The constructor expects session.jobs to be a JobsAPI instance
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)
    assert model.id == "p1"


def test_predictor_model_name_from_metadata(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    """PredictorModel.name and .description come from the predictor's own metadata, not the job."""
    mock_session.jobs = MagicMock(spec=JobsAPI)
    job = mock_predictor_train_job.model_copy(
        update={"name": "renamed-job", "description": "job note"}
    )
    mock_session.jobs.get_job.return_value = job
    metadata = mock_predictor_metadata.model_copy(
        update={"name": "predictor-name", "description": "predictor note"}
    )
    model = PredictorModel(session=mock_session, metadata=metadata)

    assert model.name == "predictor-name"
    assert model.description == "predictor note"


def test_predictor_model_predict(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)

    with patch("openprotein.predictor.api.predictor_predict_post") as mock_predict_post:
        with patch("openprotein.predictor.prediction.PredictionResultFuture.create"):
            model.predict(sequences=[b"ACGT"])
            mock_predict_post.assert_called_once()


def test_predictor_model_group(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model1 = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)
    model2 = PredictorModel(
        session=mock_session,
        metadata=mock_predictor_metadata.model_copy(update={"id": "p2"}),
    )

    group = model1 | model2
    assert isinstance(group, PredictorModelGroup)
    assert len(group.__models__) == 2

    with patch(
        "openprotein.predictor.api.predictor_predict_multi_post"
    ) as mock_predict_multi:
        with patch("openprotein.predictor.prediction.PredictionResultFuture.create"):
            group.predict(sequences=[b"ACGT"])
            mock_predict_multi.assert_called_once()
            call_args, call_kwargs = mock_predict_multi.call_args
            assert call_kwargs["predictor_ids"] == ["p1", "p2"]


def test_predictor_model_does_not_rename(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = mock_predictor_train_job
    model = PredictorModel(session=mock_session, metadata=mock_predictor_metadata)

    with pytest.raises(NotImplementedError, match="set when it is trained"):
        model.rename(name="other")
    mock_session.put.assert_not_called()


def _hyperparameters(lengthscale, period=None, alpha=None) -> dict:
    return {
        "p1": {
            "mean": 0.5,
            "noise": 0.1,
            "lengthscale": [lengthscale],
            "outputscale": [2.0],
            "period": [period],
            "alpha": [alpha],
        }
    }


def _gp(mock_session, metadata, train_job, kernel: str, hyperparameters: dict):
    mock_session.jobs = MagicMock(spec=JobsAPI)
    mock_session.jobs.get_job.return_value = train_job
    mock_session.get.return_value.json.return_value = hyperparameters
    spec = metadata.model_spec.model_copy(
        update={"kernel": Kernel(type=kernel.upper())}
    )
    return PredictorModel(
        session=mock_session, metadata=metadata.model_copy(update={"model_spec": spec})
    )


@pytest.mark.parametrize(
    "kernel,hyperparameters,expected",
    [
        ("linear", _hyperparameters(None), LinearKernel(outputscale=2.0)),
        ("rbf", _hyperparameters(0.8), RBFKernel(outputscale=2.0, lengthscale=0.8)),
        (
            "periodic",
            _hyperparameters(0.8, period=3.0),
            PeriodicKernel(outputscale=2.0, lengthscale=0.8, period=3.0),
        ),
        (
            "rational_quadratic",
            _hyperparameters(0.8, alpha=1.5),
            RationalQuadraticKernel(outputscale=2.0, lengthscale=0.8, alpha=1.5),
        ),
    ],
)
def test_predictor_kernel_has_its_type_hyperparameters(
    mock_session,
    mock_predictor_metadata,
    mock_predictor_train_job,
    kernel,
    hyperparameters,
    expected,
):
    model = _gp(
        mock_session,
        mock_predictor_metadata,
        mock_predictor_train_job,
        kernel,
        hyperparameters,
    )

    assert model.kernel == expected
    assert type(model.kernel) is type(expected)
    assert model.kernel.type == kernel
    assert (model.mean, model.noise) == (0.5, 0.1)


def test_rbf_kernel_has_no_period(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    model = _gp(
        mock_session,
        mock_predictor_metadata,
        mock_predictor_train_job,
        "rbf",
        _hyperparameters(0.8),
    )

    assert not hasattr(model.kernel, "period")
    assert not hasattr(model.kernel, "alpha")


def test_predictor_hyperparameters_are_fetched_once(
    mock_session, mock_predictor_metadata, mock_predictor_train_job
):
    model = _gp(
        mock_session,
        mock_predictor_metadata,
        mock_predictor_train_job,
        "rbf",
        _hyperparameters(0.8),
    )

    for _ in range(2):
        _ = (model.kernel, model.mean, model.noise)

    hyperparameter_reads = [
        c
        for c in mock_session.get.call_args_list
        if c.args[0].endswith("/hyperparameters")
    ]
    assert len(hyperparameter_reads) == 1


@pytest.mark.parametrize("lengthscale", [[None], []])
def test_a_rejected_kernel_is_read_again(
    mock_session, mock_predictor_metadata, mock_predictor_train_job, lengthscale
):
    rejected = _hyperparameters(0.8)
    rejected["p1"]["lengthscale"] = lengthscale
    model = _gp(
        mock_session,
        mock_predictor_metadata,
        mock_predictor_train_job,
        "rbf",
        rejected,
    )

    assert model.mean == 0.5
    with pytest.raises(ValueError):
        _ = model.kernel
    mock_session.get.return_value.json.return_value = _hyperparameters(0.8)

    assert model.kernel == RBFKernel(outputscale=2.0, lengthscale=0.8)
    hyperparameter_reads = [
        c
        for c in mock_session.get.call_args_list
        if c.args[0].endswith("/hyperparameters")
    ]
    assert len(hyperparameter_reads) == 2


@pytest.mark.parametrize(
    "kernel,hyperparameters,expected",
    [
        ("rbf", _hyperparameters(0.8), ["lengthscale", "outputscale", "type"]),
        (
            "periodic",
            _hyperparameters(0.8, period=3.0),
            ["lengthscale", "outputscale", "period", "type"],
        ),
    ],
)
def test_ipython_completes_the_attributes_of_the_trained_kernel(
    mock_session,
    mock_predictor_metadata,
    mock_predictor_train_job,
    kernel,
    hyperparameters,
    expected,
):
    pytest.importorskip("IPython")
    from IPython.core.completer import provisionalcompleter
    from IPython.core.interactiveshell import InteractiveShell

    model = _gp(
        mock_session,
        mock_predictor_metadata,
        mock_predictor_train_job,
        kernel,
        hyperparameters,
    )
    shell = InteractiveShell.instance()
    shell.user_ns["predictor"] = model
    text = "predictor.kernel."
    with provisionalcompleter():
        completions = {
            c.text.rsplit(".", 1)[-1]
            for c in shell.Completer.completions(text, len(text))
        }

    assert (
        sorted(completions & {"lengthscale", "outputscale", "period", "alpha", "type"})
        == expected
    )
