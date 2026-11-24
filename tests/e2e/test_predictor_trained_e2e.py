"""E2E tests on a few predictors trained once per module: crossvalidation, ensembles,
single-site predictions and deletion."""

import io
import os

import numpy as np
import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.common.reduction import ReductionType
from openprotein.data import AssayDataset
from openprotein.errors import HTTPError, InvalidParameterError, JobFailedException
from openprotein.jobs import CrossvalidateResults
from openprotein.models import GPModel
from openprotein.predictor import api
from openprotein.predictor.models import PredictorModel
from tests.e2e.config import scaled_timeout
from tests.e2e.conftest import TEST_ASSAY_DATA_DIR, TEST_ASSAY_SMALL_FILE

pytestmark = [pytest.mark.e2e, pytest.mark.xdist_group("predictor_trained")]

TIMEOUT = scaled_timeout(2.0)

# one null measurement, at row 3, so crossvalidation scores 14 of the 15 rows
PROPERTY = "isobutyramide_normalized_fitness"

# three folds over the 15 assay rows, with rows 0 and 5 excluded
FOLDS = [-1 if i in (0, 5) else i % 3 for i in range(15)]


@pytest.fixture(scope="module")
def assay_table() -> pd.DataFrame:
    return pd.read_csv(os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_SMALL_FILE))


@pytest.fixture(scope="module")
def scored_rows(assay_table: pd.DataFrame) -> list[int]:
    return assay_table.index[assay_table[PROPERTY].notna()].tolist()


@pytest.fixture(scope="module")
def predictors(
    session: OpenProtein, assay_small: AssayDataset
) -> dict[str, PredictorModel]:
    """rbf and linear GPs on ESM2, which crossvalidate through the GP v1 and
    BayesReg paths, and an rbf GP on prot-seq to ensemble with the first."""
    esm2, prot_seq = session.embedding.esm2, session.embedding.get_model("prot-seq")
    futures = {
        "rbf": esm2.fit_gp(
            assay=assay_small,
            properties=[PROPERTY],
            reduction=ReductionType.MEAN,
            kernel="rbf",
        ),
        "linear": esm2.fit_gp(
            assay=assay_small,
            properties=[PROPERTY],
            reduction=ReductionType.MEAN,
            kernel="linear",
        ),
        "rbf_prot_seq": prot_seq.fit_gp(
            assay=assay_small,
            properties=[PROPERTY],
            reduction=ReductionType.MEAN,
            kernel="rbf",
        ),
    }
    for name, future in futures.items():
        assert future.wait(timeout=TIMEOUT), f"training the {name} predictor failed"
    return futures


@pytest.fixture(scope="module")
def ensemble(
    session: OpenProtein, predictors: dict[str, PredictorModel]
) -> PredictorModel:
    return session.predictor.ensemble([predictors["rbf"], predictors["rbf_prot_seq"]])


def _results(session: OpenProtein, future) -> CrossvalidateResults:
    """The run's rows from its raw result, after checking the returned tuple
    against them."""
    y, y_mu, y_var = future.wait(timeout=TIMEOUT)
    raw = pd.read_csv(io.BytesIO(api.predictor_crossvalidate_get(session, future.id)))
    # an ensemble result has a y_mu/y_var pair per member after its own pair
    result = CrossvalidateResults(
        row_index=raw["row_index"].to_numpy(),
        sequence=raw["sequence"].to_numpy(),
        measurement_name=raw["measurement_name"].to_numpy(),
        fold=raw["fold"].to_numpy() if "fold" in raw else None,
        y=raw["y"].to_numpy(dtype=float),
        y_mu=raw.iloc[:, 4].to_numpy(dtype=float),
        y_var=raw.iloc[:, 5].to_numpy(dtype=float),
    )
    np.testing.assert_array_equal(y, result.y)
    np.testing.assert_array_equal(y_mu, result.y_mu)
    np.testing.assert_array_equal(y_var, result.y_var)
    return result


def _crossvalidate(
    session: OpenProtein, predictor: PredictorModel, **kwargs
) -> CrossvalidateResults:
    return _results(session, predictor.crossvalidate(**kwargs))


def _assert_scored(result: CrossvalidateResults, rows: np.ndarray) -> None:
    assert np.isfinite(result.y_mu[rows]).all()
    assert (result.y_var[rows] >= 0).all()


def _assert_rows_match_assay(
    result: CrossvalidateResults, assay_table: pd.DataFrame, scored_rows: list[int]
) -> None:
    """Each result row carries the sequence and measurement of its assay row."""
    assert sorted(result.row_index.tolist()) == scored_rows
    assert (result.measurement_name == PROPERTY).all()
    np.testing.assert_array_equal(
        result.sequence, assay_table["sequence"].to_numpy()[result.row_index]
    )
    np.testing.assert_allclose(
        result.y, assay_table[PROPERTY].to_numpy()[result.row_index], rtol=1e-6
    )


def _raw_crossvalidation(session: OpenProtein, job_id: str) -> pd.DataFrame:
    raw = api.predictor_crossvalidate_get(session, job_id)
    return pd.read_csv(io.BytesIO(raw)).set_index("row_index").sort_index()


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_crossvalidate_default_split(
    session, predictors, assay_table, scored_rows, kernel: str
):
    """The default split puts every scored row in one of 5 folds."""
    result = _crossvalidate(session, predictors[kernel])

    _assert_rows_match_assay(result, assay_table, scored_rows)
    assert result.fold is not None
    assert sorted(set(result.fold.tolist())) == [0, 1, 2, 3, 4]
    _assert_scored(result, np.ones(len(result.y), dtype=bool))


def test_crossvalidate_n_splits(session, predictors, scored_rows):
    result = _crossvalidate(session, predictors["rbf"], n_splits=3)

    assert sorted(result.row_index.tolist()) == scored_rows
    assert result.fold is not None
    assert sorted(set(result.fold.tolist())) == [0, 1, 2]
    _assert_scored(result, np.ones(len(result.y), dtype=bool))


def _gp(session: OpenProtein, predictor: PredictorModel) -> GPModel:
    gp = session.models.get_model(f"gp/{predictor.id}")
    assert isinstance(gp, GPModel)
    return gp


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_crossvalidate_custom_folds(
    session, predictors, assay_table, scored_rows, kernel: str
):
    """Each row is held out in the fold it was given, and excluded rows are unscored."""
    result = (
        _gp(session, predictors[kernel])
        .crossvalidate(folds=FOLDS)
        .wait(timeout=TIMEOUT)
    )

    assert isinstance(result, CrossvalidateResults)
    _assert_rows_match_assay(result, assay_table, scored_rows)
    assert result.fold is not None
    expected = np.array([FOLDS[i] for i in result.row_index])
    np.testing.assert_array_equal(result.fold, expected)
    excluded = expected == -1
    assert excluded.sum() == 2
    assert np.isnan(result.y_mu[excluded]).all()
    assert np.isnan(result.y_var[excluded]).all()
    _assert_scored(result, ~excluded)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_splits": 3, "folds": FOLDS}, "mutually exclusive"),
        ({"folds": FOLDS[:4]}, r"one fold per row of the training assay \(15\), got 4"),
        ({"folds": [-2] + FOLDS[1:]}, "fold must be -1 to exclude a row or >= 0"),
        (
            {"folds": [0 if f == -1 else f * 2 for f in FOLDS]},
            "labelled 0..2 with none empty",
        ),
        ({"folds": [0] * 15}, "between 2 and 10 folds, got 1"),
    ],
)
def test_crossvalidate_rejects_invalid_folds_client_side(
    session, predictors, kwargs, match
):
    with pytest.raises(InvalidParameterError, match=match):
        _gp(session, predictors["rbf"]).crossvalidate(**kwargs)


@pytest.mark.parametrize("n_splits", [1, 11])
def test_crossvalidate_rejects_n_splits_out_of_range(predictors, n_splits: int):
    with pytest.raises(HTTPError) as error:
        predictors["rbf"].crossvalidate(n_splits=n_splits)

    assert error.value.status_code == 400
    assert (
        f"invalid n_splits: expected between 2 and 10, got {n_splits}"
        in error.value.text
    )


def test_crossvalidate_failure_reports_its_cause(session, predictors):
    """A fold made only of rows without a measurement is emptied when the trainer drops
    them, and the job fails with the trainer's reason."""
    folds = [i % 2 for i in range(15)]
    folds[3] = 2
    future = _gp(session, predictors["rbf"]).crossvalidate(folds=folds)

    with pytest.raises(
        JobFailedException, match=r"folds \[2\] have no training rows"
    ) as error:
        future.wait(timeout=TIMEOUT)
    assert "null measurement" in (error.value.failure_message or "")


def test_ensemble_crossvalidate_matches_its_members(session, predictors, ensemble):
    """An ensemble splits rows like its members and carries each member's predictions."""
    members = {name: predictors[name] for name in ("rbf", "rbf_prot_seq")}
    futures = {name: p.crossvalidate() for name, p in members.items()}
    ensemble_future = ensemble.crossvalidate()
    member_results = {name: _results(session, f) for name, f in futures.items()}
    result = _results(session, ensemble_future)

    order = np.argsort(result.row_index)
    assert result.fold is not None
    for name, member in member_results.items():
        member_order = np.argsort(member.row_index)
        assert member.fold is not None
        np.testing.assert_array_equal(
            result.fold[order],
            member.fold[member_order],
            err_msg=f"{name} folds differ",
        )
    _assert_scored(result, np.ones(len(result.y), dtype=bool))

    raw = _raw_crossvalidation(session, ensemble_future.id)
    for name, member in member_results.items():
        column = next(
            c
            for c in raw.columns
            if c.startswith(f"{members[name].id}:") and c.endswith(":y_mu")
        )
        member_mu = pd.Series(member.y_mu, index=member.row_index).sort_index()
        np.testing.assert_allclose(
            raw[column].to_numpy(),
            member_mu.to_numpy(),
            rtol=1e-6,
            err_msg=f"{name} y_mu differs",
        )


def test_ensemble_predict(session, predictors, ensemble, assay_table):
    """An ensemble prediction carries each member's own prediction, and its mean is the
    members' mean."""
    sequence = assay_table["sequence"][0]
    members = [predictors[name] for name in ("rbf", "rbf_prot_seq")]
    member_futures = {m.id: m.predict(sequences=[sequence]) for m in members}
    future = ensemble.predict(sequences=[sequence])
    mus, variances = future.wait(timeout=TIMEOUT)

    assert mus.shape == (1, 1 + len(members))
    assert variances.shape == mus.shape
    assert np.isfinite(mus).all()
    assert (variances >= 0).all()

    raw = pd.read_csv(
        io.BytesIO(api.predictor_predict_get_batched_result(session, future.id))
    )
    member_mus = []
    for member_id, member_future in member_futures.items():
        member_mu, member_var = member_future.wait(timeout=TIMEOUT)
        columns = [c for c in raw.columns if c.startswith(f"{member_id}:")]
        mu_column = next(c for c in columns if c.endswith(":y_mu"))
        var_column = next(c for c in columns if c.endswith(":y_var"))
        np.testing.assert_allclose(raw[mu_column], member_mu[:, 0], rtol=1e-6)
        np.testing.assert_allclose(raw[var_column], member_var[:, 0], rtol=1e-6)
        member_mus.append(member_mu[:, 0])
    ensemble_mu = next(
        c
        for c in raw.columns
        if c.startswith(f"{ensemble.id}:") and c.endswith(":y_mu")
    )
    np.testing.assert_allclose(raw[ensemble_mu], np.mean(member_mus, axis=0), rtol=1e-6)


def test_ensemble_has_no_hyperparameters(ensemble):
    with pytest.raises(HTTPError) as error:
        _ = ensemble.kernel

    assert error.value.status_code == 404


@pytest.mark.parametrize("kernel", ["rbf", "linear"])
def test_single_site(predictors, assay_table, kernel: str):
    """A single-site scan predicts every substitution of the base sequence and the wildtype."""
    sequence = assay_table["sequence"][0]
    mus, variances = predictors[kernel].single_site(sequence).wait(timeout=TIMEOUT)

    assert mus.shape == variances.shape
    assert mus.shape[1] == 1
    assert mus.shape[0] >= 19 * len(sequence)
    assert np.isfinite(mus).all()
    assert (variances >= 0).all()


def test_delete_predictor(session, predictors):
    """A deleted predictor can't be fetched."""
    disposable = session.predictor.ensemble([predictors["rbf"], predictors["linear"]])

    assert disposable.delete()
    with pytest.raises(HTTPError) as error:
        session.predictor.get_predictor(disposable.id)
    assert error.value.status_code == 404
