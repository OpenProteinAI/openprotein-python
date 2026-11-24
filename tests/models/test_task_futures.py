import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openprotein.errors import APIError
from openprotein.jobs import futures as results
from openprotein.jobs.futures import _ResultFuture
from openprotein.models import _inputs
from openprotein.models.structure_generation import (
    BoltzGenFuture,
    RFdiffusionFuture,
    StructureGenerationFuture,
    StructureGenerationJob,
)
from openprotein.molecules import Complex, Structure


class _TaggedFuture(_ResultFuture):
    _primary_output = "structures"

    def _convert_output(self, output, position, value):
        return output, position, value


_READERS = [
    lambda f: f.get()[0],
    lambda f: f.wait(interval=0)[0],
    lambda f: f.get_item(0),
    lambda f: f[0],
    lambda f: next(f.stream()),
    lambda f: next(iter(f)),
    lambda f: f.unit(0).get(),
]


@pytest.mark.parametrize("read", _READERS)
def test_positional_readers_apply_the_result_conversion(task_outputs, read):
    future, _ = task_outputs(_TaggedFuture, values={"structures": ["value"]})
    assert read(future) == ("structures", 0, "value")


def test_task_repr_names_the_concrete_type(task_outputs):
    future, _ = task_outputs(_TaggedFuture)
    assert repr(future).startswith("_TaggedFuture(")


@pytest.mark.parametrize(
    ("future_type", "path", "output"),
    [
        (results.EmbeddingFuture, "batch/embeddings", "embeddings"),
        (results.LogitsFuture, "batch/logits", "logits"),
        (results.AttentionFuture, "batch/attn", "attn"),
    ],
)
def test_array_tasks_read_their_primary_output(task_outputs, future_type, path, output):
    array = np.ones((2, 3))
    future, _ = task_outputs(
        future_type,
        path=path,
        provider="esm2",
        values={"auxiliary": ["other"], output: [array]},
    )

    assert future.get_item(0) is array
    assert future.get()[0] is array
    assert future.get(output="auxiliary") == ["other"]


_DATA = Path(__file__).parents[1] / "data"


@pytest.mark.parametrize("read", _READERS)
@pytest.mark.parametrize(
    ("structure_file", "chains"),
    [
        ("test_protein.cif", ["A"]),
        ("boltzgen_entity_one_letter_missing_residues.cif", ["A", "B"]),
    ],
)
def test_design_accessors_return_complexes(task_outputs, read, structure_file, chains):
    future, _ = task_outputs(
        results.StructureDesignFuture,
        values={"structures": [(_DATA / structure_file).read_text()]},
    )
    value = read(future)
    assert isinstance(value, Complex)
    assert sorted(value.get_chains()) == chains


def test_design_returns_decoded_complexes_as_is(task_outputs):
    _, values = task_outputs(results.StructureDesignFuture)
    decoded = Complex.from_string(values["structures"][0], format="cif", verbose=False)
    future, _ = task_outputs(
        results.StructureDesignFuture, values={"structures": [decoded]}
    )
    assert future.get_item(0) is decoded


def test_design_rejects_a_complex_without_chains(task_outputs):
    future, _ = task_outputs(
        results.StructureDesignFuture, values={"structures": [Complex({})]}
    )
    with pytest.raises(
        ValueError,
        match=f"job {future.id} position 0: .*contains no molecular chains",
    ):
        future.get_item(0)


def test_prediction_returns_decoded_structures_as_is(task_outputs):
    _, values = task_outputs(results.StructureDesignFuture)
    decoded = Structure(
        [Complex.from_string(values["structures"][0], format="cif", verbose=False)]
    )
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="boltz-2",
        values={"structures": [decoded]},
    )
    assert future.get_item(0) is decoded


def test_structure_auxiliary_outputs_keep_their_values(task_outputs):
    confidence = np.ones((2, 2))
    score = b"ptm,iptm\n0.8,0.7\n"
    structure = (Path(__file__).parents[1] / "data" / "test_protein.cif").read_text()
    future, _ = task_outputs(
        results.StructureDesignFuture,
        values={
            "structures": [structure],
            "confidence": [confidence],
            "score": [score],
        },
    )

    assert future.get_item(0, output="confidence") is confidence
    assert future.unit(0, output="confidence").get() is confidence
    assert list(future.stream("score")) == [score]


def test_prediction_preserves_every_model(task_outputs):
    _, values = task_outputs(results.StructureDesignFuture)
    first = Complex.from_string(values["structures"][0], format="cif", verbose=False)
    text = Structure([first, first.copy()]).to_string(format="cif")
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="boltz-2",
        values={"structures": [text]},
    )

    value = future.get_item(0)

    assert isinstance(value, Structure)
    assert len(value) == 2
    assert all(isinstance(value[i], Complex) for i in range(2))


def test_design_rejects_multiple_models_in_one_position(task_outputs):
    future, values = task_outputs(results.StructureDesignFuture)
    first = Complex.from_string(values["structures"][0], format="cif", verbose=False)
    values["structures"][0] = Structure([first, first.copy()]).to_string(format="cif")

    with pytest.raises(ValueError, match=f"job {future.id} position 0: .*one complex"):
        future.get_item(0)


def test_structure_type_errors_are_not_returned_as_results(task_outputs):
    future, _ = task_outputs(
        results.StructureDesignFuture,
        values={"structures": [{"unexpected": "value"}]},
    )

    with pytest.raises(
        TypeError, match=f"job {future.id} position 0: structure output"
    ):
        future.get_item(0)
    with pytest.raises(APIError) as error:
        future.get()
    assert isinstance(error.value.__cause__, TypeError)


def test_malformed_structures_report_the_parser_failure(task_outputs):
    future, _ = task_outputs(
        results.StructureDesignFuture,
        values={"structures": ["not a cif document"]},
    )

    with pytest.raises(
        (ValueError, RuntimeError), match=f"job {future.id} position 0: "
    ):
        future.get_item(0)
    with pytest.raises(APIError) as error:
        future.get()
    assert isinstance(error.value.__cause__, (ValueError, RuntimeError))


def test_design_type_error_names_the_specialised_future():
    with pytest.raises(TypeError, match="StructureDesignFuture"):
        _inputs._check_design("job-id")


def test_equal_unit_identities_keep_distinct_positions(task_outputs):
    future, _ = task_outputs(
        results.SequenceGenerationFuture,
        path="generate/sequences",
        values={"sequences": ["MKT", "MKT"]},
    )

    assert future.get() == ["MKT", "MKT"]
    assert [unit.position for unit in future.units] == [0, 1]
    assert [unit.get() for unit in future.units] == ["MKT", "MKT"]


@pytest.mark.parametrize(
    ("future_type", "method"),
    [
        (RFdiffusionFuture, "session.models.rfdiffusion.generate"),
        (BoltzGenFuture, "session.models.boltzgen.generate"),
    ],
)
def test_design_aliases_name_the_current_submission_method(
    mock_session, future_type, method
):
    job = StructureGenerationJob(
        job_id="design-1",
        job_type="/models/design",
        status="SUCCESS",
        created_date="2026-10-21T08:00:00Z",
    )
    mock_session.get.return_value.text = (
        Path(__file__).parents[1] / "data" / "test_protein.cif"
    ).read_text()

    with pytest.warns(FutureWarning) as warnings:
        future = future_type(mock_session, job, N=1, result_format="cif", max_workers=0)

    message = str(warnings[0].message)
    assert method in message
    assert "StructureDesignFuture" in message
    assert "0.18.0" in message
    assert "0.19.0" in message
    assert isinstance(future, StructureGenerationFuture)
    assert not isinstance(future, results.StructureDesignFuture)
    assert isinstance(future.get_item(0), Complex)
    mock_session.get.assert_called_once_with(
        "v1/design/design-1/results", params={"replicate": 0}
    )


def test_result_futures_iterate_through_their_stream(task_outputs, monkeypatch):
    sentinel = iter(["streamed"])
    monkeypatch.setattr(
        results.EmbeddingFuture, "stream", lambda self, *args, **kwargs: sentinel
    )
    future, _ = task_outputs(results.EmbeddingFuture)

    assert iter(future) is sentinel


def test_prediction_conversion_errors_name_the_job_and_position(task_outputs):
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="boltz-2",
        values={"structures": [{"unexpected": "value"}]},
    )

    with pytest.raises(
        TypeError, match=f"job {future.id} position 0: structure output"
    ):
        future.get_item(0)


@pytest.mark.parametrize("output", ["score", "metrics"])
@pytest.mark.parametrize("stored", [bytes, str])
def test_prediction_reads_csv_outputs_as_one_table_per_position(
    task_outputs, output, stored
):
    text = "example_id,ptm,batch_idx\nbatch_0,0.8,0\nbatch_0,0.7,1\n"
    value = text.encode() if stored is bytes else text
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="rosettafold-3",
        values={"structures": ["unused"], output: [value]},
    )
    expected = pd.DataFrame(
        {"example_id": ["batch_0", "batch_0"], "ptm": [0.8, 0.7], "batch_idx": [0, 1]}
    )

    for table in (
        future.get_item(0, output=output),
        future.get(output=output)[0],
        future.unit(0, output=output).get(),
        next(future.stream(output)),
    ):
        pd.testing.assert_frame_equal(table, expected)


def test_prediction_keeps_decoded_table_outputs(task_outputs):
    decoded = {"ptm": [0.8, 0.7]}
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="rosettafold-3",
        values={"structures": ["unused"], "metrics": [decoded]},
    )

    assert future.get_item(0, output="metrics") is decoded


def test_prediction_table_errors_name_the_job_and_position(task_outputs):
    future, _ = task_outputs(
        results.StructurePredictionFuture,
        path="batch/structures",
        provider="rosettafold-3",
        values={"structures": ["unused"], "score": [b""]},
    )

    with pytest.raises(ValueError, match=f"job {future.id} position 0: "):
        future.get_item(0, output="score")


NAN = float("nan")


@pytest.mark.parametrize(
    "future_type, path, output, values, expected",
    [
        (
            results.LogLikelihoodFuture,
            "batch/loglikelihood",
            None,
            {"loglikelihood": ["-inf", 1.5]},
            [-math.inf, 1.5],
        ),
        (
            results.SingleSiteScoreFuture,
            "single-site/scores",
            None,
            {"score": ["nan", ["inf", 0.5]]},
            [NAN, [math.inf, 0.5]],
        ),
        (
            results.IndelScoreFuture,
            "indel/scores",
            None,
            {"score": [0.0, "-inf"]},
            [0.0, -math.inf],
        ),
        (
            results.SequenceGenerationFuture,
            "generate/sequences",
            "loglikelihood",
            {"sequences": ["MKT", "MKV"], "loglikelihood": ["-inf", -2.0]},
            [-math.inf, -2.0],
        ),
        (
            results.PredictionFuture,
            "predict",
            None,
            {"predictions": [{"y": {"mean": "nan", "variance": 1.0}}]},
            [{"y": {"mean": NAN, "variance": 1.0}}],
        ),
        (
            results.SingleSitePredictionFuture,
            "single-site/predict",
            None,
            {"predictions": [{"y": {"mean": 0.1, "variance": "inf"}}]},
            [{"y": {"mean": 0.1, "variance": math.inf}}],
        ),
    ],
    ids=[
        "loglikelihood",
        "single site",
        "indel",
        "generation",
        "prediction",
        "single-site prediction",
    ],
)
def test_spelt_out_non_finite_numbers_read_as_floats(
    task_outputs, future_type, path, output, values, expected
):
    future, _ = task_outputs(future_type, path=path, provider="m", values=values)

    # repr tells the float nan from the string 'nan'
    assert repr(future.get(output=output)) == repr(expected)
    assert repr(future.get_item(0, output=output)) == repr(expected[0])


def test_a_generation_keeps_its_sequences_as_strings(task_outputs):
    future, _ = task_outputs(
        results.SequenceGenerationFuture,
        path="generate/sequences",
        provider="m",
        values={"sequences": ["nan", "MKT"], "loglikelihood": [1.0, 2.0]},
    )

    assert future.get() == ["nan", "MKT"]
