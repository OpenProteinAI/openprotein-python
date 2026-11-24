from typing import Any
from unittest.mock import MagicMock

import gemmi
import pytest
from jsonschema import ValidationError

from openprotein.errors import HTTPError
from openprotein.jobs.futures import StructurePredictionFuture
from openprotein.models import (
    AlphaFlow,
    AlphaFlowInput,
    BioEmu,
    BioEmuInput,
    ESMFlow,
    ESMFlowInput,
    ModelsAPI,
    StructureEnsemblePredictionModel,
    StructurePredictionModel,
)
from openprotein.molecules import Protein

HANDLES = {
    "bioemu-v1-1": BioEmu,
    "alphaflow-md-base-v1": AlphaFlow,
    "alphaflow-md-distilled-v1": AlphaFlow,
    "esmflow-md-base-v1": ESMFlow,
    "esmflow-md-distilled-v1": ESMFlow,
}
IDS = tuple(HANDLES)
FLOW_IDS = IDS[1:5]
MSA_ID = "3f1c2a7e-9b4d-4e2a-8c1f-0d5e6a7b8c9d"

CARD_DEFAULTS = {
    BioEmu: {
        "num_samples": 1,
        "denoiser": "dpm",
        "filter_samples": True,
        "random_seed": None,
    },
    AlphaFlow: {
        "num_samples": 1,
        "num_steps": 10,
        "tmax": 1.0,
        "msa_subsample": None,
        "self_cond": None,
        "resample": None,
        "random_seed": None,
    },
    ESMFlow: {
        "num_samples": 1,
        "num_steps": 10,
        "tmax": 1.0,
        "self_cond": None,
        "random_seed": None,
    },
}


def _handle(models_session, model_id) -> Any:
    return ModelsAPI(models_session.session).get_model(model_id)


def _input(model_id: str, sequence: str = "MKT", reference: Any = None) -> Any:
    if model_id == "bioemu-v1-1":
        return BioEmuInput(sequence, msa=MSA_ID)
    if model_id.startswith("alphaflow-"):
        return AlphaFlowInput(sequence, msa=MSA_ID, reference_structure=reference)
    return ESMFlowInput(sequence, reference_structure=reference)


@pytest.mark.parametrize("model_id", IDS)
def test_accessor_and_get_model_share_an_ensemble_handle(models_session, model_id):
    models = ModelsAPI(models_session.session)
    handle = getattr(models, model_id.replace("-", "_"))
    assert handle is models.get_model(model_id)
    assert type(handle) is HANDLES[model_id]
    assert isinstance(handle, StructureEnsemblePredictionModel)


@pytest.mark.parametrize("handle", sorted(set(HANDLES.values()), key=str))
def test_ensemble_handles_are_not_structure_prediction_handles(handle):
    assert not issubclass(handle, StructurePredictionModel)


@pytest.mark.parametrize(
    ("handle", "input_type"),
    [
        (BioEmu, BioEmuInput),
        (AlphaFlow, AlphaFlowInput),
        (ESMFlow, ESMFlowInput),
    ],
)
def test_each_handle_names_its_input_type(handle, input_type):
    assert handle.Input is input_type


@pytest.mark.parametrize("model_id", IDS)
def test_unset_controls_send_the_card_defaults(models_session, model_id):
    future = _handle(models_session, model_id).sample([_input(model_id)])
    assert type(future) is StructurePredictionFuture
    path, body = models_session.submitted
    assert path == f"v1/models/{model_id}/batch/structures"
    expected = CARD_DEFAULTS[HANDLES[model_id]]
    assert {name: body[name] for name in expected} == expected


def test_bioemu_sends_one_protein_per_input(models_session):
    ModelsAPI(models_session.session).bioemu_v1_1.sample(
        [
            BioEmuInput("MKT", msa=None),
            BioEmuInput(b"ACDE", msa=Protein.single_sequence_mode),
            BioEmuInput(Protein(sequence="GG"), msa=None),
        ]
    )
    assert models_session.submitted[1]["complexes"] == [
        [{"protein": {"id": "A", "sequence": sequence, "msa_id": None}}]
        for sequence in ("MKT", "ACDE", "GG")
    ]


@pytest.mark.parametrize("model_id", ["alphaflow-md-base-v1", "bioemu-v1-1"])
def test_aligned_inputs_send_their_msa(models_session, model_id):
    _handle(models_session, model_id).sample([_input(model_id), _input(model_id)])
    proteins = [c[0]["protein"] for c in models_session.submitted[1]["complexes"]]
    assert [protein["msa_id"] for protein in proteins] == [MSA_ID, MSA_ID]


@pytest.mark.parametrize("msa", [None, Protein.single_sequence_mode])
def test_alphaflow_refuses_an_input_without_an_msa(models_session, msa):
    with pytest.raises(ValueError, match=r"inputs\[0\]\.msa is required"):
        ModelsAPI(models_session.session).alphaflow_md_base_v1.sample(
            [AlphaFlowInput("MKT", msa=msa)]
        )
    models_session.session.post.assert_not_called()


def test_esmflow_sends_no_msa(models_session):
    ModelsAPI(models_session.session).esmflow_md_base_v1.sample([ESMFlowInput("MKT")])
    assert models_session.submitted[1]["complexes"] == [
        [{"protein": {"id": "A", "sequence": "MKT"}}]
    ]


@pytest.mark.parametrize("model_id", ["bioemu-v1-1", *FLOW_IDS])
@pytest.mark.parametrize(
    ("sequence", "error", "message"),
    [
        ("MKT:ACDE", ValueError, "several chains"),
        ("", ValueError, "is empty"),
        (42, TypeError, "Protein or sequence text"),
    ],
)
def test_sequence_inputs_take_one_protein_chain(
    models_session, model_id, sequence, error, message
):
    with pytest.raises(error, match=message):
        _handle(models_session, model_id).sample([_input(model_id, sequence)])
    models_session.session.post.assert_not_called()


@pytest.mark.parametrize("model_id", IDS)
@pytest.mark.parametrize(
    ("inputs", "error", "message"),
    [
        ("MKT", TypeError, "must be a list"),
        ([], ValueError, "at least one input"),
        (["MKT"], TypeError, r"inputs\[0\] must be a"),
    ],
)
def test_inputs_are_a_list_of_the_handles_input_type(
    models_session, model_id, inputs, error, message
):
    with pytest.raises(error, match=message):
        _handle(models_session, model_id).sample(inputs)
    models_session.session.post.assert_not_called()


def test_an_input_of_another_model_is_refused(models_session):
    with pytest.raises(TypeError, match=r"inputs\[0\] must be a ESMFlowInput"):
        ModelsAPI(models_session.session).esmflow_md_base_v1.sample(
            [BioEmuInput("MKT", msa=None)]  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("model_id", FLOW_IDS)
def test_flow_without_references_sends_none(models_session, model_id):
    _handle(models_session, model_id).sample([_input(model_id)])
    assert "reference_structures" not in models_session.submitted[1]


@pytest.mark.parametrize("model_id", FLOW_IDS)
def test_flow_sends_every_reference_in_input_order(
    models_session, ensemble_reference, model_id
):
    shorter = ensemble_reference[:50]
    _handle(models_session, model_id).sample(
        [
            _input(model_id, ensemble_reference.sequence.decode(), ensemble_reference),
            _input(model_id, shorter.sequence.decode(), shorter),
        ]
    )
    references = models_session.submitted[1]["reference_structures"]
    residues = [
        len(gemmi.read_pdb_string(record["structure_file"])[0][0])
        for record in references
    ]
    assert residues == [len(ensemble_reference), 50]


@pytest.mark.parametrize("model_id", FLOW_IDS)
def test_flow_references_are_given_for_every_input_or_none(
    models_session, ensemble_reference, model_id
):
    sequence = ensemble_reference.sequence.decode()
    with pytest.raises(ValueError, match=r"every input.*inputs\[1\] has none"):
        _handle(models_session, model_id).sample(
            [
                _input(model_id, sequence, ensemble_reference),
                _input(model_id, sequence),
            ]
        )
    models_session.session.post.assert_not_called()


@pytest.mark.parametrize("seed", [None, 0, 7])
def test_explicit_seed_value_is_sent(models_session, seed):
    ModelsAPI(models_session.session).bioemu_v1_1.sample(
        [BioEmuInput("MKT", msa=None)], random_seed=seed
    )
    assert models_session.submitted[1]["random_seed"] == seed


@pytest.mark.parametrize(
    ("model_id", "name", "value"),
    [
        ("bioemu-v1-1", "num_samples", 0),
        ("alphaflow-md-base-v1", "num_steps", 101),
        ("esmflow-md-base-v1", "tmax", 0),
        ("esmflow-md-distilled-v1", "num_samples", 251),
    ],
)
def test_out_of_range_controls_fail_the_card_schema(
    models_session, model_id, name, value
):
    with pytest.raises(ValidationError) as error:
        _handle(models_session, model_id).sample([_input(model_id)], **{name: value})
    assert list(error.value.absolute_path) == [name]


def test_a_card_refusal_reaches_the_caller_without_a_submission(models_session):
    refused = HTTPError(MagicMock(status_code=404, text="not found"))
    models_session.session.get.side_effect = refused
    with pytest.raises(HTTPError):
        ModelsAPI(models_session.session).bioemu_v1_1.sample(
            [BioEmuInput("MKT", msa=None)]
        )
    models_session.session.post.assert_not_called()


def test_a_submission_refusal_reaches_the_caller_once(models_session):
    accepted = models_session._post

    def post(path, **kwargs):
        accepted(path, **kwargs)
        raise HTTPError(MagicMock(status_code=422, text="unsupported reference"))

    models_session.session.post.side_effect = post
    with pytest.raises(HTTPError):
        ModelsAPI(models_session.session).esmflow_md_base_v1.sample(
            [ESMFlowInput("MKT")]
        )
    assert [path for path, _ in models_session.submissions] == [
        "v1/models/esmflow-md-base-v1/batch/structures"
    ]


def test_a_425_resubmit_sends_the_same_body_and_headers(models_session, monkeypatch):
    monkeypatch.setattr("openprotein.jobs._submission.time.sleep", lambda _: None)
    pending = MagicMock(status_code=425, headers={"Retry-After": "0"})
    accepted = models_session._post
    answers = iter([HTTPError(pending)])

    def post(path, **kwargs):
        error = next(answers, None)
        accepted(path, **kwargs)
        if error is not None:
            raise error
        return accepted(path, **kwargs)

    models_session.session.post.side_effect = post
    ModelsAPI(models_session.session).bioemu_v1_1.sample([BioEmuInput("MKT", msa=None)])

    first, second = models_session.session.post.call_args_list
    assert first == second
