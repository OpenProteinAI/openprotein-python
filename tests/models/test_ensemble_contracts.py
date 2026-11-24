import pytest

from tests.models.support import PARAMS, card

ENSEMBLE_IDS = (
    "bioemu-v1-1",
    "alphaflow-md-base-v1",
    "alphaflow-md-distilled-v1",
    "esmflow-md-base-v1",
    "esmflow-md-distilled-v1",
)
FLOW_IDS = ENSEMBLE_IDS[1:5]


@pytest.mark.parametrize("model_id", ENSEMBLE_IDS)
def test_ensemble_cards_declare_structure_prediction(model_id):
    metadata = card(model_id)
    assert "ensemble" in metadata["tags"]
    assert metadata["input_types"] == ["protein"]
    assert metadata["multichain"] is False
    method = PARAMS["methods"][f"{model_id}/batch/structures"]
    assert method["outputs"] == (
        ["structures", "plddt", "confidence"]
        if model_id in FLOW_IDS
        else ["structures"]
    )


@pytest.mark.parametrize("model_id", FLOW_IDS)
def test_distilled_flow_cards_are_a_variant_of_their_base(model_id):
    metadata = card(model_id)
    distilled = "distilled" in model_id
    assert metadata["variant"] == ("distilled" if distilled else None)
    assert metadata["base"] is not distilled


@pytest.mark.parametrize("model_id", ENSEMBLE_IDS)
def test_ensemble_params_use_public_names(model_id):
    properties = PARAMS["methods"][f"{model_id}/batch/structures"]["params"][
        "properties"
    ]
    assert "num_samples" in properties
    assert "random_seed" in properties
    assert not {"steps", "seed", "batch_size", "batch_size_100"} & properties.keys()
    assert "complexes" in properties


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "bioemu-v1-1",
            {
                "num_samples": 1,
                "denoiser": "dpm",
                "filter_samples": True,
                "random_seed": None,
            },
        ),
        *[
            (
                model_id,
                {
                    "num_samples": 1,
                    "num_steps": 10,
                    "tmax": 1.0,
                    "self_cond": None,
                    "random_seed": None,
                },
            )
            for model_id in FLOW_IDS
        ],
    ],
)
def test_ensemble_defaults_match_the_card(model_id, expected):
    properties = PARAMS["methods"][f"{model_id}/batch/structures"]["params"][
        "properties"
    ]
    assert {name: properties[name]["default"] for name in expected} == expected
