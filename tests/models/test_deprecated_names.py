import importlib

import pytest

import openprotein.models.models as models_module
from openprotein import models
from openprotein.models import ModelsAPI

MOVED = [
    *(
        ("openprotein.fold", name)
        for name in (
            "AlphaFold2Model",
            "Boltz1Model",
            "Boltz1xModel",
            "Boltz2Model",
            "ESMFold2FastModel",
            "ESMFold2Model",
            "ESMFoldModel",
            "MiniFoldModel",
            "ProtenixModel",
            "ProtenixV2Model",
            "RosettaFold3Model",
        )
    ),
    *(
        ("openprotein.embeddings", name)
        for name in ("ESMModel", "OpenProteinModel", "PoET2Model", "PoETModel")
    ),
]

RENAMED = [
    ("BoltzGenModel", "BoltzGen"),
    ("ESMIF1Model", "ESMIF1"),
    ("ProteinMPNNModel", "ProteinMPNN"),
    ("RFdiffusionModel", "RFdiffusion"),
]


def _message(name: str, use: str) -> str:
    return (
        f"`{name}` is deprecated since 0.18.0 and will be removed in 0.19.0. Use {use}."
    )


@pytest.mark.parametrize(("module", "name"), MOVED)
def test_moved_model_names_resolve_to_their_new_module_and_warn(module, name):
    with pytest.warns(FutureWarning) as record:
        resolved = getattr(models, name)
    assert resolved is getattr(importlib.import_module(module), name)
    assert [str(w.message) for w in record] == [
        _message(f"openprotein.models.{name}", f"`{module}.{name}`")
    ]
    assert record[0].filename == __file__


@pytest.mark.parametrize(("old", "new"), RENAMED)
def test_renamed_model_names_resolve_to_the_new_name_and_warn(old, new):
    with pytest.warns(FutureWarning) as record:
        resolved = getattr(models, old)
    assert resolved is getattr(models, new)
    assert [str(w.message) for w in record] == [
        _message(f"openprotein.models.{old}", f"`openprotein.models.{new}`")
    ]


@pytest.mark.parametrize(("old", "new"), RENAMED)
def test_renamed_names_on_the_models_module_resolve_and_warn(old, new):
    with pytest.warns(FutureWarning) as record:
        resolved = getattr(models_module, old)
    assert resolved is getattr(models, new)
    assert [str(w.message) for w in record] == [
        _message(f"openprotein.models.models.{old}", f"`openprotein.models.{new}`")
    ]


def test_from_import_of_a_deprecated_name_warns():
    with pytest.warns(FutureWarning, match="openprotein.models.ESMModel"):
        from openprotein.models import ESMModel  # noqa: F401


@pytest.mark.parametrize("module", [models, models_module])
def test_unknown_names_raise_attribute_error(module):
    with pytest.raises(AttributeError):
        module.__getattr__("NotAModel")


@pytest.mark.parametrize(
    ("module", "path"),
    [(models, "openprotein.models"), (models_module, "openprotein.models.models")],
)
def test_soluble_mpnn_model_builds_a_solublempnn_proteinmpnn(
    module, path, mock_session
):
    with pytest.warns(FutureWarning) as record:
        resolved = getattr(module, "SolubleMPNNModel")
    assert [str(w.message) for w in record] == [
        _message(f"{path}.SolubleMPNNModel", "`session.models.solublempnn`")
    ]
    handle = resolved(mock_session)
    assert isinstance(handle, models.ProteinMPNN)
    assert handle.id == "solublempnn"


def test_deprecated_names_are_not_in_all():
    deprecated = (
        {name for _, name in MOVED} | {old for old, _ in RENAMED} | {"SolubleMPNNModel"}
    )
    assert deprecated.isdisjoint(models.__all__)


QUERY_UUID = "c0ffee00-0000-4000-8000-000000000001"


@pytest.mark.parametrize("model", ["esmif1", "proteinmpnn", "solublempnn"])
def test_inverse_folding_generate_takes_seed_as_random_seed(models_session, model):
    handle = getattr(ModelsAPI(models_session.session), model)
    with pytest.warns(
        FutureWarning, match=r"generate\(seed=\.\.\.\).*0\.18\.0.*0\.19\.0"
    ):
        handle.generate(query=QUERY_UUID, seed=7)
    assert models_session.submitted[1]["random_seed"] == 7
    assert "seed" not in models_session.submitted[1]


@pytest.mark.parametrize("model", ["esmif1", "proteinmpnn"])
def test_inverse_folding_single_site_takes_sequence_as_base_sequence(
    models_session, model
):
    handle = getattr(ModelsAPI(models_session.session), model)
    with pytest.warns(FutureWarning, match=r"single_site\(sequence=\.\.\.\)"):
        handle.single_site(sequence="MKT", query=QUERY_UUID)
    assert models_session.submitted[1]["base_sequence"] == "MKT"


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.esmif1.generate(query=QUERY_UUID, seed=1, random_seed=2),
        lambda m: m.esmif1.single_site("MKT", sequence="MKT", query=QUERY_UUID),
    ],
    ids=["generate", "single_site"],
)
def test_inverse_folding_refuses_both_keyword_forms(models_session, call):
    with pytest.warns(FutureWarning), pytest.raises(TypeError, match="not both"):
        call(ModelsAPI(models_session.session))
    models_session.session.post.assert_not_called()
