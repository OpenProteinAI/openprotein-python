"""Static model accessors submit through the models API."""

from unittest.mock import MagicMock

import pytest

from openprotein.errors import HTTPError
from openprotein.models import (
    ESM1,
    ESM2,
    ESMC,
    ESMIF1,
    AbLang2,
    AlphaFlow,
    AlphaFold2,
    BioEmu,
    Boltz1,
    Boltz2,
    BoltzGen,
    ESMFlow,
    ESMFold,
    ESMFold2,
    GPModel,
    MiniFold,
    Model,
    ModelsAPI,
    OpenDDE,
    PoET,
    PoET2,
    ProteinMPNN,
    Protenix,
    ProtenixV2,
    ProtSeq,
    ProtT5,
    RFdiffusion,
    RosettaFold3,
    RotaProt,
    SVDModel,
    UMAPModel,
)

HANDLES: dict[str, type[Model]] = {
    "rfdiffusion": RFdiffusion,
    "proteinmpnn": ProteinMPNN,
    "solublempnn": ProteinMPNN,
    "boltzgen": BoltzGen,
    "esm-if1": ESMIF1,
    "poet-2": PoET2,
    "poet": PoET,
    "prot-seq": ProtSeq,
    "rotaprot-large-uniref50w": RotaProt,
    "rotaprot-large-uniref90-ft": RotaProt,
    "ablang2": AbLang2,
    "esm1b_t33_650M_UR50S": ESM1,
    **{f"esm1v_t33_650M_UR90S_{i}": ESM1 for i in range(1, 6)},
    "esm2_t12_35M_UR50D": ESM2,
    "esm2_t30_150M_UR50D": ESM2,
    "esm2_t33_650M_UR50D": ESM2,
    "esm2_t36_3B_UR50D": ESM2,
    "esm2_t6_8M_UR50D": ESM2,
    "esmc-300m": ESMC,
    "esmc-600m": ESMC,
    "esmc-6b": ESMC,
    "prott5-xl": ProtT5,
    "alphafold2": AlphaFold2,
    "boltz-1": Boltz1,
    "boltz-1x": Boltz1,
    "boltz-2": Boltz2,
    "esmfold": ESMFold,
    "esmfold2": ESMFold2,
    "esmfold2-fast": ESMFold2,
    "minifold": MiniFold,
    "opendde": OpenDDE,
    "opendde-abag": OpenDDE,
    "protenix": Protenix,
    "protenix-v2": ProtenixV2,
    "rosettafold-3": RosettaFold3,
    "bioemu-v1-1": BioEmu,
    "alphaflow-md-base-v1": AlphaFlow,
    "alphaflow-md-distilled-v1": AlphaFlow,
    "esmflow-md-base-v1": ESMFlow,
    "esmflow-md-distilled-v1": ESMFlow,
}

MODEL_ACCESSORS = [
    ("rfdiffusion", "rfdiffusion"),
    ("proteinmpnn", "proteinmpnn"),
    ("solublempnn", "solublempnn"),
    ("boltzgen", "boltzgen"),
    ("esmif1", "esm-if1"),
    ("poet2", "poet-2"),
    ("poet", "poet"),
    ("prot_seq", "prot-seq"),
    ("rotaprot_large_uniref50w", "rotaprot-large-uniref50w"),
    ("rotaprot_large_uniref90_ft", "rotaprot-large-uniref90-ft"),
    ("ablang2", "ablang2"),
    ("esm1b_t33_650M_UR50S", "esm1b_t33_650M_UR50S"),
    ("esm1v_t33_650M_UR90S_1", "esm1v_t33_650M_UR90S_1"),
    ("esm1v_t33_650M_UR90S_2", "esm1v_t33_650M_UR90S_2"),
    ("esm1v_t33_650M_UR90S_3", "esm1v_t33_650M_UR90S_3"),
    ("esm1v_t33_650M_UR90S_4", "esm1v_t33_650M_UR90S_4"),
    ("esm1v_t33_650M_UR90S_5", "esm1v_t33_650M_UR90S_5"),
    ("esm2_t12_35M_UR50D", "esm2_t12_35M_UR50D"),
    ("esm2_t30_150M_UR50D", "esm2_t30_150M_UR50D"),
    ("esm2_t33_650M_UR50D", "esm2_t33_650M_UR50D"),
    ("esm2_t36_3B_UR50D", "esm2_t36_3B_UR50D"),
    ("esm2_t6_8M_UR50D", "esm2_t6_8M_UR50D"),
    ("esmc_300m", "esmc-300m"),
    ("esmc_600m", "esmc-600m"),
    ("esmc_6b", "esmc-6b"),
    ("prott5_xl", "prott5-xl"),
    ("alphafold2", "alphafold2"),
    ("boltz_1", "boltz-1"),
    ("boltz_1x", "boltz-1x"),
    ("boltz_2", "boltz-2"),
    ("esmfold", "esmfold"),
    ("esmfold2", "esmfold2"),
    ("esmfold2_fast", "esmfold2-fast"),
    ("minifold", "minifold"),
    ("opendde", "opendde"),
    ("opendde_abag", "opendde-abag"),
    ("protenix", "protenix"),
    ("protenix_v2", "protenix-v2"),
    ("rosettafold_3", "rosettafold-3"),
    ("bioemu_v1_1", "bioemu-v1-1"),
    ("alphaflow_md_base_v1", "alphaflow-md-base-v1"),
    ("alphaflow_md_distilled_v1", "alphaflow-md-distilled-v1"),
    ("esmflow_md_base_v1", "esmflow-md-base-v1"),
    ("esmflow_md_distilled_v1", "esmflow-md-distilled-v1"),
]


@pytest.mark.parametrize("attr,model_id", MODEL_ACCESSORS)
def test_static_accessor_is_a_lazy_handle(mock_session, attr, model_id):
    api = ModelsAPI(mock_session)

    handle = getattr(api, attr)

    assert type(handle) is HANDLES[model_id]
    assert handle.id == model_id
    mock_session.get.assert_not_called()


def test_every_accessor_has_a_model_class():
    assert {model_id for _, model_id in MODEL_ACCESSORS} == HANDLES.keys()


@pytest.mark.parametrize("handle", sorted(set(HANDLES.values()), key=str))
def test_a_model_class_defaults_to_one_of_its_checkpoints(mock_session, handle):
    assert HANDLES[handle(mock_session).id] is handle


def test_a_model_without_a_default_needs_an_id(mock_session):
    with pytest.raises(TypeError, match="Model needs a model_id"):
        Model(mock_session)


ALIASES = {
    "poet_2": "poet2",
    "esm_if1": "esmif1",
    "esm1b": "esm1b_t33_650M_UR50S",
    "esm1v": "esm1v_t33_650M_UR90S_1",
    "esm2": "esm2_t33_650M_UR50D",
    "af2": "alphafold2",
    "rf3": "rosettafold_3",
    "boltz1": "boltz_1",
    "boltz1x": "boltz_1x",
    "boltz2": "boltz_2",
    "bioemu": "bioemu_v1_1",
    "alphaflow": "alphaflow_md_base_v1",
    "esmflow": "esmflow_md_base_v1",
}


def test_every_property_is_a_listed_accessor_or_alias():
    properties = {
        name for name, value in vars(ModelsAPI).items() if isinstance(value, property)
    }
    assert properties == {attr for attr, _ in MODEL_ACCESSORS} | ALIASES.keys()


def _not_found(session):
    session.get.side_effect = HTTPError(
        MagicMock(status_code=404, text="Model not found.")
    )


@pytest.mark.parametrize("attr,model_id", MODEL_ACCESSORS)
def test_get_model_returns_the_accessors_handle(models_session, attr, model_id):
    api = ModelsAPI(models_session.session)

    assert api.get_model(model_id) is getattr(api, attr)


def test_a_listed_id_without_an_accessor_is_a_plain_model(models_session):
    handle = ModelsAPI(models_session.session).get_model("other-model")

    assert (type(handle), handle.id) == (Model, "other-model")


@pytest.mark.parametrize("model_id", ["not-a-model", "esm2"])
def test_get_model_refuses_an_unlisted_id(mock_session, model_id):
    _not_found(mock_session)

    with pytest.raises(ValueError, match=f"model '{model_id}' not found"):
        ModelsAPI(mock_session).get_model(model_id)


def test_get_model_passes_on_other_card_errors(mock_session):
    mock_session.get.side_effect = HTTPError(MagicMock(status_code=500, text="down"))

    with pytest.raises(HTTPError):
        ModelsAPI(mock_session).get_model("esmfold")


@pytest.mark.parametrize(
    ("model_id", "handle"),
    [("gp/fit-1", GPModel), ("svd/fit-1", SVDModel), ("umap/fit-1", UMAPModel)],
)
def test_a_fitted_id_is_checked_on_first_use(mock_session, model_id, handle):
    _not_found(mock_session)

    assert type(ModelsAPI(mock_session).get_model(model_id)) is handle
    mock_session.get.assert_not_called()


def test_aliases_resolve_to_canonical_handles(mock_session):
    api = ModelsAPI(mock_session)

    for alias, canonical in ALIASES.items():
        assert getattr(api, alias) is getattr(api, canonical)
    assert "esm2" in dir(api)


def test_an_esm_id_without_an_accessor_is_a_plain_model(models_session):
    handle = ModelsAPI(models_session.session).get_model("esm2_t48_15B_UR50D")

    assert type(handle) is Model


def test_the_package_exports_no_embeddings_or_fold_classes():
    import openprotein.models

    modules = {
        getattr(openprotein.models, name).__module__
        for name in openprotein.models.__all__
    }
    assert not any(
        module.startswith(("openprotein.embeddings", "openprotein.fold"))
        for module in modules
    )
