"""Structure prediction handles shape inputs into complexes and run each model's checks."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openprotein.align import AlignAPI, MSAFuture
from openprotein.data.msa import MSA, MSAMetadata
from openprotein.models import ModelsAPI, OpenDDE, StructurePredictionModel
from openprotein.molecules import DNA, Complex, Ligand, Protein, Template
from tests.models.support import PARAMS, card, created_query

FOLD_IDS = [
    "esmfold",
    "minifold",
    "alphafold2",
    "rosettafold-3",
    "boltz-1",
    "boltz-1x",
    "boltz-2",
    "protenix",
    "protenix-v2",
    "esmfold2",
    "esmfold2-fast",
    "opendde",
    "opendde-abag",
]
PROTEIN = {"protein": {"id": "A", "sequence": "MKT"}}
PROTEIN_NULL_MSA = {"protein": {"id": "A", "sequence": "MKT", "msa_id": None}}


@pytest.fixture
def models(models_session) -> ModelsAPI:
    return ModelsAPI(models_session.session)


def _ligand_complex() -> Complex:
    return Complex({"A": Protein(sequence="MKT"), "L": Ligand(ccd="ATP")})


@pytest.mark.parametrize("alias", ["af2", "rf3", "boltz1", "boltz1x", "boltz2"])
def test_short_aliases_resolve_to_fold_handles(models, alias):
    handle = getattr(models, alias)

    assert isinstance(handle, StructurePredictionModel)
    assert handle.id in FOLD_IDS


def test_every_fold_id_resolves_to_a_structure_prediction_handle(models):
    for model_id in FOLD_IDS:
        assert isinstance(models.get_model(model_id), StructurePredictionModel)


@pytest.mark.parametrize(
    "model_id, chain",
    [
        ("esmfold", PROTEIN),
        ("minifold", PROTEIN),
        ("esmfold2-fast", PROTEIN),
        ("alphafold2", PROTEIN_NULL_MSA),
        ("boltz-2", PROTEIN_NULL_MSA),
        ("esmfold2", PROTEIN_NULL_MSA),
    ],
)
def test_a_sequence_folds_as_one_complex(models, models_session, model_id, chain):
    models.get_model(model_id).fold("MKT")

    path, body = models_session.submitted
    assert path == f"v1/models/{model_id}/batch/structures"
    assert body["complexes"] == [[chain]]


def test_a_later_complex_with_a_chain_the_first_lacks_folds(models, models_session):
    models.boltz_2.fold(["MKT", "MKT:AC"])

    _, body = models_session.submitted
    assert [len(complex) for complex in body["complexes"]] == [1, 2]


def test_folding_an_msa_future_reads_its_seed_once(models, models_session, monkeypatch):
    align = AlignAPI(models_session.session)
    get_seed = MagicMock(return_value="MKT:AC")
    monkeypatch.setattr(align, "get_seed", get_seed)
    models_session.session.align = align
    msa = MagicMock(spec=MSAFuture)
    msa.job = MagicMock(job_id=MSA_ID)
    msa.id = MSA_ID

    models.boltz_2.fold(msa)

    get_seed.assert_called_once_with(job_id=MSA_ID)
    _, body = models_session.submitted
    assert len(body["complexes"][0]) == 2
    assert all(chain["protein"]["msa_id"] == MSA_ID for chain in body["complexes"][0])


def test_serialized_complexes_pass_through(models, models_session):
    complexes = [[PROTEIN]]

    models.esmfold.fold(complexes, num_recycles=2)

    assert models_session.submitted[1] == {"complexes": complexes, "num_recycles": 2}


def test_an_msa_future_folds_its_seed_as_one_complex(models, models_session):
    msa = MagicMock(spec=MSAFuture)
    complex = Complex({"A": Protein(sequence="MKT"), "B": Protein(sequence="AC")})
    serialized = [[PROTEIN_NULL_MSA]]

    with (
        patch(
            "openprotein.models.folding.base.msa_future_to_complex",
            return_value=complex,
        ) as to_complex,
        patch(
            "openprotein.models.folding.base.serialize_input",
            return_value=serialized,
        ) as serialize,
    ):
        models.esmfold2.fold(msa, random_seed=1)

    to_complex.assert_called_once_with(models_session.session, msa)
    serialize.assert_called_once_with(models_session.session, [complex], True)
    assert models_session.submitted[1]["complexes"] == serialized


@pytest.mark.parametrize("complexes", [3, [3], [{"protein": {}}], None])
def test_a_wrong_input_type_raises_before_any_request(
    models, models_session, complexes
):
    with pytest.raises(TypeError, match="complexes must be"):
        models.esmfold.fold(complexes)
    models_session.session.post.assert_not_called()
    models_session.session.get.assert_not_called()


def test_no_complexes_raise_before_any_request(models, models_session):
    with pytest.raises(ValueError, match="at least one complex"):
        models.esmfold.fold([])
    models_session.session.get.assert_not_called()


@pytest.mark.parametrize(
    "model_id, kept",
    [("esmfold", ["A"]), ("alphafold2", ["A"]), ("rosettafold-3", ["A", "L"])],
)
def test_unsupported_chains_are_left_out_with_a_warning(
    models, models_session, model_id, kept
):
    complex = Complex(
        {
            "A": Protein(sequence="MKT"),
            "D": DNA(sequence="ACGT"),
            "L": Ligand(ccd="ATP"),
        }
    )
    complex.get_protein("A").msa = Protein.single_sequence_mode

    with pytest.warns(UserWarning, match="left out"):
        models.get_model(model_id).fold(complex)

    chains = models_session.submitted[1]["complexes"][0]
    assert [next(iter(c.values()))["id"] for c in chains] == kept
    assert list(complex.get_chains()) == ["A", "D", "L"]


@pytest.mark.parametrize(
    "complex, match",
    [
        (Complex({"A": Protein(sequence="M"), "B": Protein(sequence="K")}), "monomers"),
        (_ligand_complex(), "only supports proteins"),
    ],
)
def test_minifold_folds_protein_monomers_only(models, models_session, complex, match):
    with pytest.raises(ValueError, match=match):
        models.minifold.fold(complex)
    models_session.session.post.assert_not_called()


def test_esmfold2_fast_refuses_an_msa(models, models_session):
    protein = Protein(sequence="MKT")
    protein.msa = "msa-1"

    with pytest.raises(ValueError, match="single-sequence"):
        models.esmfold2_fast.fold(protein)
    models_session.session.post.assert_not_called()


def _single_sequence_ligand_complex() -> Complex:
    complex = _ligand_complex()
    complex.get_protein("A").msa = Protein.single_sequence_mode
    return complex


@pytest.mark.parametrize("alias", ["boltz1", "boltz1x"])
def test_boltz_constraints_are_sent_as_given(models, models_session, alias):
    constraints = [
        {"contact": {"token1": ["A", 1], "token2": ["A", 2], "max_distance": 5}}
    ]

    getattr(models, alias).fold(
        _single_sequence_ligand_complex(), constraints=constraints
    )

    assert models_session.submitted[1]["constraints"] == constraints


def test_boltz2_properties_are_sent_as_given(models, models_session):
    properties = [{"affinity": {"binder": "A"}}]

    models.boltz2.fold(_single_sequence_ligand_complex(), properties=properties)

    assert models_session.submitted[1]["properties"] == properties


def test_boltz2_templates_become_template_queries(models, models_session):
    protein = Protein(sequence="ARNDCQ")
    protein.msa = Protein.single_sequence_mode
    template = Protein(sequence="ARNDCQ")
    template = template._set_coordinates(
        np.zeros(template.coordinates.shape, dtype=np.float32)
    )

    models.boltz2.fold(protein, templates=[template], random_seed=1)

    assert models_session.submitted[1]["templates"] == [{"query_id": created_query(1)}]
    assert [name for name, _ in models_session.queries] == ["query.cif"]


def test_shared_template_targets_are_checked_before_upload_or_submission(
    models, models_session
):
    template = Protein("ARNDCQ")
    template = template._set_coordinates(
        np.zeros(template.coordinates.shape, dtype=np.float32)
    )
    first = Protein("ARNDCQ").set_msa(Protein.single_sequence_mode)
    first.templates = [template]
    later = Protein("MKT").set_msa(Protein.single_sequence_mode)

    with pytest.raises(ValueError, match="Template maps.*does not exist"):
        models.boltz2.fold([Complex({"A": first}), Complex({"B": later})])

    assert models_session.queries == []
    models_session.session.post.assert_not_called()


def test_a_template_mapped_to_chain_a_folds_a_lone_protein(models, models_session):
    template = Protein("ARNDCQ")
    template = template._set_coordinates(
        np.zeros(template.coordinates.shape, dtype=np.float32)
    )
    protein = Protein("ARNDCQ").set_msa(Protein.single_sequence_mode)

    models.boltz2.fold(protein, templates=[Template(template, mapping="A")])

    assert models_session.submitted[1]["templates"] == [
        {"query_id": created_query(1), "chain_id": "A"}
    ]


@pytest.mark.parametrize("msa", [None, "not-a-uuid"])
def test_a_refused_fold_uploads_no_template(models, models_session, msa):
    template = Protein("ARNDCQ")
    template = template._set_coordinates(
        np.zeros(template.coordinates.shape, dtype=np.float32)
    )
    protein = Protein("ARNDCQ")
    protein.msa = msa

    with pytest.raises(ValueError):
        models.boltz2.fold(protein, templates=[template])

    assert models_session.queries == []
    models_session.session.post.assert_not_called()


def test_a_model_without_templates_refuses_them(models, models_session):
    with pytest.raises(ValueError, match="takes no templates"):
        models.esmfold.fold("MKT", templates=[Protein(sequence="MKT")])
    models_session.session.post.assert_not_called()


@pytest.mark.parametrize("model_id", FOLD_IDS)
def test_a_fold_sends_no_random_seed_unless_given(models, models_session, model_id):
    protein = Protein(sequence="MKT")
    protein.msa = Protein.single_sequence_mode

    models.get_model(model_id).fold(protein)

    assert "random_seed" not in models_session.submitted[1]


def test_a_fold_leaves_out_chains_its_card_does_not_take(models, models_session):
    models_session.get_routes["v1/models/rosettafold-3"] = card("rosettafold-3") | {
        "input_types": ["protein"]
    }

    with pytest.warns(UserWarning, match="left out"):
        models.rf3.fold(_single_sequence_ligand_complex())

    chains = models_session.submitted[1]["complexes"][0]
    assert [next(iter(c.values()))["id"] for c in chains] == ["A"]


def test_a_card_that_is_not_multichain_folds_one_protein(models, models_session):
    models_session.get_routes["v1/models/esmfold"] = card("esmfold") | {
        "multichain": False
    }
    complex = Complex({"A": Protein(sequence="M"), "B": Protein(sequence="K")})

    with pytest.raises(ValueError, match="monomers"):
        models.esmfold.fold(complex)
    models_session.session.post.assert_not_called()


def test_a_model_whose_params_name_no_msa_refuses_one(models, models_session):
    models_session.get_routes["v1/models/alphafold2/batch/structures/params"] = PARAMS[
        "methods"
    ]["esmfold/batch/structures"]["params"]
    protein = Protein(sequence="MKT")
    protein.msa = "msa-1"

    with pytest.raises(ValueError, match="single-sequence"):
        models.alphafold2.fold(protein)
    models_session.session.post.assert_not_called()

    models.alphafold2.fold("MKT")

    assert models_session.submitted[1]["complexes"] == [[PROTEIN]]


def test_a_model_whose_params_have_no_templates_refuses_them(models, models_session):
    models_session.get_routes["v1/models/boltz-2/batch/structures/params"] = PARAMS[
        "methods"
    ]["boltz-1/batch/structures"]["params"]

    with pytest.raises(ValueError, match="takes no templates"):
        models.boltz2.fold("MKT", templates=[Protein(sequence="MKT")])
    models_session.session.post.assert_not_called()


def test_a_handle_reads_its_card_and_params_once(models, models_session):
    models.esmfold.fold("MKT")
    models.esmfold.fold("MKV")

    reads = sorted(call.args[0] for call in models_session.session.get.call_args_list)
    assert reads == [
        "v1/models/esmfold",
        "v1/models/esmfold/batch/structures/params",
    ]


MSA_ID = "e0b1c3d4-0000-4000-8000-000000000001"
OPENDDE = [("opendde", "opendde"), ("opendde_abag", "opendde-abag")]


@pytest.mark.parametrize(("attr", "model_id"), OPENDDE)
def test_opendde_accessors_are_annotated_opendde_handles(attr, model_id):
    prop = vars(ModelsAPI)[attr]
    assert isinstance(prop, property)
    assert prop.fget.__annotations__["return"] in (OpenDDE, "OpenDDE")


@pytest.mark.parametrize(("attr", "model_id"), OPENDDE)
@pytest.mark.parametrize("kind", ["sequence", "ligand", "msa"])
def test_opendde_folds_send_bodies_its_params_accept(
    models, models_session, monkeypatch, attr, model_id, kind
):
    complexes = "MKT"
    if kind == "ligand":
        complexes = _ligand_complex()
        complexes.get_protein("A").msa = Protein.single_sequence_mode
    elif kind == "msa":
        align = AlignAPI(models_session.session)
        monkeypatch.setattr(align, "get_seed", lambda job_id: "MKT")
        models_session.session.align = align
        complexes = Protein(sequence="MKT")
        complexes.msa = MSA_ID

    getattr(models, attr).fold(complexes)

    path, body = models_session.submitted
    assert path == f"v1/models/{model_id}/batch/structures"
    if kind == "ligand":
        assert body["complexes"][0][1]["ligand"]["ccd"] == "ATP"
    elif kind == "msa":
        assert body["complexes"][0][0]["protein"]["msa_id"] == MSA_ID


def _stored_msa(session) -> MSA:
    now = datetime.now(timezone.utc)
    return MSA(session, MSAMetadata(msa_id=MSA_ID, created_at=now, updated_at=now))


@pytest.mark.parametrize("model_id", ["boltz-2", "protenix", "protenix-v2"])
def test_a_colon_string_folds_as_one_multichain_complex_on_a_template_model(
    models, models_session, model_id
):
    models.get_model(model_id).fold("MKT:AC")

    _, body = models_session.submitted
    assert [
        [(chain["protein"]["id"], chain["protein"]["sequence"]) for chain in complex]
        for complex in body["complexes"]
    ] == [[("A", "MKT"), ("B", "AC")]]


def test_an_attached_stored_msa_sends_its_id(models, models_session, monkeypatch):
    align = AlignAPI(models_session.session)
    monkeypatch.setattr(align, "get_seed", lambda job_id: "MKT")
    models_session.session.align = align
    protein = Protein(sequence="MKT")
    protein.msa = _stored_msa(models_session.session)

    models.boltz2.fold(protein)

    assert models_session.submitted[1]["complexes"] == [
        [{"protein": {"id": "A", "sequence": "MKT", "msa_id": MSA_ID}}]
    ]


def test_a_single_sequence_model_refuses_an_attached_stored_msa(models, models_session):
    protein = Protein(sequence="MKT")
    protein.msa = _stored_msa(models_session.session)

    with pytest.raises(ValueError, match="single-sequence"):
        models.esmfold.fold(protein)
    models_session.session.post.assert_not_called()


def test_an_msa_of_another_type_is_refused(models, models_session):
    protein = Protein(sequence="MKT")
    protein.msa = 42  # ty: ignore[invalid-assignment]

    with pytest.raises(TypeError, match="`.msa` must be"):
        models.boltz2.fold(protein)
    models_session.session.post.assert_not_called()


def test_a_complex_with_no_chain_the_model_folds_is_refused(models, models_session):
    with (
        pytest.warns(UserWarning, match="left out"),
        pytest.raises(ValueError, match="complex 0 has no chain esmfold can fold"),
    ):
        models.esmfold.fold(Complex({"L": Ligand(ccd="ATP")}))
    models_session.session.post.assert_not_called()
