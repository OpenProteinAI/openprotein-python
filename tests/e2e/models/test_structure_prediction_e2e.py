"""Structure prediction through session.models."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.errors import HTTPError
from openprotein.jobs.futures import StructurePredictionFuture
from openprotein.models import StructurePredictionModel
from openprotein.molecules import Complex, Ligand, Protein, Structure, Template
from tests.e2e.config import scaled_timeout
from tests.e2e.conftest import BASE_SEQUENCE, UBIQUITIN_SEQUENCE

TIMEOUT = scaled_timeout(3.0)
NUM_SAMPLES = 2
SEED = 7
ATP_ATOMS = 31
FOLD_IDS = [
    "alphafold2",
    "boltz-1",
    "boltz-1x",
    "boltz-2",
    "esmfold",
    "esmfold2",
    "esmfold2-fast",
    "minifold",
    "protenix",
    "protenix-v2",
    "rosettafold-3",
]
MULTICHAIN_IDS = [model_id for model_id in FOLD_IDS if model_id != "minifold"]
SINGLE_SEQUENCE_IDS = ["esmfold", "esmfold2-fast", "minifold"]
PROTEIN_ONLY_IDS = {"alphafold2", "esmfold"}
PER_ATOM_PLDDT_IDS = {"protenix", "protenix-v2"}
TEMPLATE_IDS = ["boltz-2", "protenix", "protenix-v2"]
OUTPUTS = {
    "structures",
    "plddt",
    "pae",
    "pde",
    "confidence",
    "ipae",
    "ptm",
    "affinity",
    "score",
    "metrics",
}
POCKET = {
    "pocket": {"binder": "L", "contacts": [["A", 15], ["B", 22]], "max_distance": 6.0}
}
CONTACT = {"contact": {"token1": ["A", 10], "token2": ["B", 20], "max_distance": 15.0}}
CONSTRAINTS = {
    "boltz-1": {"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 20, "CA"]}},
    "boltz-1x": POCKET,
    "boltz-2": CONTACT,
}


@pytest.fixture(scope="module")
def msa(session: OpenProtein) -> MSAFuture:
    msa = session.align.create_msa(seed=BASE_SEQUENCE.encode())
    msa.wait_until_done(timeout=TIMEOUT)
    return msa


@pytest.fixture(scope="module")
def paired_msa(session: OpenProtein) -> MSAFuture:
    msa = session.align.create_msa(
        seed=f"{BASE_SEQUENCE}:{UBIQUITIN_SEQUENCE}".encode()
    )
    msa.wait_until_done(timeout=TIMEOUT)
    return msa


@pytest.fixture(scope="module")
def template() -> Protein:
    protein = Protein.from_filepath("tests/data/8bo9.cif", chain_id="A", verbose=False)
    return protein[len("MHHHHHHSDN") :]


def _model(session: OpenProtein, model_id: str) -> StructurePredictionModel:
    model = session.models.get_model(model_id)
    assert isinstance(model, StructurePredictionModel)
    return model


def _protein(sequence: str, msa: MSAFuture | None, model_id: str) -> Protein:
    protein = Protein(sequence=sequence)
    if model_id not in SINGLE_SEQUENCE_IDS:
        protein.msa = Protein.single_sequence_mode if msa is None else msa
    return protein


def _complex(model_id: str, msa: MSAFuture) -> Complex:
    return Complex(
        {
            "A": _protein(BASE_SEQUENCE, msa, model_id),
            "B": _protein(UBIQUITIN_SEQUENCE, msa, model_id),
            "L": Ligand(ccd="ATP"),
        }
    )


def _sampling(model: StructurePredictionModel) -> tuple[dict, int]:
    properties = model._params_schema("batch/structures")["properties"]
    params = {"random_seed": SEED} if "random_seed" in properties else {}
    for name in ("num_samples", "num_models"):
        if name in properties:
            return {**params, name: NUM_SAMPLES}, NUM_SAMPLES
    return params, 1


def _assert_structure(
    structure: Structure, chains: dict[str, str | None], samples: int
) -> None:
    """``chains`` maps each chain id to its protein sequence, or None for a ligand."""
    assert isinstance(structure, Structure)
    assert len(structure) == samples
    for conformation in structure:
        found = conformation.get_chains()
        assert set(found) == set(chains)
        for chain_id, sequence in chains.items():
            chain = found[chain_id]
            if sequence is None:
                assert isinstance(chain, Ligand)
                continue
            assert isinstance(chain, Protein)
            assert chain.sequence.decode() == sequence
            assert not chain.get_structure_mask().any()


def _read(future: StructurePredictionFuture, output: str, position: int) -> np.ndarray:
    return np.asarray(future.get_item(position, output=output), dtype=float)


def _assert_outputs(
    future: StructurePredictionFuture,
    model_id: str,
    samples: int,
    tokens: int,
    position: int = 0,
    has_ligand: bool = False,
    protein_chains: int = 1,
) -> None:
    outputs = set(future.outputs)
    assert "structures" in outputs
    assert outputs <= OUTPUTS
    if "plddt" in outputs:
        plddt = _read(future, "plddt", position).reshape(samples, -1)
        if model_id in PER_ATOM_PLDDT_IDS:
            assert plddt.shape[1] > tokens
        else:
            assert plddt.shape[1] == tokens
        assert ((plddt >= 0) & (plddt <= 100)).all()
    for name in outputs & {"pae", "pde"}:
        errors = _read(future, name, position)
        assert errors.shape[-2:] == (tokens, tokens)
        assert errors.size == samples * tokens * tokens
        assert (errors >= 0).all()
    if "confidence" in outputs:
        confidence = future.get_item(position, output="confidence")
        assert len(confidence) == samples
        assert all(0 <= sample["ptm"] <= 1 for sample in confidence)
    if "ptm" in outputs:
        ptm = _read(future, "ptm", position)
        assert ptm.shape == (samples,)
        assert ((ptm >= 0) & (ptm <= 1)).all()
    # test_ipae_reads_for_a_complex_with_a_ligand covers ligand complexes;
    # a single protein chain has no interface to score
    if "ipae" in outputs and not has_ligand and protein_chains >= 2:
        assert _read(future, "ipae", position).shape == (samples,)
    if "score" in outputs:
        score = future.get_item(position, output="score")
        assert isinstance(score, pd.DataFrame)
        assert set(score["batch_idx"]) == set(range(samples))
    if "metrics" in outputs:
        metrics = future.get_item(position, output="metrics")
        assert isinstance(metrics, pd.DataFrame)
        assert len(metrics) == 1
        assert {f"ptm.ptm_{i}" for i in range(samples)} <= set(metrics.columns)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", FOLD_IDS)
def test_folds_a_protein(session: OpenProtein, msa: MSAFuture, model_id: str):
    model = _model(session, model_id)
    params, samples = _sampling(model)

    future = model.fold(_protein(BASE_SEQUENCE, msa, model_id), **params)

    structures = future.wait(timeout=TIMEOUT)
    assert len(structures) == 1
    _assert_structure(structures[0], {"A": BASE_SEQUENCE}, samples)
    _assert_outputs(future, model_id, samples, len(BASE_SEQUENCE))


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", MULTICHAIN_IDS)
def test_folds_the_chains_of_a_complex_its_card_takes(
    session: OpenProtein, paired_msa: MSAFuture, model_id: str
):
    model = _model(session, model_id)
    params, samples = _sampling(model)
    complex = _complex(model_id, paired_msa)
    chains: dict[str, str | None] = {"A": BASE_SEQUENCE, "B": UBIQUITIN_SEQUENCE}
    tokens = len(BASE_SEQUENCE) + len(UBIQUITIN_SEQUENCE)

    if model_id in PROTEIN_ONLY_IDS:
        with pytest.warns(UserWarning, match="left out"):
            future = model.fold(complex, **params)
    else:
        future = model.fold(complex, **params)
        chains["L"] = None
        tokens += ATP_ATOMS

    (structure,) = future.wait(timeout=TIMEOUT)
    _assert_structure(structure, chains, samples)
    _assert_outputs(
        future,
        model_id,
        samples,
        tokens,
        has_ligand="L" in chains,
        protein_chains=sum(sequence is not None for sequence in chains.values()),
    )


@pytest.mark.e2e
@pytest.mark.xfail(
    strict=True,
    raises=HTTPError,
    reason="ipae is not served for complexes with a ligand",
)
def test_ipae_reads_for_a_complex_with_a_ligand(
    session: OpenProtein, paired_msa: MSAFuture
):
    model = _model(session, "boltz-2")
    params, samples = _sampling(model)

    future = model.fold(_complex("boltz-2", paired_msa), **params)

    future.wait_until_done(timeout=TIMEOUT)
    assert _read(future, "ipae", 0).shape == (samples,)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ["esmfold", "boltz-2"])
def test_a_batch_reads_back_in_input_order(session: OpenProtein, model_id: str):
    model = _model(session, model_id)
    params, samples = _sampling(model)
    sequences = [UBIQUITIN_SEQUENCE, BASE_SEQUENCE]

    future = model.fold(
        [_protein(sequence, None, model_id) for sequence in sequences], **params
    )

    structures = future.wait(timeout=TIMEOUT)
    for read in (
        structures,
        list(future.stream()),
        [future.get_item(i) for i in range(len(sequences))],
        [future.unit(i).get() for i in range(len(sequences))],
    ):
        assert len(read) == len(sequences)
        for structure, sequence in zip(read, sequences):
            _assert_structure(structure, {"A": sequence}, samples)
    for position, sequence in enumerate(sequences):
        _assert_outputs(future, model_id, samples, len(sequence), position=position)


@pytest.mark.e2e
def test_an_msa_folds_its_seed_as_one_complex(
    session: OpenProtein, paired_msa: MSAFuture
):
    future = session.models.alphafold2.fold(paired_msa)

    (structure,) = future.wait(timeout=TIMEOUT)
    _assert_structure(structure, {"A": BASE_SEQUENCE, "B": UBIQUITIN_SEQUENCE}, 1)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", TEMPLATE_IDS)
def test_folds_with_a_template(session: OpenProtein, template: Protein, model_id: str):
    sequence = template.sequence.decode()
    target = Complex({"A": _protein(sequence, None, model_id)})

    future = _model(session, model_id).fold(
        target,
        templates=[Template(template=template, mapping="A")],
        random_seed=SEED,
    )

    (structure,) = future.wait(timeout=TIMEOUT)
    _assert_structure(structure, {"A": sequence}, 1)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_id",
    [
        "boltz-1",
        "boltz-1x",
        "boltz-2",
    ],
)
def test_boltz_folds_under_a_constraint(
    session: OpenProtein, paired_msa: MSAFuture, model_id: str
):
    future = _model(session, model_id).fold(
        _complex(model_id, paired_msa),
        constraints=[CONSTRAINTS[model_id]],
        random_seed=SEED,
    )

    (structure,) = future.wait(timeout=TIMEOUT)
    _assert_structure(
        structure, {"A": BASE_SEQUENCE, "B": UBIQUITIN_SEQUENCE, "L": None}, 1
    )


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ["boltz-1", "boltz-1x"])
def test_boltz_1_refuses_a_contact_constraint(
    session: OpenProtein, paired_msa: MSAFuture, model_id: str
):
    with pytest.raises(HTTPError) as caught:
        _model(session, model_id).fold(
            _complex(model_id, paired_msa), constraints=[CONTACT]
        )

    assert caught.value.status_code == 422
    assert "contact" in caught.value.text


@pytest.mark.e2e
def test_boltz_2_predicts_the_affinity_of_a_binder(
    session: OpenProtein, msa: MSAFuture
):
    complex = Complex(
        {"A": _protein(BASE_SEQUENCE, msa, "boltz-2"), "L": Ligand(smiles="CCO")}
    )

    future = session.models.boltz_2.fold(
        complex, properties=[{"affinity": {"binder": "L"}}], random_seed=SEED
    )

    future.wait_until_done(timeout=TIMEOUT)
    affinity = future.get_item(0, output="affinity")
    assert np.isfinite(affinity["affinity_pred_value"])
    assert 0 <= affinity["affinity_probability_binary"] <= 1


@pytest.mark.e2e
def test_a_fold_reloads_from_its_job_id(session: OpenProtein):
    future = session.models.esmfold.fold(BASE_SEQUENCE)
    (structure,) = future.wait(timeout=TIMEOUT)

    reloaded = session.jobs.get(future.id)

    assert isinstance(reloaded, StructurePredictionFuture)
    assert reloaded.outputs == future.outputs
    (again,) = reloaded.get()
    np.testing.assert_array_equal(
        again[0].get_proteins()["A"].coordinates,
        structure[0].get_proteins()["A"].coordinates,
    )
    np.testing.assert_array_equal(_read(reloaded, "pae", 0), _read(future, "pae", 0))


@pytest.mark.e2e
def test_minifold_refuses_a_complex(session: OpenProtein):
    complex = Complex({"A": Protein(sequence=BASE_SEQUENCE), "B": Protein("MKT")})

    with pytest.raises(ValueError, match="only supports monomers"):
        session.models.minifold.fold(complex)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", SINGLE_SEQUENCE_IDS)
def test_a_single_sequence_model_refuses_an_msa(
    session: OpenProtein, msa: MSAFuture, model_id: str
):
    protein = Protein(sequence=BASE_SEQUENCE)
    protein.msa = msa

    with pytest.raises(ValueError, match="single-sequence model"):
        _model(session, model_id).fold(protein)


@pytest.mark.e2e
@pytest.mark.parametrize(
    "model_id", [model_id for model_id in FOLD_IDS if model_id not in TEMPLATE_IDS]
)
def test_a_model_without_templates_refuses_them(
    session: OpenProtein, template: Protein, model_id: str
):
    with pytest.raises(ValueError, match="takes no templates"):
        _model(session, model_id).fold(BASE_SEQUENCE, templates=[template])


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ["esmfold", "minifold", "boltz-2"])
def test_a_sequence_over_the_card_maximum_is_refused(
    session: OpenProtein, model_id: str
):
    model = _model(session, model_id)
    max_length = model.metadata.max_sequence_length
    assert max_length is not None

    with pytest.raises(HTTPError) as caught:
        model.fold(_protein("A" * (max_length + 1), None, model_id))

    assert caught.value.status_code == 400
    assert str(max_length) in caught.value.text


@pytest.mark.e2e
@pytest.mark.parametrize(
    ("model_id", "name", "value"),
    [
        ("boltz-2", "num_samples", 21),
        ("rosettafold-3", "num_samples", 21),
        ("alphafold2", "num_models", 6),
    ],
)
def test_an_out_of_range_param_is_refused(
    session: OpenProtein, model_id: str, name: str, value: int
):
    params: dict[str, Any] = {name: value}

    with pytest.raises(HTTPError) as caught:
        _model(session, model_id).fold(
            _protein(BASE_SEQUENCE, None, model_id), **params
        )

    assert caught.value.status_code == 422
    assert name in caught.value.text


@pytest.mark.e2e
def test_a_template_index_past_the_batch_is_refused(
    session: OpenProtein, template: Protein
):
    target = Complex({"A": _protein(template.sequence.decode(), None, "boltz-2")})

    with pytest.raises(HTTPError) as caught:
        session.models.boltz_2.fold(
            target, templates=[Template(template=template, mapping="A", index=[0, 5])]
        )

    assert caught.value.status_code == 400
    assert "out of range" in caught.value.text
