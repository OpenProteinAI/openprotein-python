"""Structure ensemble sampling through session.models."""

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.align.msa import MSAFuture
from openprotein.jobs.futures import StructurePredictionFuture
from openprotein.models import AlphaFlow, ESMFlow
from openprotein.molecules import Protein, Structure
from tests.e2e.config import scaled_timeout
from tests.e2e.conftest import BASE_SEQUENCE, UBIQUITIN_SEQUENCE

TIMEOUT = scaled_timeout(3.0)
NUM_SAMPLES = 2
SEED = 7
ALPHAFLOW_IDS = ["alphaflow-md-base-v1", "alphaflow-md-distilled-v1"]
ESMFLOW_IDS = ["esmflow-md-base-v1", "esmflow-md-distilled-v1"]


@pytest.fixture(scope="module")
def msa(session: OpenProtein) -> MSAFuture:
    msa = session.align.create_msa(seed=BASE_SEQUENCE.encode())
    msa.wait_until_done(timeout=TIMEOUT)
    return msa


@pytest.fixture(scope="module")
def reference_fold(session: OpenProtein) -> StructurePredictionFuture:
    future = session.models.esmfold.fold(BASE_SEQUENCE)
    future.wait_until_done(timeout=TIMEOUT)
    return future


@pytest.fixture(scope="module")
def reference(reference_fold: StructurePredictionFuture) -> Protein:
    (structure,) = reference_fold.get()
    (protein,) = structure[0].get_proteins().values()
    return protein


def _assert_ensembles(
    structures: list[Structure], sequences: list[str], num_samples: int
) -> None:
    assert len(structures) == len(sequences)
    for structure, sequence in zip(structures, sequences):
        assert isinstance(structure, Structure)
        assert len(structure) == num_samples
        for conformation in structure:
            (protein,) = conformation.get_proteins().values()
            assert protein.sequence.decode() == sequence
            assert not protein.get_structure_mask().all()


def _assert_flow_outputs(future: StructurePredictionFuture, sequence: str) -> None:
    plddt = np.asarray(future.get_item(0, output="plddt"))
    assert plddt.shape == (NUM_SAMPLES, len(sequence))
    assert len(future.get(output="confidence")) == 1


def _sample_flow(
    session: OpenProtein,
    model_id: str,
    msa: MSAFuture,
    reference: Protein | None = None,
) -> StructurePredictionFuture:
    model = session.models.get_model(model_id)
    if isinstance(model, AlphaFlow):
        entry = model.Input(BASE_SEQUENCE, msa=msa, reference_structure=reference)
        return model.sample([entry], num_samples=NUM_SAMPLES, random_seed=SEED)
    assert isinstance(model, ESMFlow)
    entry = model.Input(BASE_SEQUENCE, reference_structure=reference)
    return model.sample([entry], num_samples=NUM_SAMPLES, random_seed=SEED)


@pytest.mark.e2e
def test_bioemu_samples_an_aligned_protein(session: OpenProtein, msa: MSAFuture):
    bioemu = session.models.bioemu_v1_1
    future = bioemu.sample(
        [bioemu.Input(BASE_SEQUENCE, msa=msa)],
        num_samples=NUM_SAMPLES,
        filter_samples=False,
        random_seed=SEED,
    )

    structures = future.wait(timeout=TIMEOUT)

    _assert_ensembles(structures, [BASE_SEQUENCE], NUM_SAMPLES)


@pytest.mark.e2e
def test_bioemu_samples_several_proteins_without_an_msa(session: OpenProtein):
    sequences = [BASE_SEQUENCE, UBIQUITIN_SEQUENCE]
    bioemu = session.models.bioemu_v1_1
    future = bioemu.sample(
        [bioemu.Input(sequence, msa=None) for sequence in sequences],
        num_samples=NUM_SAMPLES,
        filter_samples=False,
        random_seed=SEED,
    )

    structures = future.wait(timeout=TIMEOUT)

    _assert_ensembles(structures, sequences, NUM_SAMPLES)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ALPHAFLOW_IDS + ESMFLOW_IDS)
def test_flow_samples_a_protein(session: OpenProtein, msa: MSAFuture, model_id: str):
    future = _sample_flow(session, model_id, msa)

    structures = future.wait(timeout=TIMEOUT)

    _assert_ensembles(structures, [BASE_SEQUENCE], NUM_SAMPLES)
    _assert_flow_outputs(future, BASE_SEQUENCE)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", ALPHAFLOW_IDS + ESMFLOW_IDS)
def test_flow_samples_from_a_reference_structure(
    session: OpenProtein, msa: MSAFuture, reference: Protein, model_id: str
):
    future = _sample_flow(session, model_id, msa, reference)

    structures = future.wait(timeout=TIMEOUT)

    _assert_ensembles(structures, [BASE_SEQUENCE], NUM_SAMPLES)
    _assert_flow_outputs(future, BASE_SEQUENCE)
