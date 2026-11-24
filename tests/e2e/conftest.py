import os
import pickle
from typing import Any, Callable, List, Tuple

import pandas as pd
import pytest
from filelock import FileLock

from openprotein import OpenProtein, connect
from openprotein.align.msa import MSAFuture
from openprotein.data import AssayDataset
from openprotein.molecules import Protein, Complex
from openprotein.utils.chain_id import id_generator
from tests.e2e.config import scaled_timeout
from tests.utils.sequences import (
    ANTIBODY_HEAVY_SEQUENCE,
    random_mutated_sequences,
    random_sequence_fake,
    random_sequence_real,
)
from tests.utils.strings import random_string

E2E_TIMEOUT = scaled_timeout(1.0)

# Test data paths
TEST_ASSAY_DATA_DIR = "tests/data"
TEST_ASSAY_SMALL_FILE = "AMIE_PSEAE_Whitehead.wide.15.csv"
TEST_ASSAY_MEDIUM_FILE = "AMIE_PSEAE_Whitehead.wide.1000.csv"
TEST_ASSAY_LARGE_FILE = "AMIE_PSEAE_Whitehead.wide.csv"  # ~6000 sequences

# Shared sequence constants for E2E tests
BASE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
UBIQUITIN_SEQUENCE = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
TEST_ANTIBODY_SEQUENCE = ANTIBODY_HEAVY_SEQUENCE
MSA_SEED_SEQUENCE_LENGTH = 1000


@pytest.fixture(scope="session")
def session() -> OpenProtein:
    """
    Session-scoped so the connection is established once for the whole E2E run.
    """
    try:
        return connect()
    except Exception as exc:
        pytest.fail(f"E2E tests require credentials to be setup: {exc}")


@pytest.fixture(scope="session")
def base_sequence() -> str:
    return BASE_SEQUENCE


@pytest.fixture(scope="session")
def test_antibody_sequence() -> str:
    return TEST_ANTIBODY_SEQUENCE


@pytest.fixture(scope="session")
def msa_seed_sequence_length() -> int:
    """Length used for random MSA seed sequence generation in align E2E tests."""
    return MSA_SEED_SEQUENCE_LENGTH


@pytest.fixture
def mutated_sequences(base_sequence: str) -> Callable[..., list[str]]:
    """Factory to generate mutated sequence sets from a shared base sequence."""

    def _build(
        *,
        num_sequences: int,
        mutation_rate: float = 0.05,
        sequence: str | None = None,
    ) -> list[str]:
        source_sequence = sequence if sequence is not None else base_sequence
        return random_mutated_sequences(
            source_sequence,
            num_sequences=num_sequences,
            mutation_rate=mutation_rate,
        )

    return _build


@pytest.fixture
def fixture_lookup(request: pytest.FixtureRequest) -> Callable[[str], Any]:
    """Safely resolve dynamic fixtures, skipping the test if unavailable."""

    def _lookup(fixture_name: str) -> Any:
        try:
            return request.getfixturevalue(fixture_name)
        except pytest.FixtureLookupError:
            pytest.skip(f"Fixture {fixture_name} not available")

    return _lookup


def _protein_complex_with_msa(
    session: OpenProtein,
) -> Tuple[Complex, MSAFuture]:
    """
    Creates a set of protein sequences and a corresponding MSA.
    """
    sequences = [random_sequence_real(56) for _ in range(2)]
    msa_future = session.align.create_msa(seed=":".join(sequences).encode())

    chains = {}
    id_gen = id_generator()
    for seq in sequences:
        protein = Protein(sequence=seq)
        protein.msa = msa_future
        chains[next(id_gen)] = protein
    complex = Complex(chains=chains)

    return complex, msa_future


# NOTE: ensure that MSA is only created once even w xdist
@pytest.fixture(scope="session")
def protein_complex_with_msa(session, tmp_path_factory, worker_id):
    """
    Session-scoped, and file-locked across xdist workers, so the MSA is created once.
    """
    if worker_id == "master":
        # Not running with xdist
        return _protein_complex_with_msa(session)

    # Get a temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent
    msa_file = root_tmp_dir / "protein_complex_with_msa.pkl"
    lock_file = root_tmp_dir / "msa.lock"

    with FileLock(str(lock_file)):
        if msa_file.exists():
            # Another worker already created it
            with open(msa_file, "rb") as f:
                return pickle.load(f)
        else:
            # First worker to acquire lock creates it
            protein_complex_with_msa = _protein_complex_with_msa(session)

            with open(msa_file, "wb") as f:
                pickle.dump(protein_complex_with_msa, f)

            return protein_complex_with_msa


@pytest.fixture(scope="session")
def assay_small(session: OpenProtein) -> AssayDataset:
    """
    Upload the small assay dataset (~41 sequences) for training tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_SMALL_FILE)
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Small_{random_string(8)}"
    return session.data.Assay.create(
        table=df, name=assay_name, description="Small test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def assay_medium(session: OpenProtein) -> AssayDataset:
    """
    Upload the medium assay dataset (~1000 sequences) for training tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_MEDIUM_FILE)
    if not os.path.exists(filepath):
        pytest.skip(f"Medium assay dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Medium_{random_string(8)}"
    return session.data.Assay.create(
        table=df, name=assay_name, description="Medium test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def assay_large(session: OpenProtein) -> AssayDataset:
    """
    Upload the large assay dataset (~10000 sequences) for training tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_LARGE_FILE)
    if not os.path.exists(filepath):
        pytest.skip(f"Large assay dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Large_{random_string(8)}"
    return session.data.Assay.create(
        table=df, name=assay_name, description="Large test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def test_sequences_varied() -> List[bytes]:
    """
    Sequences of varied length, as bytes for API compatibility.
    """
    seqs = [random_sequence_fake(64).encode() for _ in range(150)]
    # remove two residues at every other sequence
    for i in range(0, len(seqs), 2):
        seqs[i] = seqs[i][:-2]
    return seqs


@pytest.fixture(scope="session")
def test_sequences_short() -> List[bytes]:
    seqs = [random_sequence_fake(64).encode() for _ in range(10)]
    return seqs


@pytest.fixture(scope="session")
def test_sequences_same_length() -> List[bytes]:
    """
    SVD and UMAP require same-length sequences when not using a reduction.
    """
    # All sequences are 64 residues long
    return [random_sequence_fake(64).encode() for _ in range(150)]
