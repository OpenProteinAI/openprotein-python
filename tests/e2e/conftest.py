import os
import pickle
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
from filelock import FileLock

from openprotein import OpenProtein, connect
from openprotein.align.msa import MSAFuture
from openprotein.data import AssayDataset
from openprotein.molecules import Protein, Complex
from openprotein.utils.chain_id import id_generator
from tests.utils.sequences import (
    random_mutated_sequences,
    random_sequence_fake,
    random_sequence_real,
)
from tests.utils.strings import random_string

E2E_TIMEOUT = 600

# Test data paths
TEST_ASSAY_DATA_DIR = "tests/data"
TEST_ASSAY_SMALL_FILE = "AMIE_PSEAE_Whitehead.wide.15.csv"
TEST_ASSAY_MEDIUM_FILE = "AMIE_PSEAE_Whitehead.wide.1000.csv"
TEST_ASSAY_LARGE_FILE = "AMIE_PSEAE_Whitehead.wide.csv"  # ~6000 sequences


@pytest.fixture(scope="session")
def session() -> OpenProtein:
    """
    Establishes a session for the entire E2E test run using environment variables.
    Scope is 'session' to ensure connection is established only once.
    """
    try:
        return connect()
    except:
        pytest.fail("E2E tests require credentials to be setup.")


def _protein_complex_with_msa(
    session: OpenProtein,
) -> Tuple[Complex, MSAFuture]:
    """
    Creates a set of protein sequences and a corresponding MSA.
    """
    sequences = [random_sequence_real(56) for _ in range(2)]
    msa_future = session.align.create_msa(seed=":".join(sequences).encode())
    # msa_future.wait_until_done(timeout=E2E_TIMEOUT)

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
    Creates a set of protein sequences and a corresponding MSA.
    This is session-scoped to avoid recreating the MSA for every test.
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
    Upload small assay dataset (~41 sequences) for training tests.
    Session-scoped to reuse across tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_SMALL_FILE)
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Small_{random_string(8)}"
    return session.data.create(
        table=df, name=assay_name, description="Small test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def assay_medium(session: OpenProtein) -> AssayDataset:
    """
    Upload medium assay dataset (~1000 sequences) for training tests.
    Session-scoped to reuse across tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_MEDIUM_FILE)
    if not os.path.exists(filepath):
        pytest.skip(f"Medium assay dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Medium_{random_string(8)}"
    return session.data.create(
        table=df, name=assay_name, description="Medium test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def assay_large(session: OpenProtein) -> AssayDataset:
    """
    Upload large assay dataset (~10000 sequences) for training tests.
    Session-scoped to reuse across tests.
    """
    filepath = os.path.join(TEST_ASSAY_DATA_DIR, TEST_ASSAY_LARGE_FILE)
    if not os.path.exists(filepath):
        pytest.skip(f"Large assay dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    assay_name = f"E2E_Large_{random_string(8)}"
    return session.data.create(
        table=df, name=assay_name, description="Large test assay for E2E tests"
    )


@pytest.fixture(scope="session")
def test_sequences_varied() -> List[bytes]:
    """
    Generate a variety of test sequences with different characteristics.
    Returns sequences as bytes for API compatibility.
    """
    seqs = [random_sequence_fake(64).encode() for _ in range(150)]
    # remove two residues at every other sequence
    for i in range(0, len(seqs), 2):
        seqs[i] = seqs[i][:-2]
    return seqs


@pytest.fixture(scope="session")
def test_sequences_short() -> List[bytes]:
    """Short sequences for quick tests."""
    seqs = [random_sequence_fake(64).encode() for _ in range(10)]
    return seqs


@pytest.fixture(scope="session")
def test_sequences_same_length() -> List[bytes]:
    """
    Sequences of the same length for SVD/UMAP tests.
    SVD and UMAP require same-length sequences when not using reduction.
    """
    # All sequences are 64 residues long
    return [random_sequence_fake(64).encode() for _ in range(150)]
