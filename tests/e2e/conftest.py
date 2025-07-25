import os
from typing import Tuple, List

import pytest

from openprotein import OpenProtein
from openprotein.protein import Protein
from openprotein.align.msa import MSAFuture
from tests.utils.sequences import generate_mutated_sequences

BASE_SEQUENCE = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
E2E_TIMEOUT = 600


@pytest.fixture(scope="session")
def api_session() -> OpenProtein:
    """
    Establishes a session for the entire E2E test run using environment variables.
    Scope is 'session' to ensure connection is established only once.
    """
    username = os.environ.get("OPENPROTEIN_USERNAME")
    password = os.environ.get("OPENPROTEIN_PASSWORD")

    if not username or not password:
        pytest.fail(
            "E2E tests require OPENPROTEIN_USERNAME and OPENPROTEIN_PASSWORD to be set as environment variables."
        )

    return OpenProtein(
        username=username,
        password=password,
    )


@pytest.fixture(scope="session")
def protein_complex_with_msa(
    api_session: OpenProtein,
) -> Tuple[List[Protein], MSAFuture]:
    """
    Creates a set of protein sequences and a corresponding MSA.
    This is session-scoped to avoid recreating the MSA for every test.
    """
    sequences = generate_mutated_sequences(BASE_SEQUENCE, num_sequences=3)
    msa_future = api_session.align.create_msa(seed=":".join(sequences).encode())
    msa_future.wait_until_done(timeout=E2E_TIMEOUT)

    proteins = []
    for seq in sequences:
        protein = Protein(sequence=seq)
        protein.msa = msa_future
        proteins.append(protein)

    return proteins, msa_future
