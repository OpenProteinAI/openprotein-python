"""E2E tests for align workflows."""

from typing import Callable

import pytest

from openprotein import OpenProtein
from openprotein.align.schemas import AbNumberScheme
from openprotein.molecules import Protein
from tests.e2e.config import scaled_timeout
from tests.utils.sequences import random_sequence_real

E2E_TIMEOUT = scaled_timeout(1.0)


def _strip_alignment_gaps(sequence: str) -> str:
    return sequence.replace("-", "").replace(".", "")


@pytest.mark.e2e
def test_e2e_msa_workflow(session: OpenProtein, msa_seed_sequence_length: int):
    """
    Tests the full MMSeqs MSA search workflow.
    """
    # 1. Generate unique seed sequence for this test run
    seed_sequence = random_sequence_real(msa_seed_sequence_length)

    # 2. Start an MSA search job
    msa_future = session.align.create_msa(seed=seed_sequence)

    # 3. Wait for completion
    msa_future.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    # 4. Assert status and get results
    assert msa_future.status == "SUCCESS"

    msa_iterator = msa_future.get()
    msa_data = list(msa_iterator)
    # returns list of (name, sequence) tuples
    assert (
        msa_data[0][1] == seed_sequence
    ), "Expected seed sequence to be first returned sequence"


@pytest.mark.e2e
def test_e2e_msa_caching(session: OpenProtein, msa_seed_sequence_length: int):
    """
    Tests the full MMSeqs MSA search workflow.
    """
    # Use a static seed sequence for this test
    seed_sequence = random_sequence_real(msa_seed_sequence_length)

    # 1. First Run (to populate the cache)
    msa_future_1 = session.align.create_msa(seed=seed_sequence)
    msa_future_1.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)
    assert msa_future_1.status == "SUCCESS"

    # 2. Second Run (should hit the cache)
    msa_future_2 = session.align.create_msa(seed=seed_sequence)
    msa_future_2.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    assert msa_future_2.status == "SUCCESS"
    assert list(msa_future_1.get()) == list(msa_future_2.get())


@pytest.mark.e2e
def test_e2e_msa_sample_workflow(session: OpenProtein, msa_seed_sequence_length: int):
    """
    Tests the full MMSeqs MSA search and sample workflow.
    """
    # 1. Generate unique seed sequence for this test run
    seed_sequence = random_sequence_real(msa_seed_sequence_length)

    # 2. Start an MSA search job
    msa_future = session.align.create_msa(seed=seed_sequence)

    # 3. Sample from pending MSA
    prompt_future = msa_future.sample_prompt(num_ensemble_prompts=3)

    # 4. Wait for completion
    prompt_future.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    # 5. Assert status and get results
    assert prompt_future.status == "SUCCESS"

    ensemble_prompts = prompt_future.get()
    assert len(ensemble_prompts) == 3
    for prompt in ensemble_prompts:
        if len(prompt) > 0:
            assert isinstance(prompt[0], Protein)


@pytest.mark.e2e
def test_e2e_mafft_workflow(
    session: OpenProtein,
    mutated_sequences: Callable[..., list[str]],
):
    """
    Tests the full MAFFT alignment workflow:
    1. Starts a MAFFT job with dynamically generated sequences to avoid caching.
    2. Waits for it to complete.
    3. Fetches and validates the resulting MSA.
    """
    # 1. Generate unique sequences for this test run
    generated_sequences = mutated_sequences(num_sequences=3, mutation_rate=0.05)

    # 2. Start a MAFFT job
    align_future = session.align.mafft(sequences=generated_sequences)

    # 3. Wait for completion
    align_future.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    # 4. Assert status and get results
    assert align_future.status == "SUCCESS"

    msa_iterator = align_future.get()
    msa_proteins = list(msa_iterator)

    # The number of rows should be the number of sequences
    assert len(msa_proteins) == len(
        generated_sequences
    ), "MSA data should contain a row for each sequence"

    # Verify that the original sequences can be recovered by removing gaps from the data rows
    original_sequences_from_msa = {row[1].replace("-", "") for row in msa_proteins}
    assert set(generated_sequences) == original_sequences_from_msa


@pytest.mark.e2e
def test_e2e_mafft_caching(session: OpenProtein, base_sequence: str):
    """
    Tests that submitting an identical job soon after the first one completes
    returns a cached result almost instantly.
    """
    # Use a static, non-random set of sequences for this test
    static_sequences = [
        base_sequence,
        "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVAAAA",
    ]

    # 1. First Run (to populate the cache)
    align_future_1 = session.align.mafft(sequences=static_sequences)
    align_future_1.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)
    assert align_future_1.status == "SUCCESS"

    # 2. Second Run (should hit the cache)
    align_future_2 = session.align.mafft(sequences=static_sequences)
    align_future_2.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    assert align_future_2.status == "SUCCESS"
    assert list(align_future_1.get()) == list(align_future_2.get())


@pytest.mark.e2e
def test_e2e_clustalo_workflow(
    session: OpenProtein,
    mutated_sequences: Callable[..., list[str]],
):
    """
    Tests the full ClustalO alignment workflow.
    """
    # 1. Generate unique sequences for this test run
    generated_sequences = mutated_sequences(num_sequences=3, mutation_rate=0.05)

    # 2. Start a ClustalO job
    align_future = session.align.clustalo(sequences=generated_sequences)

    # 3. Wait for completion
    align_future.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    # 4. Assert status and get results
    assert align_future.status == "SUCCESS"

    msa_iterator = align_future.get()
    msa_data = list(msa_iterator)

    # Verify that the original sequences can be recovered by removing gaps
    assert len(msa_data) == len(
        generated_sequences
    ), "MSA data should contain a row for each sequence"
    original_sequences_from_msa = {row[1].replace("-", "") for row in msa_data}
    assert set(generated_sequences) == original_sequences_from_msa


@pytest.mark.e2e
def test_e2e_abnumber_workflow(
    session: OpenProtein,
    test_antibody_sequence: str,
    mutated_sequences: Callable[..., list[str]],
):
    """
    Tests the full antibody numbering workflow.
    """
    # 1. Generate unique antibody sequences for this test run
    generated_sequences = mutated_sequences(
        sequence=test_antibody_sequence, num_sequences=3, mutation_rate=0.02
    )

    # 2. Start an AbNumber job
    align_future = session.align.abnumber(
        sequences=generated_sequences, scheme=AbNumberScheme.CHOTHIA
    )

    # 3. Wait for completion
    align_future.wait_until_done(verbose=True, timeout=E2E_TIMEOUT)

    # 4. Assert status and get results
    assert align_future.status == "SUCCESS"

    msa_iterator = align_future.get()
    msa_data = list(msa_iterator)

    # Verify that the original sequences can be recovered
    assert len(msa_data) == len(
        generated_sequences
    ), "MSA data should contain a row for each sequence"
    original_sequences_from_msa = {_strip_alignment_gaps(row[1]) for row in msa_data}
    assert set(generated_sequences) == original_sequences_from_msa
