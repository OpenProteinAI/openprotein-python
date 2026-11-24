"""Strict public v1 MSA → prompt → PoET chain."""

from typing import Any, cast

import numpy as np
import pytest

from openprotein import OpenProtein
from openprotein.jobs import JobStatus
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(3.0)


@pytest.mark.e2e
def test_msa_prompt_poet_embed(
    session: OpenProtein,
    base_sequence: str,
    mutated_sequences,
):
    """All three public stages run as jobs."""
    sequences = mutated_sequences(
        sequence=base_sequence,
        num_sequences=8,
        mutation_rate=0.05,
    )

    msa = session.align.create_msa(seed=base_sequence)
    prompt = msa.sample_prompt(num_sequences=20, random_seed=7)
    poet_embeddings = session.embedding.poet.embed(
        sequences, prompt=prompt, reduction=None
    )

    jobs = [msa, prompt, poet_embeddings]
    assert len({future.id for future in jobs}) == 3

    assert msa.wait(timeout=TIMEOUT)
    prompt_values = cast(list[list[str]], prompt.wait(timeout=TIMEOUT))
    poet_values = cast(list[Any], poet_embeddings.wait(timeout=TIMEOUT))

    assert len(prompt_values) == 1 and prompt_values[0]
    assert len(poet_values) == len(sequences)
    assert all(np.isfinite(embedding).all() for _, embedding in poet_values)
    assert all(future.status is JobStatus.SUCCESS for future in jobs)
