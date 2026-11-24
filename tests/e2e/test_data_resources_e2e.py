"""Stored assays, MSAs and prompts through ``session.data``."""

import pandas as pd
import pytest

from openprotein import OpenProtein
from openprotein.data import MSA, Assay, Prompt
from openprotein.errors import HTTPError
from openprotein.jobs import JobStatus
from tests.e2e.config import scaled_timeout

TIMEOUT = scaled_timeout(3.0)

QUERY = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
A3M = (
    f">query\n{QUERY}\n"
    f">h1\n{QUERY[:10]}aa{QUERY[10:]}\n"
    f">h2\n{QUERY[:20]}-{QUERY[21:]}\n"
    f">h3\n{QUERY[:5]}gg{QUERY[5:30]}--{QUERY[32:]}\n"
    f">h4\n{QUERY}\n"
    f">h5\n{QUERY[:-1]}A\n"
)


def _raises_not_found(load) -> None:
    with pytest.raises(HTTPError) as error:
        load()
    assert error.value.status_code == 404


@pytest.mark.e2e
def test_an_msa_is_created_read_sampled_and_deleted(session: OpenProtein):
    msa = session.data.MSA.create(A3M, format="a3m", name="e2e-resource")
    try:
        assert isinstance(msa, MSA) and msa.status is JobStatus.SUCCESS
        assert (len(msa), msa.has_insertions) == (6, True)

        rows = list(msa)
        assert [name for name, _ in rows] == ["query", "h1", "h2", "h3", "h4", "h5"]
        assert {len(sequence) for _, sequence in rows} == {len(QUERY)}
        assert msa[0] == ("query", QUERY)
        assert "aa" in msa.to_string() and "aa" not in msa.to_string("fasta")

        loaded = session.data.MSA(id=msa.id)
        assert loaded.update(description="read back").description == "read back"
        assert msa.id in [listed.id for listed in session.data.MSA.list(limit=20)]

        prompt = msa.sample_prompt(replicates=2, seed=7, max_msa_sequences=3)
        assert isinstance(prompt, Prompt)
        assert prompt.wait(timeout=TIMEOUT) is prompt
        assert len(prompt) == 2 and all(len(replicate) == 3 for replicate in prompt)
        assert session.data.Prompt(id=prompt.id).status is JobStatus.SUCCESS

        embeddings = session.embedding.poet.embed([QUERY.encode()], prompt=prompt)
        assert len(embeddings.wait(timeout=TIMEOUT)) == 1
    finally:
        msa.delete()
    _raises_not_found(lambda: session.data.MSA(id=msa.id))


@pytest.mark.e2e
def test_a_prompt_is_created_read_used_and_deleted(session: OpenProtein):
    replicate = f">a\n{QUERY}\n>b\n{QUERY[:-1]}A\n"
    prompt = session.data.Prompt.create([replicate, replicate], name="e2e-resource")
    try:
        assert (prompt.status, prompt.job_id) == (JobStatus.SUCCESS, None)
        assert list(prompt) == [[QUERY, QUERY[:-1] + "A"]] * 2
        assert prompt.update(name="e2e-renamed").name == "e2e-renamed"
        assert prompt.id in [listed.id for listed in session.data.Prompt.list(limit=20)]

        embeddings = session.embedding.poet.embed([QUERY.encode()], prompt=prompt)
        assert len(embeddings.wait(timeout=TIMEOUT)) == 1
    finally:
        prompt.delete()
    _raises_not_found(lambda: session.data.Prompt(id=prompt.id))


@pytest.mark.e2e
def test_an_assay_is_created_read_and_deleted(session: OpenProtein):
    table = pd.DataFrame({"sequence": [QUERY, QUERY[:-1] + "A"], "y": [1.0, 2.0]})
    assay = session.data.Assay.create(table, "e2e-resource")
    assert isinstance(assay, Assay) and len(assay) == 2
    assert assay.update(description="read back").description == "read back"
    assert session.data.Assay(id=assay.id).name == "e2e-resource"
    assay.delete()
    _raises_not_found(lambda: session.data.Assay(id=assay.id))
