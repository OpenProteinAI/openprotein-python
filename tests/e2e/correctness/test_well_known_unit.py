"""Offline unit tests for well-known fixtures."""

from pathlib import Path

import pytest

from tests.e2e.correctness import well_known as wk


def test_ubiquitin_is_76_residues():
    assert len(wk.UBIQUITIN) == 76
    assert set(wk.UBIQUITIN) <= set("ACDEFGHIKLMNPQRSTVWY")


def test_amino_acid_alphabet():
    assert wk.AMINO_ACIDS == "ACDEFGHIKLMNPQRSTVWY"


def test_load_amie_wildtype_and_variants():
    path = Path("tests/data/AMIE_PSEAE_Whitehead.wide.csv")
    if not path.exists():
        pytest.skip("AMIE DMS table not present")
    wt, variants = wk.load_amie_dms(path, measurement="acetamide_normalized_fitness")
    assert isinstance(wt, str) and len(wt) > 100
    assert set(wt) <= set(wk.AMINO_ACIDS)
    # variants: list of (position_0based, mutant_aa, fitness)
    assert len(variants) > 100
    pos, mut, fit = variants[0]
    assert 0 <= pos < len(wt)
    assert mut in wk.AMINO_ACIDS
    # every variant differs from wildtype at exactly its recorded position
    for pos, mut, fit in variants[:50]:
        assert mut != wt[pos]


def test_poet_context_is_fixed_and_distinct():
    assert isinstance(wk.POET_CONTEXT, list) and len(wk.POET_CONTEXT) >= 2
    assert len(set(wk.POET_CONTEXT)) == len(wk.POET_CONTEXT)
    assert all(set(s) <= set(wk.AMINO_ACIDS) for s in wk.POET_CONTEXT)
    assert set(wk.POET_QUERY) <= set(wk.AMINO_ACIDS)


def test_amie_prompt_context_is_deterministic():
    path = Path("tests/data/AMIE_PSEAE_Whitehead.wide.csv")
    if not path.exists():
        pytest.skip("AMIE DMS table not present")
    a = wk.amie_prompt_context(path, n=8)
    b = wk.amie_prompt_context(path, n=8)
    assert a == b  # deterministic
    assert len(a) == 8
    assert len(set(a)) == 8  # distinct
