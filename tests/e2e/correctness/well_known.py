"""Well-known sequences and reference data for correctness tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Human ubiquitin (P0CG48 residues 1-76): small, extremely conserved -> a strong
# logit-recovery and attention fixture, and small enough to freeze full arrays.
UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

# Antibody variable-region pair (reused from tests/utils/sequences.py) for ablang2.
ANTIBODY_HEAVY = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKYYYYGMDVWGQGTTVTVSS"
ANTIBODY_LIGHT = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"

# Reference monomer structures already committed under tests/data/ (chain id).
REFERENCE_STRUCTURES = {
    "8bo9": ("tests/data/8bo9.cif", "A"),
    "1a3n": ("tests/data/1A3N.cif", "A"),
}

# Fixed PoET conditioning context + query. PoET conditions on prompt *content*, so
# minting a fresh prompt_id from this fixed content each run busts the cache (genuine
# recompute) while the output stays reproducible against a committed baseline. The
# query is embedded/scored against the prompt. Biological relevance is irrelevant for
# the determinism/baseline checks; only fixedness matters.
POET_QUERY = UBIQUITIN
POET_CONTEXT = [
    UBIQUITIN,
    "MQIFVKTLTGKTITLEVESSDTIDNVKSKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "MQIFVKTLTGKTITLDVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLADYNIQKESTLHLVLRLRGG",
]


def amie_prompt_context(path: str | Path, n: int = 24) -> list[str]:
    """Deterministic fixed slice of AMIE sequences for use as a PoET prompt context.

    Returns the first ``n`` distinct full-length sequences in file order, so the
    context is stable across runs (reproducible variant-effect scores) yet on-family
    enough for a meaningful zero-shot correlation.
    """
    df = pd.read_csv(path)
    seqs = df["sequence"].astype(str).tolist()
    length = len(seqs[0])
    out: list[str] = []
    seen: set[str] = set()
    for s in seqs:
        if len(s) == length and s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    return out


def load_amie_dms(
    path: str | Path, measurement: str
) -> tuple[str, list[tuple[int, str, float]]]:
    """Derive the AMIE wildtype sequence and single-mutant variant effects.

    Returns (wildtype, variants) where each variant is (position_0based, mutant_aa, fitness).
    Wildtype is the per-position consensus over all sequences. Only rows that differ
    from the wildtype at exactly one position and have a finite measurement are kept.
    """
    df = pd.read_csv(path)
    sequences = df["sequence"].astype(str).tolist()
    length = len(sequences[0])
    sequences = [s for s in sequences if len(s) == length]

    # Per-column consensus = wildtype.
    wildtype_chars = []
    for col in range(length):
        counts = Counter(s[col] for s in sequences)
        wildtype_chars.append(counts.most_common(1)[0][0])
    wildtype = "".join(wildtype_chars)

    variants: list[tuple[int, str, float]] = []
    for _, row in df.iterrows():
        seq = str(row["sequence"])
        if len(seq) != length:
            continue
        fitness = row.get(measurement)
        if fitness is None or pd.isna(fitness):
            continue
        diffs = [(i, seq[i]) for i in range(length) if seq[i] != wildtype[i]]
        if len(diffs) != 1:
            continue
        pos, mut = diffs[0]
        variants.append((pos, mut, float(fitness)))
    return wildtype, variants
