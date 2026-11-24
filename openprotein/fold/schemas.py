"""Schema for OpenProtein fold system."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from openprotein.jobs import BatchJob, Job, JobType

__all__ = ["FoldConfidence", "FoldJob", "FoldMetadata"]


class FoldMetadata(BaseModel):
    """
    Metadata for a folding job.

    Attributes
    ----------
    job_id : str
        Unique identifier for the job.
    model_id : str
        Identifier for the model used in the job.
    args : dict or None, optional
        Additional arguments for the job.
    """

    job_id: str
    model_id: str
    args: dict | None = None
    sequences: list[list[dict[str, Any]]] | None = None


class FoldJob(Job, BatchJob):
    """
    Folding job class.

    Attributes
    ----------
    job_type : Literal[JobType.embeddings_fold]
        Discriminator identifying a fold job.
    """

    job_type: Literal[JobType.embeddings_fold]


_UnitScore = Annotated[float, Field(ge=0, le=1)]
_PlddtScore = Annotated[float, Field(ge=0, le=100)]
_Angstrom = Annotated[float, Field(ge=0)]


class FoldConfidence(BaseModel):
    """
    Confidence scores for one sample of a fold prediction.

    The ``confidence`` output of a fold job holds, per complex, a
    ``json-utf8-v1`` list with one object per sample. Every key is present in
    every object, null when the model does not produce that score, so the
    objects of every fold model share one layout. Numbers are finite.

    Fold outputs, per complex; a model leaves out any output it does not
    produce:

    - ``structures``: one mmCIF holding all S samples as models 1..S;
      ``auth_asym_id`` is the request chain id, B-factor is pLDDT 0-100.
    - ``plddt``: ``[S, T]`` float32, 0-100, per token.
    - ``pae``, ``pde``: ``[S, T, T]`` float32, in Å.
    - ``confidence``: ``list[FoldConfidence]`` of length S.
    - ``affinity`` (boltz-2): inline object, null when not requested.

    Conventions:

    - Samples are sorted by ``ranking_score``, highest first; ties and models
      without a ranker keep generation order. mmCIF model k+1, array index k
      and ``confidence[k]`` are one sample.
    - Chains follow request order with copies expanded; every per-chain list
      follows ``chain_ids``.
    - Tokens follow AF3 tokenisation: one per standard residue or nucleotide,
      one per heavy atom of a ligand or modified residue, in chain order then
      sequence order. Protein-only models use one token per residue.
    - NaN marks a token with no placed atoms and appears nowhere else.
    - pLDDT-type values are 0-100 and pTM-type values 0-1.
    - PAE and PDE are expectations over bin centres,
      ``X[i, j] = sum_b p[i, j, b] * (b + 0.5) * width``; 64 bins over
      [0, 32] Å give centres 0.25 to 31.75. ``pae[s, i, j]`` is the error at
      token j aligned on token i's frame.
    - ``ptm``, ``iptm``, ``ranking_score``, ``complex_plddt`` and
      ``chain_pair_iptm`` are model-native; compare them within one model.

    Attributes
    ----------
    chain_ids : list[str]
        Request chain ids with copies expanded to distinct ids; C entries,
        non-empty and unique.
    ranking_score : float or None
        The model's sample ranker, higher is better; None when it has none.
    ptm : float or None
        pTM, 0-1.
    iptm : float or None
        ipTM, 0-1; None when C = 1.
    complex_plddt : float
        Mean pLDDT, 0-100.
    chain_plddt : list[float]
        Mean pLDDT per chain, 0-100.
    chain_ptm : list[float or None]
        pTM per chain, 0-1.
    chain_pair_iptm : list[list[float or None]]
        C x C pairwise ipTM, 0-1; the diagonal is ``chain_ptm``.
    ipae : float or None
        Interface PAE in Å. With T0 the tokens of chain 0 and R all other
        tokens, ``(mean(pae[s, T0, R]) + mean(pae[s, R, T0])) / 2``, NaN
        skipped. None when C = 1 or the model predicts no PAE.
    complex_iplddt : float or None
        Interface-weighted mean pLDDT, 0-100.
    complex_pde : float or None
        Contact-probability-weighted mean PDE, in Å.
    complex_ipde : float or None
        Interface mean PDE, in Å.
    protein_iptm : float or None
        ipTM over protein-protein interfaces, 0-1.
    ligand_iptm : float or None
        ipTM over protein-ligand interfaces, 0-1.
    chain_iptm : list[float] or None
        ipTM per chain, 0-1.
    has_clash : bool or None
        Whether the sample has a steric clash.
    """

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    chain_ids: list[str] = Field(min_length=1)
    ranking_score: float | None
    ptm: _UnitScore | None
    iptm: _UnitScore | None
    complex_plddt: _PlddtScore
    chain_plddt: list[_PlddtScore]
    chain_ptm: list[_UnitScore | None]
    chain_pair_iptm: list[list[_UnitScore | None]]
    ipae: _Angstrom | None
    complex_iplddt: _PlddtScore | None
    complex_pde: _Angstrom | None
    complex_ipde: _Angstrom | None
    protein_iptm: _UnitScore | None
    ligand_iptm: _UnitScore | None
    chain_iptm: list[_UnitScore] | None
    has_clash: StrictBool | None

    @model_validator(mode="after")
    def _check_layout(self) -> "FoldConfidence":
        num_chains = len(self.chain_ids)
        if len(set(self.chain_ids)) != num_chains:
            raise ValueError("chain_ids must be unique")
        for key in ("chain_plddt", "chain_ptm", "chain_iptm"):
            values = getattr(self, key)
            if values is not None and len(values) != num_chains:
                raise ValueError(f"{key} needs one entry per chain in chain_ids")
        if len(self.chain_pair_iptm) != num_chains or any(
            len(row) != num_chains for row in self.chain_pair_iptm
        ):
            raise ValueError("chain_pair_iptm must be C x C for C chain_ids")
        if num_chains == 1:
            for key in ("iptm", "ipae"):
                if getattr(self, key) is not None:
                    raise ValueError(f"{key} must be null for one chain")
        return self
