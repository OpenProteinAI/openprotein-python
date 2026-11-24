"""OpenDDE all-atom co-folding model for complex structure prediction."""

from pydantic import BaseModel

from openprotein.align import MSAFuture
from openprotein.fold.common import (
    FoldInput,
    msa_future_to_complex,
    normalize_inputs,
    serialize_input,
)

from . import api
from .future import FoldResultFuture
from .models import FoldModel

__all__ = ["OpenDDEAbagModel", "OpenDDEConfidence", "OpenDDEModel"]


class OpenDDEConfidence(BaseModel):
    """
    Per-sample confidence scores from an OpenDDE structure prediction.

    Attributes
    ----------
    ranking_score : float
        Composite ranking metric:
        ``0.8 * iptm + 0.2 * ptm + 0.5 * disorder - 100 * has_clash``.
    ptm : float
        Predicted TM-score for the full complex.
    iptm : float
        Interface pTM aggregated over inter-chain residue pairs.
    plddt : float
        Mean per-atom pLDDT in [0, 100].
    gpde : float
        Global PDE weighted by contact probabilities.
    has_clash : float
        Binary clash flag (1.0 if atomic clashes detected, else 0.0).
    num_recycles : int
        Number of recycling iterations used.
    disorder : float
        Disorder score (currently always 0.0).
    chain_ptm : list[float]
        Per-chain pTM scores, indexed by chain.
    chain_iptm : list[float]
        Per-chain ipTM scores.
    chain_plddt : list[float]
        Per-chain mean pLDDT scores.
    chain_gpde : list[float]
        Per-chain global PDE scores.
    chain_pair_iptm : list[list[float]]
        Chain-pair ipTM matrix.
    chain_pair_iptm_global : list[list[float]]
        Chain-pair ipTM matrix with ligand-aware weighting.
    chain_pair_gpde : list[list[float]]
        Chain-pair global PDE matrix.
    chain_pair_plddt : list[list[float]]
        Chain-pair mean pLDDT matrix.

    Notes
    -----
    OpenDDE emits three further scores - ``pb_ranking_score``,
    ``has_vdw_pl_clash`` and ``pb_ranking_score_vdw_penalized`` - only when a
    prediction targets a specific binder chain. This API never requests that, so
    they are not declared here; ``extra = "allow"`` passes them through should
    that change.
    """

    ranking_score: float
    ptm: float
    iptm: float
    plddt: float
    gpde: float
    has_clash: float
    num_recycles: int
    disorder: float
    chain_ptm: list[float]
    chain_iptm: list[float]
    chain_plddt: list[float]
    chain_gpde: list[float]
    chain_pair_iptm: list[list[float]]
    chain_pair_iptm_global: list[list[float]]
    chain_pair_gpde: list[list[float]]
    chain_pair_plddt: list[list[float]]

    class Config:
        extra = "allow"


class OpenDDEModel(FoldModel):
    """
    Class providing inference endpoints for OpenDDE structure prediction.

    OpenDDE folds protein, DNA, RNA, ligand and ion entities into a single
    all-atom complex, optionally conditioned on an MSA per protein chain.
    """

    model_id: str = "opendde"

    def fold(
        self,
        sequences: FoldInput | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 10,
        num_steps: int = 200,
        seed: int | None = None,
        **_,
    ) -> FoldResultFuture:
        """
        Request structure prediction with OpenDDE.

        Parameters
        ----------
        sequences : FoldInput | MSAFuture
            List of complexes to fold. `Protein` objects must be tagged with an
            `msa`, which can be `Protein.single_sequence_mode` to fold without
            one. Alternatively, supply an `MSAFuture` to use all its query
            sequences as a multimer.
        diffusion_samples : int
            Number of diffusion samples to generate per prediction.
        num_recycles : int
            Number of recycling iterations to run.
        num_steps : int
            Number of diffusion sampling steps.
        seed : int | None
            Seed for the diffusion sampling, from 0 to 2**32 - 1. None lets
            the model pick its own, so every call is a fresh prediction.
            Input preparation, ligand conformers included, is not seeded,
            so folding the same input with the same seed can still give
            different structures, especially for complexes with a ligand.

        Returns
        -------
        FoldResultFuture
            Future for the folding results.
        """
        if isinstance(sequences, MSAFuture):
            normalized_complexes = [msa_future_to_complex(self.session, sequences)]
        else:
            normalized_complexes = normalize_inputs(sequences)
        _complexes = serialize_input(self.session, normalized_complexes, needs_msa=True)

        if len(_complexes) == 0:
            raise ValueError("Expected non-empty sequences")

        return FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                diffusion_samples=diffusion_samples,
                num_recycles=num_recycles,
                num_steps=num_steps,
                seed=seed,
            ),
            complexes=normalized_complexes,
        )


class OpenDDEAbagModel(OpenDDEModel):
    """
    Class providing inference endpoints for the OpenDDE antibody-antigen model.

    Shares OpenDDE's fold API surface and confidence schema, with weights tuned
    on antibody-antigen complexes.
    """

    model_id = "opendde-abag"
