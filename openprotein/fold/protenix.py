"""Community-based Protenix model for complex structure prediction."""

from collections.abc import Sequence

from pydantic import BaseModel

from openprotein.align import MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import (
    msa_future_to_complex,
    normalize_inputs,
    normalize_templates,
    resolve_templates,
    serialize_input,
)
from openprotein.molecules import Complex, Protein, Template

from . import api
from .future import FoldResultFuture
from .models import FoldModel


class ProtenixConfidence(BaseModel):
    """
    Per-sample confidence scores from a Protenix structure prediction.

    Attributes
    ----------
    ranking_score : float
        Composite ranking metric: ``0.8 * iptm + 0.2 * ptm - 100 * has_clash``.
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

    class Config:
        extra = "allow"


class ProtenixModel(FoldModel):
    """
    Class providing inference endpoints for Protenix structure prediction.
    """

    model_id: str = "protenix"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 10,
        num_steps: int = 200,
        templates: Sequence[Protein | Complex | Template] | None = None,
        **_,
    ) -> FoldResultFuture:
        """
        Request structure prediction with Protenix.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of protein complexes to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use
        templates: list[Protein | Complex | Template] | None = None
            List of templates to use for structure prediction.

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

        template_dicts = resolve_templates(
            session=self.session,
            templates=normalize_templates(
                session=self.session,
                sequences=sequences,
                templates=templates,
            ),
        )

        return FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                diffusion_samples=diffusion_samples,
                num_recycles=num_recycles,
                num_steps=num_steps,
                templates=template_dicts or None,
            ),
            complexes=normalized_complexes,
        )
