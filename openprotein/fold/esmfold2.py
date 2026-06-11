"""Community-based ESMFold2 models for complex structure prediction."""

from collections.abc import Sequence

from pydantic import BaseModel

from openprotein.align import MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import (
    msa_future_to_complex,
    normalize_inputs,
    serialize_input,
)
from openprotein.molecules import Complex, Protein

from . import api
from .future import FoldResultFuture
from .models import FoldModel


class ESMFold2Confidence(BaseModel):
    """
    Per-sample confidence scores from an ESMFold2 structure prediction.

    Attributes
    ----------
    ptm : float
        Predicted TM-score for the full complex.
    iptm : float
        Interface pTM aggregated over inter-chain residue pairs.
    complex_plddt : float
        Mean per-atom pLDDT for the complex.
    chains_ptm : dict[str, float]
        Per-chain pTM scores, keyed by chain index as a string.
    pair_chains_iptm : dict[str, dict[str, float]]
        Predicted ipTM between each pair of chains, keyed by chain indices
        as strings.
    """

    ptm: float
    iptm: float
    complex_plddt: float
    chains_ptm: dict[str, float]
    pair_chains_iptm: dict[str, dict[str, float]]


def _assert_no_protein_msa(complexes: list[Complex]) -> None:
    """Raise if any protein chain carries an MSA reference."""
    for complex in complexes:
        for chain_id, protein in complex.get_proteins().items():
            msa = protein.msa
            if isinstance(msa, (str, MSAFuture)):
                raise ValueError(
                    f"ESMFold2-Fast is a single-sequence variant and does not "
                    f"accept MSAs; chain {chain_id!r} has an MSA attached. "
                    f"Set `protein.msa = Protein.single_sequence_mode`."
                )


def _esmfold2_fold(
    session: APISession,
    model_id: str,
    sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
    diffusion_samples: int | None,
    num_recycles: int | None,
    num_steps: int | None,
    seed: int | None,
    supports_msa: bool = True,
) -> FoldResultFuture:
    for name, value in (
        ("diffusion_samples", diffusion_samples),
        ("num_recycles", num_recycles),
        ("num_steps", num_steps),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")

    if isinstance(sequences, MSAFuture):
        normalized_complexes = [msa_future_to_complex(session, sequences)]
    else:
        normalized_complexes = normalize_inputs(sequences)

    if not supports_msa:
        _assert_no_protein_msa(normalized_complexes)

    _complexes = serialize_input(session, normalized_complexes, needs_msa=True)

    if len(_complexes) == 0:
        raise ValueError("Expected non-empty sequences")

    return FoldResultFuture(
        session=session,
        job=api.fold_models_post(
            session=session,
            model_id=model_id,
            sequences=_complexes,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            seed=seed,
        ),
        complexes=normalized_complexes,
    )


class ESMFold2Model(FoldModel):
    """
    Class providing inference endpoints for ESMFold2 structure prediction.
    """

    model_id: str = "esmfold2"

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
        num_recycles: int = 3,
        num_steps: int = 100,
        seed: int | None = None,
        **_,
    ) -> FoldResultFuture:
        """
        Request structure prediction with ESMFold2.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of complexes to fold. `Protein` objects must be tagged with
            an `msa`, which can be `Protein.single_sequence_mode` for single
            sequence mode. Alternatively, supply an `MSAFuture` to use all
            query sequences as a multimer.
        diffusion_samples : int
            Number of diffusion samples to use.
        num_recycles : int
            Number of recycling steps to use.
        num_steps : int
            Number of sampling steps to use.
        seed : int | None
            Seed for the diffusion sampler.

        Returns
        -------
        FoldResultFuture
            Future for the folding result.
        """
        return _esmfold2_fold(
            session=self.session,
            model_id=self.model_id,
            sequences=sequences,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            seed=seed,
            supports_msa=True,
        )


class ESMFold2FastModel(FoldModel):
    """
    Class providing inference endpoints for ESMFold2-Fast structure prediction.

    Single-sequence variant of ESMFold2 (no MSA).
    """

    model_id: str = "esmfold2-fast"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes],
        diffusion_samples: int = 1,
        num_recycles: int = 3,
        num_steps: int = 100,
        seed: int | None = None,
        **_,
    ) -> FoldResultFuture:
        """
        Request structure prediction with ESMFold2-Fast.

        Single-sequence variant: protein chains must use `Protein.single_sequence_mode`
        rather than an MSA.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes]
            List of complexes to fold. `Protein` objects must be tagged with
            `Protein.single_sequence_mode`.
        diffusion_samples : int
            Number of diffusion samples to use.
        num_recycles : int
            Number of recycling steps to use.
        num_steps : int
            Number of sampling steps to use.
        seed : int | None
            Seed for the diffusion sampler.

        Returns
        -------
        FoldResultFuture
            Future for the folding result.
        """
        return _esmfold2_fold(
            session=self.session,
            model_id=self.model_id,
            sequences=sequences,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            seed=seed,
            supports_msa=False,
        )
