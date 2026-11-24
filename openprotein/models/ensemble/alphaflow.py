"""AlphaFlow."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openprotein.jobs import futures

from . import _inputs
from .base import StructureEnsemblePredictionModel

if TYPE_CHECKING:
    from openprotein.align import MSAFuture
    from openprotein.data.msa import MSA
    from openprotein.molecules import Complex, Protein, Structure

__all__ = ["AlphaFlow", "AlphaFlowInput"]


@dataclass(frozen=True)
class AlphaFlowInput:
    """One AlphaFlow ensemble: a protein chain, its MSA and an optional reference.

    Attributes
    ----------
    sequence : Protein, str or bytes
        The protein chain to sample.
    msa : str, MSAFuture or MSA
        An MSA whose query is ``sequence``.
    reference_structure : str, bytes, Protein, Complex, Structure or None
        A structure to sample around, as PDB text or a molecule holding one
        protein chain. Its residues must all be modelled and match
        ``sequence``.
    """

    sequence: "Protein | str | bytes"
    msa: "str | MSAFuture | MSA"
    reference_structure: "str | bytes | Protein | Complex | Structure | None" = None


class AlphaFlow(StructureEnsemblePredictionModel):
    """AlphaFlow samples protein ensembles from a sequence and its MSA.

    A call currently takes one MSA, shared by every input, so a call samples
    one protein. A reference structure, given for every input or none, draws
    each ensemble around it. Its checkpoints are ``alphaflow-md-base-v1``, the
    default, and ``alphaflow-md-distilled-v1``, which samples each conformation
    in one step.
    """

    Input = AlphaFlowInput
    _default_id = "alphaflow-md-base-v1"

    def sample(
        self,
        inputs: Sequence[AlphaFlowInput],
        num_samples: int = 1,
        num_steps: int = 10,
        tmax: float = 1.0,
        msa_subsample: int | None = None,
        self_cond: bool | None = None,
        resample: bool | None = None,
        random_seed: int | None = None,
    ) -> futures.StructurePredictionFuture:
        """Sample one ensemble per input.

        Parameters
        ----------
        inputs : sequence of AlphaFlowInput
            One entry per ensemble.
        num_samples : int
            Conformations to sample per input.
        num_steps : int
            Integration steps per conformation. A distilled checkpoint samples
            each conformation in one step and ignores it.
        tmax : float
            Noise level the flow starts from, in (0, 1]. A distilled
            checkpoint ignores it.
        msa_subsample : int or None
            Cap on the MSA depth the model reads. ``None`` keeps the
            checkpoint's setting.
        self_cond : bool or None
            Feed each step's prediction back in as the next step's input.
            ``None`` keeps the checkpoint's setting.
        resample : bool or None
            Redraw the MSA clusters for each conformation. ``None`` keeps the
            checkpoint's setting.
        random_seed : int or None
            Random seed. ``None`` lets the model pick its own, so every call
            is a fresh sample.

        Returns
        -------
        StructurePredictionFuture
            One ensemble per input, in input order. ``plddt`` and
            ``confidence`` outputs are read with ``get(output=...)``.

        Examples
        --------
        >>> alphaflow = session.models.alphaflow
        >>> future = alphaflow.sample(
        ...     [
        ...         alphaflow.Input(
        ...             sequence=sequence, msa=msa, reference_structure=reference
        ...         )
        ...     ],
        ...     num_samples=10,
        ...     num_steps=10,
        ... )
        >>> ensembles = future.get()
        >>> plddt = future.get(output="plddt")

        The distilled checkpoint samples each conformation in one step:

        >>> distilled = session.models.alphaflow_md_distilled_v1
        >>> future = distilled.sample(
        ...     [distilled.Input(sequence=sequence, msa=msa)], num_samples=10
        ... )
        """
        entries = _inputs.input_list(inputs, AlphaFlowInput)
        sequences = [
            _inputs.sequence_text(entry.sequence, f"inputs[{i}].sequence")
            for i, entry in enumerate(entries)
        ]
        complexes = [
            _inputs.protein_complex(
                sequence,
                _inputs.msa_id(entry.msa, f"inputs[{i}].msa", required=True),
                takes_msa=True,
            )
            for i, (entry, sequence) in enumerate(zip(entries, sequences))
        ]
        references = _inputs.flow_references(
            [entry.reference_structure for entry in entries], sequences
        )
        params: dict[str, Any] = {"complexes": complexes}
        if references is not None:
            params["reference_structures"] = references
        return self._sample(
            **params,
            num_samples=num_samples,
            num_steps=num_steps,
            tmax=tmax,
            msa_subsample=msa_subsample,
            self_cond=self_cond,
            resample=resample,
            random_seed=random_seed,
        )
