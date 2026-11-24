"""ESMFlow."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openprotein.jobs import futures

from . import _inputs
from .base import StructureEnsemblePredictionModel

if TYPE_CHECKING:
    from openprotein.molecules import Complex, Protein, Structure

__all__ = ["ESMFlow", "ESMFlowInput"]


@dataclass(frozen=True)
class ESMFlowInput:
    """One ESMFlow ensemble: a protein chain and an optional reference.

    Attributes
    ----------
    sequence : Protein, str or bytes
        The protein chain to sample.
    reference_structure : str, bytes, Protein, Complex, Structure or None
        A structure to sample around, as PDB text or a molecule holding one
        protein chain. Its residues must all be modelled and match
        ``sequence``.
    """

    sequence: "Protein | str | bytes"
    reference_structure: "str | bytes | Protein | Complex | Structure | None" = None


class ESMFlow(StructureEnsemblePredictionModel):
    """ESMFlow samples protein ensembles from a sequence alone.

    A reference structure, given for every input or none, draws each ensemble
    around it. Its checkpoints are ``esmflow-md-base-v1``, the default, and
    ``esmflow-md-distilled-v1``, which samples each conformation in one step.
    """

    Input = ESMFlowInput
    _default_id = "esmflow-md-base-v1"

    def sample(
        self,
        inputs: Sequence[ESMFlowInput],
        num_samples: int = 1,
        num_steps: int = 10,
        tmax: float = 1.0,
        self_cond: bool | None = None,
        random_seed: int | None = None,
    ) -> futures.StructurePredictionFuture:
        """Sample one ensemble per input.

        Parameters
        ----------
        inputs : sequence of ESMFlowInput
            One entry per ensemble.
        num_samples : int
            Conformations to sample per input.
        num_steps : int
            Integration steps per conformation. A distilled checkpoint samples
            each conformation in one step and ignores it.
        tmax : float
            Noise level the flow starts from, in (0, 1]. A distilled
            checkpoint ignores it.
        self_cond : bool or None
            Feed each step's prediction back in as the next step's input.
            ``None`` keeps the checkpoint's setting.
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
        >>> esmflow = session.models.esmflow
        >>> future = esmflow.sample(
        ...     [esmflow.Input(sequence=sequence, reference_structure=reference)],
        ...     num_samples=10,
        ... )
        >>> ensembles = future.get()

        Several proteins in one call, on the distilled checkpoint:

        >>> distilled = session.models.esmflow_md_distilled_v1
        >>> future = distilled.sample(
        ...     [distilled.Input(sequence=s) for s in sequences], num_samples=10
        ... )
        """
        entries = _inputs.input_list(inputs, ESMFlowInput)
        sequences = [
            _inputs.sequence_text(entry.sequence, f"inputs[{i}].sequence")
            for i, entry in enumerate(entries)
        ]
        params: dict[str, Any] = {
            "complexes": [
                _inputs.protein_complex(sequence, None, takes_msa=False)
                for sequence in sequences
            ]
        }
        references = _inputs.flow_references(
            [entry.reference_structure for entry in entries], sequences
        )
        if references is not None:
            params["reference_structures"] = references
        return self._sample(
            **params,
            num_samples=num_samples,
            num_steps=num_steps,
            tmax=tmax,
            self_cond=self_cond,
            random_seed=random_seed,
        )
