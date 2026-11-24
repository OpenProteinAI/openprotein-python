"""BioEmu."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from openprotein.jobs import futures

from . import _inputs
from .base import StructureEnsemblePredictionModel

if TYPE_CHECKING:
    from openprotein.align import MSAFuture
    from openprotein.data.msa import MSA
    from openprotein.molecules import Protein

__all__ = ["BioEmu", "BioEmuInput"]


@dataclass(frozen=True)
class BioEmuInput:
    """One BioEmu ensemble: a protein chain and, optionally, its MSA.

    Attributes
    ----------
    sequence : Protein, str or bytes
        The protein chain to sample.
    msa : str, MSAFuture, MSA or None
        An MSA whose query is ``sequence``. ``None`` or
        ``Protein.single_sequence_mode`` samples from the sequence alone.
    """

    sequence: "Protein | str | bytes"
    msa: "str | MSAFuture | MSA | type[Protein.NullMSA] | None"


class BioEmu(StructureEnsemblePredictionModel):
    """BioEmu samples protein ensembles from a sequence and, optionally, its MSA.

    A call currently takes one MSA or none, shared by every input, so a call
    with an MSA samples one protein. Its checkpoint is ``bioemu-v1-1``.
    """

    Input = BioEmuInput
    _default_id = "bioemu-v1-1"

    def sample(
        self,
        inputs: Sequence[BioEmuInput],
        num_samples: int = 1,
        denoiser: Literal["dpm", "heun", "physical"] = "dpm",
        filter_samples: bool = True,
        random_seed: int | None = None,
    ) -> futures.StructurePredictionFuture:
        """Sample one ensemble per input.

        Parameters
        ----------
        inputs : sequence of BioEmuInput
            One entry per ensemble.
        num_samples : int
            Conformations to sample per input.
        denoiser : {"dpm", "heun", "physical"}
            Sampler for the reverse diffusion.
        filter_samples : bool
            Drop samples that fail BioEmu's physicality checks, so an ensemble
            can hold fewer than ``num_samples`` conformations.
        random_seed : int or None
            Random seed. ``None`` lets the model pick its own, so every call
            is a fresh sample.

        Returns
        -------
        StructurePredictionFuture
            One ensemble per input, in input order.

        Examples
        --------
        One protein with its MSA:

        >>> bioemu = session.models.bioemu
        >>> future = bioemu.sample(
        ...     [bioemu.Input(sequence=sequence, msa=msa)],
        ...     num_samples=10,
        ... )
        >>> ensembles = future.get()

        Several proteins without an MSA:

        >>> future = bioemu.sample(
        ...     [bioemu.Input(sequence=s, msa=None) for s in sequences],
        ...     num_samples=10,
        ... )
        """
        complexes = [
            _inputs.protein_complex(
                _inputs.sequence_text(entry.sequence, f"inputs[{i}].sequence"),
                _inputs.msa_id(entry.msa, f"inputs[{i}].msa", required=False),
                takes_msa=True,
            )
            for i, entry in enumerate(_inputs.input_list(inputs, BioEmuInput))
        ]
        return self._sample(
            complexes=complexes,
            num_samples=num_samples,
            denoiser=denoiser,
            filter_samples=filter_samples,
            random_seed=random_seed,
        )
