"""Community-based Protenix model for complex structure prediction."""

from collections.abc import Sequence

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
