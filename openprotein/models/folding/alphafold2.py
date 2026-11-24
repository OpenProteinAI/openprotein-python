"""AlphaFold2."""

from .base import StructurePredictionModel

__all__ = ["AlphaFold2"]


class AlphaFold2(StructurePredictionModel):
    """AlphaFold2, which predicts proteins and protein complexes from an MSA.

    Its checkpoint is ``alphafold2``.
    """

    _default_id = "alphafold2"
