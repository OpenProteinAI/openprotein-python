"""RosettaFold-3."""

from .base import StructurePredictionModel

__all__ = ["RosettaFold3"]


class RosettaFold3(StructurePredictionModel):
    """RosettaFold-3, an all-atom structure prediction model.

    It predicts protein and ligand complexes; DNA and RNA chains are left out
    with a warning. Its checkpoint is ``rosettafold-3``.
    """

    _default_id = "rosettafold-3"
