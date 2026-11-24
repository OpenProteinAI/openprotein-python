"""Boltz-2."""

from .base import StructurePredictionModel

__all__ = ["Boltz2"]


class Boltz2(StructurePredictionModel):
    """Boltz-2, an all-atom structure and binding affinity prediction model.

    It predicts protein, DNA, RNA and ligand complexes, and ligand binding affinity. Its
    checkpoint is ``boltz-2``.
    """

    _default_id = "boltz-2"
