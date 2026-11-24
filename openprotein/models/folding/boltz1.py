"""Boltz-1."""

from .base import StructurePredictionModel

__all__ = ["Boltz1"]


class Boltz1(StructurePredictionModel):
    """Boltz-1, an all-atom structure prediction model.

    It predicts protein, DNA, RNA and ligand complexes. Its checkpoints are ``boltz-1``,
    the default, and ``boltz-1x``.
    """

    _default_id = "boltz-1"
