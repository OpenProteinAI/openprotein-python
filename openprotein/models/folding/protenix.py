"""Protenix."""

from .base import StructurePredictionModel

__all__ = ["Protenix"]


class Protenix(StructurePredictionModel):
    """Protenix, an all-atom structure prediction model.

    It predicts protein, DNA, RNA and ligand complexes. Its checkpoint is ``protenix``.
    """

    _default_id = "protenix"
