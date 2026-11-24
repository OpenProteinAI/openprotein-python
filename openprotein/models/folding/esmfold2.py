"""ESMFold2."""

from .base import StructurePredictionModel

__all__ = ["ESMFold2"]


class ESMFold2(StructurePredictionModel):
    """ESMFold2, an all-atom structure prediction model.

    It predicts protein, DNA, RNA and ligand complexes. Its checkpoints are
    ``esmfold2``, the default, and ``esmfold2-fast``.
    """

    _default_id = "esmfold2"
