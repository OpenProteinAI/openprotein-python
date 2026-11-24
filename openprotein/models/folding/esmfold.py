"""ESMFold."""

from .base import StructurePredictionModel

__all__ = ["ESMFold"]


class ESMFold(StructurePredictionModel):
    """ESMFold, which predicts protein structures from ESM2 embeddings.

    Its checkpoint is ``esmfold``.
    """

    _default_id = "esmfold"
