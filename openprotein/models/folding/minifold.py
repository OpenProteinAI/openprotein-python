"""MiniFold."""

from .base import StructurePredictionModel

__all__ = ["MiniFold"]


class MiniFold(StructurePredictionModel):
    """MiniFold, which predicts single protein chains from ESM2 embeddings.

    Its checkpoint is ``minifold``.
    """

    _default_id = "minifold"
