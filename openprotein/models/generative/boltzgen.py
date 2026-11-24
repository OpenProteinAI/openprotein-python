"""BoltzGen."""

from .base import StructureGenerativeModel

__all__ = ["BoltzGen"]


class BoltzGen(StructureGenerativeModel):
    """BoltzGen, an all-atom model that designs binders.

    It designs protein binders against proteins, nucleic acids or small
    molecules. The query has a fixed length. Its checkpoint is ``boltzgen``.
    """

    _default_id = "boltzgen"
