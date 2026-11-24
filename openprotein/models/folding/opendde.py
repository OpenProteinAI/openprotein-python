"""OpenDDE."""

from .base import StructurePredictionModel

__all__ = ["OpenDDE"]


class OpenDDE(StructurePredictionModel):
    """OpenDDE, an all-atom co-folding model.

    It predicts protein, DNA, RNA, ligand and ion complexes. Its checkpoints are
    ``opendde``, the default, and ``opendde-abag``, for antibody-antigen complexes.
    """

    _default_id = "opendde"
