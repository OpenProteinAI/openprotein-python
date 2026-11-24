"""Protenix-v2."""

from .base import StructurePredictionModel

__all__ = ["ProtenixV2"]


class ProtenixV2(StructurePredictionModel):
    """Protenix-v2, a larger Protenix.

    It improves on Protenix for antibody-antigen complexes and ligands. Its checkpoint
    is ``protenix-v2``.
    """

    _default_id = "protenix-v2"
