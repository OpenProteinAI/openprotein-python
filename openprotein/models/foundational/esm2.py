"""ESM2."""

from .base import ProteinLanguageModel

__all__ = ["ESM2"]


class ESM2(ProteinLanguageModel):
    """ESM2, a masked protein language model trained on UniRef50.

    Its checkpoints differ only in size: ``esm2_t6_8M_UR50D``, ``esm2_t12_35M_UR50D``,
    ``esm2_t30_150M_UR50D``, ``esm2_t33_650M_UR50D``, the default, and
    ``esm2_t36_3B_UR50D``.
    """

    _default_id = "esm2_t33_650M_UR50D"
