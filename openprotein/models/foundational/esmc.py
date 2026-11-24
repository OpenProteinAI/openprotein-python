"""ESM Cambrian."""

from .base import ProteinLanguageModel

__all__ = ["ESMC"]


class ESMC(ProteinLanguageModel):
    """ESM Cambrian, a protein language model.

    It was trained on sequences from UniRef, MGnify and JGI. Its checkpoints differ in
    size: ``esmc-300m``, ``esmc-600m``, the default, and ``esmc-6b``.
    """

    _default_id = "esmc-600m"
