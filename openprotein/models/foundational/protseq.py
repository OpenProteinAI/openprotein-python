"""ProtSeq."""

from .base import ProteinLanguageModel

__all__ = ["ProtSeq"]


class ProtSeq(ProteinLanguageModel):
    """ProtSeq, a masked protein language model trained on UniRef50.

    It has about 300M parameters and was trained with contact and secondary structure
    prediction as secondary objectives. Its checkpoint is ``prot-seq``.
    """

    _default_id = "prot-seq"
