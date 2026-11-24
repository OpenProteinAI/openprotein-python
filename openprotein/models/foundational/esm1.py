"""ESM-1."""

from .base import ProteinLanguageModel

__all__ = ["ESM1"]


class ESM1(ProteinLanguageModel):
    """ESM-1, a masked protein language model with 650M parameters.

    Its checkpoints are ``esm1b_t33_650M_UR50S``, ESM-1b trained on UniRef50 and the
    default, and ``esm1v_t33_650M_UR90S_1`` to ``esm1v_t33_650M_UR90S_5``, the five
    ESM-1v models trained on UniRef90 for variant effect prediction.
    """

    _default_id = "esm1b_t33_650M_UR50S"
