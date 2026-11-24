"""ProtT5."""

from .base import ProteinLanguageModel

__all__ = ["ProtT5"]


class ProtT5(ProteinLanguageModel):
    """ProtT5-XL, an encoder that embeds protein sequences.

    Its checkpoint is ``prott5-xl``.
    """

    _default_id = "prott5-xl"
