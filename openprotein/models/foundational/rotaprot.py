"""RotaProt."""

from .base import ProteinLanguageModel

__all__ = ["RotaProt"]


class RotaProt(ProteinLanguageModel):
    """RotaProt, a masked protein language model.

    It has about 900M parameters and uses rotary position embeddings. Its checkpoints
    are ``rotaprot-large-uniref50w``, the default, and ``rotaprot-large-uniref90-ft``.
    """

    _default_id = "rotaprot-large-uniref50w"
