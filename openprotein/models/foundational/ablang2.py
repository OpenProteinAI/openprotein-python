"""AbLang2."""

from .base import ProteinLanguageModel

__all__ = ["AbLang2"]


class AbLang2(ProteinLanguageModel):
    """AbLang2, an antibody language model for paired heavy and light chains.

    Its checkpoint is ``ablang2``.
    """

    _default_id = "ablang2"
