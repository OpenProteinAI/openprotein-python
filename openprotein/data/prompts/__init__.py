"""Client for stored prompts (``session.data.Prompt``).

.. deprecated:: 0.18.2
    ``PromptSummary``: use ``openprotein.data.PromptMetadata``.
"""

from typing import Any

from openprotein import _deprecation

from .prompts import Prompt
from .schemas import PromptMetadata, PromptRecord

__all__ = [
    "Prompt",
    "PromptMetadata",
    "PromptRecord",
]


def __getattr__(name: str) -> Any:
    if name == "PromptSummary":
        _deprecation.warn_deprecated(
            f"{__name__}.PromptSummary",
            "`openprotein.data.PromptMetadata`",
            since="0.18.2",
            removed_in="0.19.0",
        )
        return PromptMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
