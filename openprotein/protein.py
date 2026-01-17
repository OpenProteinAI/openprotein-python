import warnings

warnings.warn(
    "openprotein.protein is deprecated and will be removed in v0.11. "
    "Use `from openprotein.molecules import Protein` instead.",
    FutureWarning,
    stacklevel=2,
)

from openprotein.molecules import Protein

__all__ = ["Protein"]
