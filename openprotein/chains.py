import warnings

warnings.warn(
    "openprotein.chains is deprecated and will be removed in v0.11. "
    "Use `from openprotein.molecules import DNA, RNA, Ligand` instead.",
    FutureWarning,
    stacklevel=2,
)

from openprotein.molecules import DNA, RNA, Ligand

__all__ = ["DNA", "RNA", "Ligand"]
