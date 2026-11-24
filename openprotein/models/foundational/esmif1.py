"""ESM-IF1."""

from ._inverse_folding import _InverseFoldingModel

__all__ = ["ESMIF1"]


class ESMIF1(_InverseFoldingModel):
    """ESM-IF1, an inverse-folding model.

    It scores and designs sequences for a fixed backbone. Its checkpoint is
    ``esm-if1``.
    """

    _default_id = "esm-if1"
