"""ProteinMPNN."""

from typing import Any

from openprotein.common import Reduction
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import futures

from ..model import _prepared
from ._inverse_folding import _InverseFoldingModel

__all__ = ["ProteinMPNN"]


class ProteinMPNN(_InverseFoldingModel):
    """ProteinMPNN, a message-passing inverse-folding model.

    It scores, embeds and designs sequences for a fixed backbone. Its
    checkpoints are ``proteinmpnn``, the default, and ``solublempnn``, trained
    on soluble proteins.
    """

    _default_id = "proteinmpnn"

    def embed(
        self,
        sequences: list[str] | list[bytes] | AssayDataset,
        query: Any = None,
        reduction: Reduction | None = "MEAN",
        **params: Any,
    ) -> futures.EmbeddingFuture:
        """Embed sequences or an assay's on the query's backbone;
        ``reduction=None`` is per residue.
        """
        sequences = _prepared(sequences)
        return super().embed(
            sequences,
            reduction=reduction,
            **self._on_structure(params, query, method=("embed", "batch/embeddings")),
        )


class _SolubleMPNN(ProteinMPNN):
    _default_id = "solublempnn"
