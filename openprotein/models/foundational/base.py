"""The base handle of the foundational protein models."""

from typing import TYPE_CHECKING, Any, cast

from openprotein.common import Reduction
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import futures

from .. import _inputs
from ..model import Model, _texts

if TYPE_CHECKING:
    from openprotein.jobs.futures import GPFitFuture, SVDFitFuture, UMAPFitFuture

__all__ = ["ProteinLanguageModel"]


class ProteinLanguageModel(Model):
    """A protein language model: embeddings pooled by default, and fits on them.

    ``fit_svd``, ``fit_umap`` and ``fit_gp`` fit on this model's embeddings,
    passing ``params`` as the embedding params, such as ``prompt_id``.
    """

    _embeds_assay_by_id = True

    def embed(
        self,
        sequences: list[str] | list[bytes] | AssayDataset,
        reduction: Reduction | None = "MEAN",
        **params: Any,
    ) -> futures.EmbeddingFuture:
        """Embed sequences or an assay's (``batch/embeddings``);
        ``reduction=None`` is per residue.
        """
        return super().embed(sequences, reduction=reduction, **params)

    def single_site(
        self, base_sequence: str | bytes, **params: Any
    ) -> futures.SingleSiteScoreFuture:
        """Score the sequence's single-site substitutions."""
        return cast(
            futures.SingleSiteScoreFuture,
            super().single_site(base_sequence, **params),
        )

    def fit_svd(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: Any = None,
        n_components: int | None = None,
        reduction: Reduction | None = None,
        **params: Any,
    ) -> "SVDFitFuture":
        """Fit an SVD on this model's embeddings of ``sequences`` or an assay's."""
        from ..svd import _fit

        _inputs.check_fit_source(sequences, assay)
        return _fit(
            self.session,
            self,
            sequences=None if sequences is None else _texts(sequences),
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            **params,
        )

    def fit_umap(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: Any = None,
        n_components: int | None = None,
        reduction: Reduction = "MEAN",
        n_neighbors: int | None = None,
        min_dist: float | None = None,
        **params: Any,
    ) -> "UMAPFitFuture":
        """Fit a UMAP on this model's embeddings of ``sequences`` or an assay's."""
        from ..umap import _fit

        _inputs.check_fit_source(sequences, assay)
        return _fit(
            self.session,
            self,
            sequences=None if sequences is None else _texts(sequences),
            assay=assay,
            reduction=reduction,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **params,
        )

    def fit_gp(
        self,
        assay: Any,
        properties: list[str],
        reduction: Reduction,
        name: str | None = None,
        description: str | None = None,
        kernel: str = "rbf",
        period: float | None = None,
        alpha: float | None = None,
        **params: Any,
    ) -> "GPFitFuture":
        """Train a Gaussian process on an assay's properties, featurized by this model."""
        from ..gp import _fit

        return _fit(
            self.session,
            self,
            assay=assay,
            properties=properties,
            reduction=reduction,
            name=name,
            description=description,
            kernel=kernel,
            period=period,
            alpha=alpha,
            **params,
        )
