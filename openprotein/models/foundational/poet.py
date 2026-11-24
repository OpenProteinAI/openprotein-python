"""PoET."""

from typing import TYPE_CHECKING, Any, cast

from openprotein.common import Reduction
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import futures

from .. import _inputs
from ..model import _prepared, _text
from .base import ProteinLanguageModel

if TYPE_CHECKING:
    from openprotein.jobs.futures import GPFitFuture, SVDFitFuture, UMAPFitFuture

__all__ = ["PoET"]


class PoET(ProteinLanguageModel):
    """PoET, a protein language model conditioned on a protein family.

    It embeds, scores and generates sequences given a prompt sampled from the
    family's MSA. Every verb takes ``prompt``, a prompt or its id, sent as
    ``prompt_id``. Its checkpoint is ``poet``.
    """

    _default_id = "poet"

    def _conditioned(
        self,
        params: dict[str, Any],
        prompt: Any = None,
        query: Any = None,
        assay: Any = None,
        method: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        _inputs.check_inputs(params, prompt=prompt, query=query, assay=assay)
        if method is not None:
            self._method(*method)
        params = dict(params)
        _inputs.with_param(params, "prompt_id", _inputs.prompt_id(prompt), "prompt")
        _inputs.with_param(
            params, "query_id", _inputs.query_ids(self.session, query), "query"
        )
        return params

    def embed(
        self,
        sequences: list[str] | list[bytes] | AssayDataset,
        reduction: Reduction | None = "MEAN",
        prompt: Any = None,
        **params: Any,
    ) -> futures.EmbeddingFuture:
        """Embed sequences or an assay's (``batch/embeddings``);
        ``reduction=None`` is per residue.
        """
        sequences = _prepared(sequences)
        return super().embed(
            sequences,
            reduction=reduction,
            **self._conditioned(params, prompt, method=("embed", "batch/embeddings")),
        )

    def logits(
        self, sequences: list[str] | list[bytes], prompt: Any = None, **params: Any
    ) -> futures.LogitsFuture:
        """Per-position logits for sequences (``batch/logits``)."""
        sequences = _prepared(sequences)
        return super().logits(
            sequences,
            **self._conditioned(params, prompt, method=("logits", "batch/logits")),
        )

    def score(
        self, sequences: list[str] | list[bytes], prompt: Any = None, **params: Any
    ) -> futures.LogLikelihoodFuture:
        """Log-likelihoods of sequences (``batch/loglikelihood``)."""
        sequences = _prepared(sequences)
        return super().score(
            sequences,
            **self._conditioned(
                params, prompt, method=("score", "batch/loglikelihood")
            ),
        )

    def single_site(
        self, base_sequence: str | bytes, prompt: Any = None, **params: Any
    ) -> futures.SingleSiteScoreFuture:
        """Score every single-site mutant of ``base_sequence`` (``single-site/scores``)."""
        base_sequence = _text(base_sequence, "base_sequence")
        return super().single_site(
            base_sequence,
            **self._conditioned(
                params, prompt, method=("single_site", "single-site/*")
            ),
        )

    def indel(
        self,
        base_sequence: str | bytes,
        insert: str | None = None,
        delete: list[int] | None = None,
        prompt: Any = None,
        **params: Any,
    ) -> futures.IndelScoreFuture:
        """Score ``base_sequence`` with ``insert`` added, or ``delete`` sizes removed, at each site."""
        base_sequence = _text(base_sequence, "base_sequence")
        if insert is not None:
            params["insert"] = insert
        if delete is not None:
            params["delete"] = delete
        return super().indel(
            base_sequence,
            **self._conditioned(params, prompt, method=("indel", "indel/*")),
        )

    def generate(
        self, prompt: Any = None, **params: Any
    ) -> futures.SequenceGenerationFuture:
        """Sample sequences (``generate/sequences``)."""
        return cast(
            futures.SequenceGenerationFuture,
            super().generate(
                **self._conditioned(params, prompt, method=("generate", "generate/*"))
            ),
        )

    def fit_svd(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: Any = None,
        n_components: int | None = None,
        reduction: Reduction | None = None,
        prompt: Any = None,
        **params: Any,
    ) -> "SVDFitFuture":
        """Fit an SVD on this model's embeddings of ``sequences`` or an assay's."""
        _inputs.check_fit_source(sequences, assay)
        sequences = _prepared(sequences)
        return super().fit_svd(
            sequences,
            assay,
            n_components=n_components,
            reduction=reduction,
            **self._conditioned(params, prompt, assay=assay),
        )

    def fit_umap(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: Any = None,
        n_components: int | None = None,
        reduction: Reduction = "MEAN",
        n_neighbors: int | None = None,
        min_dist: float | None = None,
        prompt: Any = None,
        **params: Any,
    ) -> "UMAPFitFuture":
        """Fit a UMAP on this model's embeddings of ``sequences`` or an assay's."""
        _inputs.check_fit_source(sequences, assay)
        sequences = _prepared(sequences)
        return super().fit_umap(
            sequences,
            assay,
            n_components=n_components,
            reduction=reduction,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            **self._conditioned(params, prompt, assay=assay),
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
        prompt: Any = None,
        **params: Any,
    ) -> "GPFitFuture":
        """Train a Gaussian process on an assay's properties, featurized by this model."""
        return super().fit_gp(
            assay,
            properties,
            reduction,
            name=name,
            description=description,
            kernel=kernel,
            period=period,
            alpha=alpha,
            **self._conditioned(params, prompt, assay=assay),
        )
