"""PoET-2."""

from typing import TYPE_CHECKING, Any

from openprotein.common import Reduction
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import futures

from .. import _inputs
from ..model import _prepared, _text
from .poet import PoET

if TYPE_CHECKING:
    from openprotein.jobs.futures import GPFitFuture, SVDFitFuture, UMAPFitFuture

__all__ = ["PoET2"]


class PoET2(PoET):
    """PoET-2, a multi-modal protein language model conditioned on a protein family.

    Like PoET, every verb takes ``prompt``. Verbs other than ``indel`` also
    take ``query``, a sequence, ``Protein``, ``Complex``, ``Query`` or query id,
    sent as ``query_id``, which conditions on a protein's sequence or
    structure. Its checkpoint is ``poet-2``.
    """

    _default_id = "poet-2"

    def embed(
        self,
        sequences: list[str] | list[bytes] | AssayDataset,
        reduction: Reduction | None = "MEAN",
        prompt: Any = None,
        query: Any = None,
        **params: Any,
    ) -> futures.EmbeddingFuture:
        """Embed sequences or an assay's (``batch/embeddings``);
        ``reduction=None`` is per residue.
        """
        sequences = _prepared(sequences)
        return super().embed(
            sequences,
            reduction=reduction,
            **self._conditioned(
                params, prompt, query, method=("embed", "batch/embeddings")
            ),
        )

    def logits(
        self,
        sequences: list[str] | list[bytes],
        prompt: Any = None,
        query: Any = None,
        **params: Any,
    ) -> futures.LogitsFuture:
        """Per-position logits for sequences (``batch/logits``)."""
        sequences = _prepared(sequences)
        return super().logits(
            sequences,
            **self._conditioned(
                params, prompt, query, method=("logits", "batch/logits")
            ),
        )

    def attn(
        self,
        sequences: list[str] | list[bytes],
        prompt: Any = None,
        query: Any = None,
        **params: Any,
    ) -> futures.AttentionFuture:
        """Attention maps for sequences (``batch/attn``)."""
        sequences = _prepared(sequences)
        return super().attn(
            sequences,
            **self._conditioned(params, prompt, query, method=("attn", "batch/attn")),
        )

    def score(
        self,
        sequences: list[str] | list[bytes],
        prompt: Any = None,
        query: Any = None,
        **params: Any,
    ) -> futures.LogLikelihoodFuture:
        """Log-likelihoods of sequences (``batch/loglikelihood``)."""
        sequences = _prepared(sequences)
        return super().score(
            sequences,
            **self._conditioned(
                params, prompt, query, method=("score", "batch/loglikelihood")
            ),
        )

    def single_site(
        self,
        base_sequence: str | bytes,
        prompt: Any = None,
        query: Any = None,
        **params: Any,
    ) -> futures.SingleSiteScoreFuture:
        """Score every single-site mutant of ``base_sequence`` (``single-site/scores``)."""
        base_sequence = _text(base_sequence, "base_sequence")
        return super().single_site(
            base_sequence,
            **self._conditioned(
                params, prompt, query, method=("single_site", "single-site/*")
            ),
        )

    def generate(
        self,
        prompt: Any = None,
        query: Any = None,
        design: Any = None,
        **params: Any,
    ) -> futures.SequenceGenerationFuture:
        """Sample sequences (``generate/sequences``).

        ``query`` is one or a list. ``design``, an RFdiffusion or BoltzGen job or
        its id, in any state, takes the place of ``query``: its designed
        structures are the queries, and the job runs once the design has
        finished.
        """
        _inputs.check_inputs(params, prompt=prompt, query=query, design=design)
        _inputs.check_one_structure_source(params, query, design)
        self._method("generate", "generate/*")
        many = _inputs.takes_list(self._params_schema("generate/sequences"), "query_id")
        params = dict(params)
        _inputs.with_param(params, "prompt_id", _inputs.prompt_id(prompt), "prompt")
        _inputs.with_param(
            params,
            "query_id",
            _inputs.query_ids(self.session, query, many=many),
            "query",
        )
        _inputs.with_param(params, "design_id", _inputs.design_id(design), "design")
        return super().generate(**params)

    def fit_svd(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: Any = None,
        n_components: int | None = None,
        reduction: Reduction | None = None,
        prompt: Any = None,
        query: Any = None,
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
            **self._conditioned(params, prompt, query, assay),
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
        query: Any = None,
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
            **self._conditioned(params, prompt, query, assay),
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
        query: Any = None,
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
            **self._conditioned(params, prompt, query, assay),
        )
