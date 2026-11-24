"""The verbs inverse-folding models share."""

from typing import Any, cast

from openprotein import _deprecation
from openprotein.jobs import futures

from .. import _inputs
from ..model import Model, _prepared, _text


def _renamed(params: dict[str, Any], method: str, old: str, new: str) -> None:
    """Move a deprecated keyword in ``params`` to its current name."""
    if old not in params:
        return
    _deprecation.warn_deprecated(
        f"{method}({old}=...)",
        f"`{method}({new}=...)`",
        since="0.18.0",
        removed_in="0.19.0",
        stacklevel=4,
    )
    if params.get(new) is not None:
        raise TypeError(f"pass {new}, not both {new} and {old}")
    params[new] = params.pop(old)


class _InverseFoldingModel(Model):
    """A model that scores and designs sequences on a structure.

    Every verb takes ``query``, the structure to condition on: a ``Protein``,
    ``Complex``, ``Query`` or query id, sent as ``query_id``.
    """

    def _on_structure(
        self,
        params: dict[str, Any],
        query: Any = None,
        design: Any = None,
        many: bool = False,
        method: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        _inputs.check_inputs(params, query=query, design=design)
        _inputs.check_one_structure_source(params, query, design)
        if method is not None:
            self._method(*method)
        params = dict(params)
        _inputs.with_param(
            params,
            "query_id",
            _inputs.query_ids(self.session, query, many=many),
            "query",
        )
        _inputs.with_param(params, "design_id", _inputs.design_id(design), "design")
        return params

    def score(
        self, sequences: list[str] | list[bytes], query: Any = None, **params: Any
    ) -> futures.LogLikelihoodFuture:
        """Log-likelihoods of sequences on the query's backbone (``batch/loglikelihood``)."""
        sequences = _prepared(sequences)
        return super().score(
            sequences,
            **self._on_structure(
                params, query, method=("score", "batch/loglikelihood")
            ),
        )

    def single_site(
        self, base_sequence: str | bytes | None = None, query: Any = None, **params: Any
    ) -> futures.SingleSiteScoreFuture:
        """Score every single-site mutant of ``base_sequence`` on the query's backbone.

        .. deprecated:: 0.18.0
            ``sequence=`` is removed in 0.19.0; pass ``base_sequence``.
        """
        if base_sequence is not None:
            params["base_sequence"] = base_sequence
        _renamed(params, "single_site", "sequence", "base_sequence")
        base_sequence = params.pop("base_sequence", None)
        if base_sequence is None:
            raise TypeError("single_site() missing argument: base_sequence")
        base_sequence = _text(base_sequence, "base_sequence")
        return cast(
            futures.SingleSiteScoreFuture,
            super().single_site(
                base_sequence,
                **self._on_structure(
                    params, query, method=("single_site", "single-site/*")
                ),
            ),
        )

    def generate(
        self,
        query: Any = None,
        design: Any = None,
        **params: Any,
    ) -> futures.SequenceGenerationFuture:
        """Design sequences for a backbone (``generate/sequences``).

        Pass ``query``, the backbone, or ``design``, an RFdiffusion or BoltzGen
        job or its id, in any state. A design's job runs once the design has
        finished, with ``num_samples`` per designed structure.

        .. deprecated:: 0.18.0
            ``seed=`` is removed in 0.19.0; pass ``random_seed``.
        """
        _renamed(params, "generate", "seed", "random_seed")
        _inputs.check_inputs(params, query=query, design=design)
        _inputs.check_one_structure_source(params, query, design)
        if all(
            value is None
            for value in (
                query,
                design,
                params.get("query_id"),
                params.get("design_id"),
            )
        ):
            raise TypeError("generate() needs a query or a design")
        self._method("generate", "generate/*")
        many = _inputs.takes_list(self._params_schema("generate/sequences"), "query_id")
        return cast(
            futures.SequenceGenerationFuture,
            super().generate(**self._on_structure(params, query, design, many=many)),
        )
