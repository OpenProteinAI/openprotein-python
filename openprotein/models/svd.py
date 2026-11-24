"""SVDs, each served as an ``svd/<id>`` model."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from openprotein.base import APISession
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.jobs import Job, futures
from openprotein.jobs import _api as _jobs_api

from . import _api, _inputs
from .model import Model, _texts
from .schemas import ModelMetadata

__all__ = ["SVDEmbeddingSource", "SVDModel"]


_FAMILY = "svd"


class SVDEmbeddingSource(BaseModel):
    """The embedding method reused for an SVD's projections.

    Attributes
    ----------
    model_id : str or None
        Model used to embed the fit's sequences.
    version : str or None
        Version of that embedding method.
    reduction : str or None
        Reduction applied to each sequence's embeddings.
    prompt_id : str or None
        Prompt used to condition the embedding method.
    """

    model_id: str | None = None
    version: str | None = None
    reduction: str | None = None
    prompt_id: str | None = None


class _SVDRecord(BaseModel):
    """The fitted SVD description on its model card.

    Attributes
    ----------
    id : str
        Identifier of the fitted SVD.
    owner : str or None
        Owner identifier of the fitted model.
    requested_n_components : int or None
        Component count requested for the fit.
    n_components : int or None
        Component count of the fitted projection.
    embedding : SVDEmbeddingSource or None
        Embedding method used for fitting and projection.
    name : str or None
        Display name of the fitted model.
    description : str or None
        Description of the fitted model.
    created_at : datetime or None
        Creation time of the fitted model.
    """

    id: str
    owner: str | None = None
    requested_n_components: int | None = None
    n_components: int | None = None
    embedding: SVDEmbeddingSource | None = None
    name: str | None = None
    description: str | None = None
    created_at: datetime | None = None


def _record(card: dict[str, Any]) -> _SVDRecord:
    return _SVDRecord.model_validate(card["svd"])


def _get(session: APISession, svd_id: str) -> _SVDRecord:
    """The live SVD's record; raises a 404 unless the SVD is live."""
    return _record(_api.get_fitted(session, _FAMILY, svd_id))


class SVDModel(Model):
    """A fitted SVD for projecting sequences through ``transform`` or ``embed``.

    Its fit configuration is read from the model card on first use.
    """

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        bare = _api.bare_id(_FAMILY, model_id)
        super().__init__(session, f"{_FAMILY}/{bare}", metadata)
        self._record: _SVDRecord | None = None

    @classmethod
    def _from_record(cls, session: APISession, record: _SVDRecord | dict) -> SVDModel:
        parsed = (
            _SVDRecord.model_validate(record) if isinstance(record, dict) else record
        )
        model = cls(session, parsed.id)
        model._record = parsed
        return model

    @property
    def _svd_record(self) -> _SVDRecord:
        if self._record is None:
            self._record = _get(self.session, self.id)
        return self._record

    @property
    def embedding(self) -> SVDEmbeddingSource | None:
        """The embedding source reused when projecting new sequences."""
        return self._svd_record.embedding

    @property
    def requested_n_components(self) -> int | None:
        """The requested component count; ``n_components`` is the fitted count."""
        return self._svd_record.requested_n_components

    @property
    def n_components(self) -> int | None:
        """Rank of the fitted basis."""
        return self._svd_record.n_components

    def transform(
        self, data: list[str] | list[bytes] | AssayDataset
    ) -> futures.EmbeddingFuture:
        """Project sequences, or an assay's, through this basis."""
        return self.embed(data)

    def fit_umap(
        self,
        sequences: list[str] | list[bytes] | None = None,
        assay: AssayDataset | AssayMetadata | str | None = None,
        n_components: int | None = None,
        n_neighbors: int | None = None,
        min_dist: float | None = None,
    ) -> futures.UMAPFitFuture:
        """Fit a UMAP on this SVD's projections of ``sequences`` or an assay's."""
        from .umap import _fit

        return _fit(
            self.session,
            self,
            sequences=None if sequences is None else _texts(sequences),
            assay=assay,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )

    def fit_gp(
        self,
        assay: AssayDataset | AssayMetadata | str,
        properties: list[str],
        name: str | None = None,
        description: str | None = None,
        kernel: str = "rbf",
        period: float | None = None,
        alpha: float | None = None,
    ) -> futures.GPFitFuture:
        """Train a Gaussian process on an assay's properties, featurized by this SVD."""
        from .gp import _fit

        return _fit(
            self.session,
            self,
            assay=assay,
            properties=properties,
            name=name,
            description=description,
            kernel=kernel,
            period=period,
            alpha=alpha,
        )

    def delete(self) -> None:
        """Delete this fitted SVD from the caller's model listing."""
        _api.delete_fitted(self.session, _FAMILY, self.id)


def _submit(session: APISession, request: dict[str, Any]) -> futures.SVDFitFuture:
    """Submit a fit request; returns its fit future."""
    body = _api.create_fit(session, _FAMILY, request)
    return futures.SVDFitFuture(
        session, Job.create(_jobs_api.operation_job(session, body, "id"))
    )


def _load(session: APISession, svd_id: str) -> SVDModel:
    """A fitted SVD by its id, bare or ``svd/<id>``; raises a 404 unless it is live."""
    return SVDModel._from_record(session, _get(session, svd_id))


def _page(
    session: APISession, limit: int | None = None, offset: int | None = None
) -> list[SVDModel]:
    """The caller's fitted SVDs from every page, newest first."""
    cards = _api.list_fitted(session, _FAMILY, limit, offset)
    return [
        SVDModel._from_record(session, _record(card))
        for card in cards
        if card.get("svd") is not None
    ]


def _fit(
    session: APISession,
    model: str | Model,
    sequences: list[str] | None = None,
    assay: AssayDataset | AssayMetadata | str | None = None,
    n_components: int | None = None,
    reduction: Literal["MEAN", "SUM"] | None = None,
    *,
    prompt_id: str | None = None,
    query_id: str | None = None,
    use_query_structure_in_decoder: bool | None = None,
    decoder_type: str | None = None,
    seed: int | None = None,
) -> futures.SVDFitFuture:
    """Fit an SVD on ``model``'s embeddings of ``sequences`` or an ``assay``'s.

    ``model`` is a model id or a handle from ``get_model``. ``prompt_id``,
    ``query_id``, ``use_query_structure_in_decoder`` and ``decoder_type`` are
    its embedding params.
    """
    request: dict[str, Any] = {
        "model_id": _inputs.model_id(model),
        "output": "embeddings",
    }
    for key, value in {
        "n_components": n_components,
        "sequences": sequences,
        "assay_id": None if assay is None else _inputs.assay_id(assay),
        "prompt_id": prompt_id,
        "reduction": reduction,
        "query_id": query_id,
        "use_query_structure_in_decoder": use_query_structure_in_decoder,
        "decoder_type": decoder_type,
        "seed": seed,
    }.items():
        if value is not None:
            request[key] = value
    future = _submit(session, request)
    if isinstance(assay, (AssayDataset, AssayMetadata)):
        future.dataset = _inputs.as_dataset(session, assay)
    return future
