"""Fitted UMAPs, each served as a ``umap/<id>`` model."""

from __future__ import annotations

from typing import Any, Literal

from openprotein.base import APISession
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.jobs import Job, futures
from openprotein.jobs import _api as _jobs_api

from . import _api, _inputs
from .model import Model
from .schemas import ModelMetadata

__all__ = ["UMAPModel"]


_FAMILY = "umap"


class UMAPModel(Model):
    """A fitted UMAP; ``embed`` projects new sequences.

    Its id is its fit job's, so ``session.jobs.get`` on the bare id reads the
    fit set's ``sequences`` and ``embeddings``.
    """

    @property
    def spec(self) -> dict[str, Any] | None:
        """The fit's configuration: features, components, neighbors, distance."""
        return self.metadata.spec

    def delete(self) -> None:
        """Delete this fitted UMAP from the caller's model listing."""
        _api.delete_fitted(self.session, _FAMILY, self.id)

    def transform(
        self, data: list[str] | list[bytes] | AssayDataset
    ) -> futures.EmbeddingFuture:
        """Map sequences, or an assay's, through this UMAP."""
        return self.embed(data)


def _submit(session: APISession, request: dict[str, Any]) -> futures.UMAPFitFuture:
    """Submit a fit request; returns its fit future."""
    body = _api.create_fit(session, _FAMILY, request)
    return futures.UMAPFitFuture(
        session, Job.create(_jobs_api.operation_job(session, body, "id"))
    )


def _load(session: APISession, umap_id: str) -> UMAPModel:
    """A fitted UMAP by its id, bare or ``umap/<id>``."""
    card = ModelMetadata.model_validate(_api.get_fitted(session, _FAMILY, umap_id))
    return UMAPModel(session, card.model_id, card)


def _page(
    session: APISession, limit: int | None = None, offset: int | None = None
) -> list[UMAPModel]:
    """The caller's fitted UMAPs from every page, newest first."""
    cards = [
        ModelMetadata.model_validate(card)
        for card in _api.list_fitted(session, _FAMILY, limit, offset)
    ]
    return [UMAPModel(session, card.model_id, card) for card in cards]


def _fit(
    session: APISession,
    model: str | Model,
    sequences: list[str] | None = None,
    assay: AssayDataset | AssayMetadata | str | None = None,
    n_components: int | None = None,
    reduction: Literal["MEAN", "SUM"] | None = None,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    *,
    prompt_id: str | None = None,
    query_id: str | None = None,
    use_query_structure_in_decoder: bool | None = None,
    decoder_type: str | None = None,
) -> futures.UMAPFitFuture:
    """Fit a UMAP on ``model``'s features of ``sequences`` or an assay's.

    ``model`` is a model id, a handle from ``get_model`` or a fitted SVD.
    An embedding model needs ``reduction``. ``prompt_id``, ``query_id``,
    ``use_query_structure_in_decoder`` and ``decoder_type`` are its
    embedding params.
    """
    features = _inputs.model_id(model)
    _inputs.check_fit_source(sequences, assay)
    dataset = None
    if assay is not None:
        dataset = _inputs.as_dataset(session, assay)
        sequences = _inputs.assay_sequences(session, dataset)
    request: dict[str, Any] = {
        "model_id": features,
        "sequences": sequences,
    }
    for key, value in {
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "prompt_id": prompt_id,
        "reduction": reduction,
        "query_id": query_id,
        "use_query_structure_in_decoder": use_query_structure_in_decoder,
        "decoder_type": decoder_type,
    }.items():
        if value is not None:
            request[key] = value
    future = _submit(session, request)
    # the request carries the assay's sequences, not its id, so the job can't name it
    if dataset is not None:
        future.dataset = dataset
    return future
