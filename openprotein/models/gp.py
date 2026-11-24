"""Trained Gaussian processes, served as ``gp/<id>`` models."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import Any, Literal, cast
from uuid import uuid4

from pydantic import TypeAdapter

from openprotein.base import _IDEMPOTENCY_KEY_HEADER, APISession
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.jobs import Job, _submission, futures
from openprotein.jobs import _api as _jobs_api
from openprotein.models.kernels import (
    TrainedKernel,
    _GPHyperparameters,
    _sole_hyperparameters,
    _trained_kernel,
)

from . import _api, _folds, _inputs
from .model import Model
from .schemas import ModelMetadata

__all__ = ["GPModel"]


_FAMILY = "gp"


class GPModel(Model):
    """A Gaussian process predictor trained on an assay.

    ``predict`` and ``single_site`` submit predictions; the other methods read
    or start its cross-validation and read its training record.
    """

    @property
    def spec(self) -> dict[str, Any] | None:
        """The training configuration: features, kernel and dataset."""
        return self.metadata.spec

    def _path(self, *parts: str) -> str:
        return "/".join([_api.fitted_path(_FAMILY, self.id), *parts])

    def crossvalidate(
        self, n_splits: int | None = None, folds: Sequence[int] | None = None
    ) -> futures.CrossvalidateFuture:
        """Cross-validate on the training assay.

        Parameters
        ----------
        n_splits : int, optional
            Number of random folds, between 2 and 10 (default 5). Keeps rows sharing a
            sequence in the same fold. Mutually exclusive with ``folds``.
        folds : sequence of int, optional
            Test fold of each row of the training assay, one entry per row in assay
            order, labelled 0..K-1 with none empty and 2 <= K <= 10. Rows are assigned
            independently of which sequence they hold, unlike ``n_splits``. Use -1 to
            exclude a row, which then comes back with a fold of -1 and NaN predictions.
            Splitting a repeated sequence across folds leaks, since embeddings come
            from the sequence, so expect optimistic scores.

        Returns
        -------
        CrossvalidateFuture
            Future whose ``get`` and ``wait`` return the run's `CrossvalidateResults`.

        Raises
        ------
        InvalidParameterError
            If both ``n_splits`` and ``folds`` are given, or if ``folds`` does not
            have one valid entry per row of the training assay.
        """
        _folds.check_exclusive(n_splits, folds)
        if folds is not None:
            assert self.spec is not None
            assay = AssayDataset._for(self.session)(
                id=self.spec["training_dataset"]["assay_id"]
            )
            folds = _folds.validate_folds(folds, num_assay_rows=len(assay))
        body = {
            key: value
            for key, value in {"n_splits": n_splits, "folds": folds}.items()
            if value is not None
        }
        response = _submission.post_with_retry(
            self.session,
            self._path("crossvalidate"),
            json=body,
            headers={_IDEMPOTENCY_KEY_HEADER: str(uuid4())},
        )
        job = _jobs_api.operation_job(self.session, response.json(), "id")
        return futures.CrossvalidateFuture(self.session, Job.create(job), self)

    def single_site(
        self, base_sequence: str | bytes, **params: Any
    ) -> futures.SingleSitePredictionFuture:
        """Predict properties for the sequence's single-site substitutions."""
        return cast(
            futures.SingleSitePredictionFuture,
            super().single_site(base_sequence, **params),
        )

    def evaluations(self, evaluation_id: str | None = None) -> Any:
        """Every cross-validation run, newest first, or one run with its rows."""
        parts = (
            ["evaluations"] if evaluation_id is None else ["evaluations", evaluation_id]
        )
        return self.session.get(self._path(*parts)).json()

    def training(self) -> list[dict[str, Any]]:
        """The loss curves recorded while training, one per property."""
        return self.session.get(self._path("training")).json()

    @cached_property
    def _hyperparameters(self) -> _GPHyperparameters:
        served = self.session.get(
            self._path("hyperparameters"), timeout=self.session.timeout
        ).json()
        return _sole_hyperparameters(
            TypeAdapter(dict[str, _GPHyperparameters]).validate_python(served)
        )

    # IPython can evaluate this descriptor to complete the concrete kernel's fields.
    @cached_property
    def kernel(self) -> TrainedKernel:
        """
        The trained kernel, in data space.

        The first access, including runtime tab completion, may make a request.
        Failed reads are retried on subsequent access.

        Its class follows the kernel type and carries only the hyperparameters that
        apply to it: every kernel has an ``outputscale``, every kernel but linear a
        ``lengthscale``, periodic a ``period`` and rational quadratic an ``alpha``.

        Raises
        ------
        HTTPError
            404 for a GP trained before hyperparameters were recorded.

        Examples
        --------
        >>> trained.kernel
        PeriodicKernel(outputscale=1.92, lengthscale=0.83, period=2.31)
        >>> trained.kernel.lengthscale
        0.83
        """
        hyperparameters = self._hyperparameters
        assert self.spec is not None
        try:
            return _trained_kernel(self.spec["kernel"]["type"], hyperparameters)
        except ValueError:
            self.__dict__.pop("_hyperparameters", None)
            raise

    @property
    def mean(self) -> float | None:
        """
        Constant mean of the GP prior, in the units of the trained property.

        Raises
        ------
        HTTPError
            404 for a GP trained before hyperparameters were recorded.
        """
        return self._hyperparameters.mean

    @property
    def noise(self) -> float | None:
        """
        Observation noise variance, in the units of the trained property squared.

        Raises
        ------
        HTTPError
            404 for a GP trained before hyperparameters were recorded.
        """
        return self._hyperparameters.noise

    def delete(self) -> None:
        """Delete this trained Gaussian process from the caller's model listing."""
        _api.delete_fitted(self.session, _FAMILY, self.id)


def _submit(session: APISession, request: dict[str, Any]) -> futures.GPFitFuture:
    """Submit a training request; returns its fit future."""
    body = _api.create_fit(session, _FAMILY, request)
    return futures.GPFitFuture(
        session, Job.create(_jobs_api.operation_job(session, body, "id"))
    )


def _load(session: APISession, gp_id: str) -> GPModel:
    """A trained GP by its id, bare or ``gp/<id>``."""
    card = ModelMetadata.model_validate(_api.get_fitted(session, _FAMILY, gp_id))
    return GPModel(session, card.model_id, card)


def _page(
    session: APISession, limit: int | None = None, offset: int | None = None
) -> list[GPModel]:
    """The caller's GPs from every page, newest first."""
    cards = [
        ModelMetadata.model_validate(card)
        for card in _api.list_fitted(session, _FAMILY, limit, offset)
    ]
    return [GPModel(session, card.model_id, card) for card in cards]


def _fit(
    session: APISession,
    model: str | Model,
    assay: AssayDataset | AssayMetadata | str,
    properties: list[str],
    reduction: Literal["MEAN", "SUM"] | None = None,
    *,
    kernel: str = "rbf",
    period: float | None = None,
    alpha: float | None = None,
    name: str | None = None,
    description: str | None = None,
    prompt_id: str | None = None,
    query_id: str | None = None,
    use_query_structure_in_decoder: bool | None = None,
    decoder_type: str | None = None,
) -> futures.GPFitFuture:
    """Train a Gaussian process on an assay's properties.

    ``model`` featurizes the sequences: a model id, a handle from
    ``get_model`` or a fitted SVD. An embedding model needs ``reduction``.
    ``prompt_id``, ``query_id``, ``use_query_structure_in_decoder`` and
    ``decoder_type`` are its embedding params. ``period`` is for the
    periodic kernel, ``alpha`` for rational_quadratic.
    """
    feature_model = _inputs.model_id(model)
    features: dict[str, Any] = {"model_id": feature_model}
    for key, value in {
        "reduction": reduction,
        "prompt_id": prompt_id,
        "query_id": query_id,
        "use_query_structure_in_decoder": use_query_structure_in_decoder,
        "decoder_type": decoder_type,
    }.items():
        if value is not None:
            features[key] = value
    kernel_spec: dict[str, Any] = {"type": kernel}
    if period is not None:
        kernel_spec["period"] = period
    if alpha is not None:
        kernel_spec["alpha"] = alpha
    request: dict[str, Any] = {
        "features": features,
        "kernel": kernel_spec,
        "training_dataset": {
            "assay_id": _inputs.assay_id(assay),
            "properties": properties,
        },
    }
    if name is not None:
        request["name"] = name
    if description is not None:
        request["description"] = description
    future = _submit(session, request)
    if isinstance(assay, (AssayDataset, AssayMetadata)):
        future.dataset = _inputs.as_dataset(session, assay)
    return future
