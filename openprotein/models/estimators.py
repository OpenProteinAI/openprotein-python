"""Estimators on ``session.models``: fit on an embeddings job or on a model's embeddings."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar, cast, overload

from typing_extensions import Self

from openprotein.base import APISession
from openprotein.data import AssayMetadata
from openprotein.data.assaydataset import AssayDataset
from openprotein.jobs import Future
from openprotein.jobs.futures import (
    FitFuture,
    GPFitFuture,
    SVDFitFuture,
    UMAPFitFuture,
)

from . import _api, _inputs, gp, svd, umap
from .gp import GPModel
from .model import Model, _texts
from .svd import SVDModel
from .umap import UMAPModel

__all__ = ["GP", "SVD", "UMAP"]


_F = TypeVar("_F", bound=FitFuture)
_H = TypeVar("_H")


def _describe(x: Any) -> str:
    """How an error names ``x``."""
    if isinstance(x, AssayDataset):
        return f"assay {x.id}"
    if isinstance(x, Model):
        return f"model {x.id}"
    if isinstance(x, Future):
        return f"job {x.id}"
    return type(x).__name__


def _job_id(X: Any) -> str:
    """The id ``X`` is sent as ``embeddings_job_id``: a job's, or a fitted SVD's."""
    if isinstance(X, str):
        return _api.bare_id("svd", X)
    if isinstance(X, SVDModel):
        return _api.bare_id("svd", X.id)
    if isinstance(X, Future):
        return X.id
    raise TypeError(
        f"{_describe(X)} isn't an embeddings job; X is a job, a fitted SVD or its "
        "id. Pass a model as model= and an assay as assay="
    )


def _check_source(
    X: Any, model: Any, sequences: Any, assay: Any, params: dict[str, Any]
) -> None:
    """Refuse a fit on both or neither of ``X`` and ``model``, or on inputs it can't take."""
    if (X is None) == (model is None):
        raise TypeError("pass X, an embeddings job, or model= with sequences or assay")
    if X is not None and (sequences is not None or assay is not None or params):
        raise TypeError(
            "X names the sequences it embedded; sequences, assay and embedding "
            "params go with model="
        )
    if model is not None and (sequences is None) == (assay is None):
        raise ValueError("pass one of sequences or assay with model=")


def _feature_model(session: APISession, model: Any, params: dict[str, Any]) -> Model:
    """The handle ``model=`` names; a handle is used as is, an id is looked up."""
    if "model_id" in params:
        raise TypeError("model_id is not an embedding param; pass the model as model=")
    if isinstance(model, str):
        from .models import ModelsAPI

        return ModelsAPI(session).get_model(model)
    _inputs.model_id(model)
    return model


def _carried(x: Any) -> AssayDataset | Future | None:
    """What a fit on ``x`` takes its ``dataset`` from; a future is read when asked."""
    return x if isinstance(x, Future) else _dataset_of(x)


def _dataset_of(x: Any) -> AssayDataset | None:
    """The assay ``x`` is, or the one its sequences came from."""
    if isinstance(x, AssayDataset):
        return x
    dataset = getattr(x, "dataset", None)
    return dataset if isinstance(dataset, AssayDataset) else None


class _Estimator(Generic[_F, _H]):
    """Hyperparameters to fit with.

    ``session.models`` hands out each estimator class bound to its session.
    Calling it makes an estimator, and ``id=`` alone loads a fitted model.
    ``fit`` returns the fit's future, whose ``wait`` returns the fitted model.
    """

    session: ClassVar[APISession]

    @classmethod
    def _bind(cls, session: APISession) -> type[Self]:
        """This estimator class, fitting and loading through ``session``."""
        attrs = {
            "session": session,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__doc__": cls.__doc__,
        }
        return cast("type[Self]", type(cls.__name__, (cls,), attrs))

    @classmethod
    def _bound(cls) -> APISession:
        session = getattr(cls, "session", None)
        if session is None:
            raise TypeError(
                f"{cls.__name__} needs a session; use session.models.{cls.__name__}"
            )
        return session

    @classmethod
    def _create(
        cls, id: str | None, given: bool, load: Callable[[APISession, str], _H]
    ) -> Self | _H:
        """A new estimator, or with ``id`` the fitted model ``load`` reads."""
        session = cls._bound()
        if id is None:
            return object.__new__(cls)
        if given:
            raise TypeError("pass id alone to load a fitted model")
        return load(session, id)


class SVD(_Estimator[SVDFitFuture, SVDModel]):
    """An SVD to fit on a system model's embeddings."""

    @overload
    def __new__(cls, *, id: str) -> SVDModel: ...

    @overload
    def __new__(
        cls, n_components: int | None = None, seed: int | None = None
    ) -> Self: ...

    def __new__(cls, *args: Any, id: str | None = None, **kwargs: Any) -> Any:
        return cls._create(id, bool(args or kwargs), svd._load)

    def __init__(self, n_components: int | None = None, seed: int | None = None):
        self.n_components = n_components
        self.seed = seed

    def fit(
        self,
        X: Any = None,
        *,
        model: Any = None,
        sequences: builtins.list[str] | builtins.list[bytes] | None = None,
        assay: Any = None,
        **params: Any,
    ) -> SVDFitFuture:
        """Fit on ``X``, an embeddings job or a fitted SVD, and the sequences it embedded.

        Or fit on ``model``'s embeddings of ``sequences`` or an ``assay``'s,
        with ``params`` its embedding params such as ``reduction``.
        """
        _check_source(X, model, sequences, assay, params)
        if model is not None:
            return svd._fit(
                self.session,
                _feature_model(self.session, model, params),
                sequences=None if sequences is None else _texts(sequences),
                assay=assay,
                n_components=self.n_components,
                seed=self.seed,
                **params,
            )
        request: dict[str, Any] = {
            "embeddings_job_id": _job_id(X),
            "output": "embeddings",
        }
        for key, value in (("n_components", self.n_components), ("seed", self.seed)):
            if value is not None:
                request[key] = value
        future = svd._submit(self.session, request)
        future.dataset = _carried(X)
        return future

    def fit_transform(self, X: Any = None, **kwargs: Any) -> SVDFitFuture:
        """``fit``; a fit that names an ``embeddings`` output holds its sequences projected."""
        return self.fit(X, **kwargs)

    @classmethod
    def list(
        cls, limit: int | None = None, offset: int | None = None
    ) -> builtins.list[SVDModel]:
        """The caller's SVDs from every page, newest first, at most ``limit``."""
        return svd._page(cls._bound(), limit, offset)


class UMAP(_Estimator[UMAPFitFuture, UMAPModel]):
    """A UMAP to fit on pooled embeddings or an SVD's projections."""

    @overload
    def __new__(cls, *, id: str) -> UMAPModel: ...

    @overload
    def __new__(
        cls,
        n_components: int | None = None,
        n_neighbors: int | None = None,
        min_dist: float | None = None,
    ) -> Self: ...

    def __new__(cls, *args: Any, id: str | None = None, **kwargs: Any) -> Any:
        return cls._create(id, bool(args or kwargs), umap._load)

    def __init__(
        self,
        n_components: int | None = None,
        n_neighbors: int | None = None,
        min_dist: float | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def fit(
        self,
        X: Any = None,
        *,
        model: Any = None,
        sequences: builtins.list[str] | builtins.list[bytes] | None = None,
        assay: Any = None,
        **params: Any,
    ) -> UMAPFitFuture:
        """Fit on ``X``, an embeddings job or a fitted SVD, and the sequences it embedded.

        Or fit on ``model``'s features of ``sequences`` or an ``assay``'s, with
        ``params`` its embedding params such as ``reduction``. ``model`` is a
        system model or a fitted SVD.
        """
        _check_source(X, model, sequences, assay, params)
        if model is not None:
            future = umap._fit(
                self.session,
                _feature_model(self.session, model, params),
                sequences=None if sequences is None else _texts(sequences),
                assay=assay,
                n_components=self.n_components,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                **params,
            )
            return future
        request: dict[str, Any] = {"embeddings_job_id": _job_id(X)}
        for key, value in (
            ("n_components", self.n_components),
            ("n_neighbors", self.n_neighbors),
            ("min_dist", self.min_dist),
        ):
            if value is not None:
                request[key] = value
        future = umap._submit(self.session, request)
        future.dataset = _carried(X)
        return future

    def fit_transform(self, X: Any = None, **kwargs: Any) -> UMAPFitFuture:
        """``fit``; the fit's outputs are its sequences mapped."""
        return self.fit(X, **kwargs)

    @classmethod
    def list(
        cls, limit: int | None = None, offset: int | None = None
    ) -> builtins.list[UMAPModel]:
        """The caller's UMAPs from every page, newest first, at most ``limit``."""
        return umap._page(cls._bound(), limit, offset)


def _training(
    session: APISession, X: Any, assay: Any, property: str | None
) -> tuple[str, str]:
    """The assay id and property a GP trains on; reads the assay only to pick its property."""
    if assay is None:
        dataset = _dataset_of(X)
    elif property is not None:
        return _inputs.assay_id(assay), property
    else:
        dataset = _inputs.as_dataset(session, assay)
    if dataset is None:
        raise ValueError(f"{_describe(X)} carries no assay; pass assay to train on")
    if property is not None:
        return dataset.id, property
    names = list(dataset.measurement_names)
    if len(names) != 1:
        raise ValueError(
            f"assay {dataset.id} has measurements {', '.join(names)}; "
            "pass property to pick one"
        )
    return dataset.id, names[0]


class GP(_Estimator[GPFitFuture, GPModel]):
    """A Gaussian process to train on one assay property."""

    @overload
    def __new__(cls, *, id: str) -> GPModel: ...

    @overload
    def __new__(
        cls,
        kernel: str = "rbf",
        period: float | None = None,
        alpha: float | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> Self: ...

    def __new__(cls, *args: Any, id: str | None = None, **kwargs: Any) -> Any:
        return cls._create(id, bool(args or kwargs), gp._load)

    def __init__(
        self,
        kernel: str = "rbf",
        period: float | None = None,
        alpha: float | None = None,
        name: str | None = None,
        description: str | None = None,
    ):
        self.kernel = kernel
        self.period = period
        self.alpha = alpha
        self.name = name
        self.description = description

    def fit(
        self,
        X: Any = None,
        *,
        model: Any = None,
        assay: Any = None,
        property: str | None = None,
        **params: Any,
    ) -> GPFitFuture:
        """Train on one assay ``property``.

        The features are ``X``, an embeddings job or a fitted SVD, on the assay
        ``X`` embedded unless ``assay`` is given. Or ``model``'s features of
        ``assay``, with ``params`` its embedding params such as ``reduction``.
        Or, with ``assay`` alone, one-hot features. ``property`` may be omitted
        when the assay has one measurement.
        """
        if X is not None and model is not None:
            raise TypeError("pass X or model=, not both")
        if params and model is None:
            raise TypeError("embedding params go with model=")
        if X is None and assay is None:
            raise ValueError("pass X, or assay to train on")
        feature_model = (
            None if model is None else _feature_model(self.session, model, params)
        )
        trained_on, trained = _training(self.session, X, assay, property)
        kernel: dict[str, Any] = {"type": self.kernel}
        for key, value in (("period", self.period), ("alpha", self.alpha)):
            if value is not None:
                kernel[key] = value
        request: dict[str, Any] = {
            "kernel": kernel,
            "training_dataset": {"assay_id": trained_on, "properties": [trained]},
        }
        if X is not None:
            request["embeddings_job_id"] = _job_id(X)
        elif feature_model is not None:
            request["features"] = {"model_id": feature_model.id, **params}
        else:
            request["features"] = {}
        for key, value in (("name", self.name), ("description", self.description)):
            if value is not None:
                request[key] = value
        future = gp._submit(self.session, request)
        if assay is None:
            future.dataset = _carried(X)
        elif isinstance(assay, (AssayDataset, AssayMetadata)):
            future.dataset = _inputs.as_dataset(self.session, assay)
        return future

    @classmethod
    def list(
        cls, limit: int | None = None, offset: int | None = None
    ) -> builtins.list[GPModel]:
        """The caller's GPs from every page, newest first, at most ``limit``."""
        return gp._page(cls._bound(), limit, offset)
