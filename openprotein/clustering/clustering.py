"""ClusteringAPI — user-facing entry point for clustering jobs."""

import typing

from openprotein.base import APISession
from openprotein.common import Feature, FeatureType, Reduction, ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.embeddings import EmbeddingModel, EmbeddingsAPI
from openprotein.errors import InvalidParameterError
from openprotein.svd import SVDAPI, SVDModel

from . import api
from .models import HierarchicalClusteringFuture
from .schemas import LinkageMethod, Metric


class ClusteringAPI:
    """Top-level clustering API. Use `session.clustering.hierarchical(...)` to
    fit a hierarchical clustering job on a sequence set."""

    def __init__(self, session: APISession):
        self.session = session

    @typing.overload
    def hierarchical(
        self,
        model: EmbeddingModel,
        reduction: Reduction | ReductionType,
        feature_type: FeatureType = FeatureType.PLM,
        linkage_method: LinkageMethod | str = LinkageMethod.WARD,
        metric: Metric | str = Metric.EUCLIDEAN,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | AssayMetadata | str | None = None,
        **kwargs,
    ) -> HierarchicalClusteringFuture: ...

    @typing.overload
    def hierarchical(
        self,
        model: SVDModel,
        reduction: None = None,
        feature_type: FeatureType = FeatureType.SVD,
        linkage_method: LinkageMethod | str = LinkageMethod.WARD,
        metric: Metric | str = Metric.EUCLIDEAN,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | AssayMetadata | str | None = None,
        **kwargs,
    ) -> HierarchicalClusteringFuture: ...

    def hierarchical(
        self,
        model: EmbeddingModel | SVDModel | str,
        reduction: Reduction | ReductionType | None = None,
        feature_type: Feature | FeatureType | None = None,
        linkage_method: LinkageMethod | str = LinkageMethod.WARD,
        metric: Metric | str = Metric.EUCLIDEAN,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | AssayMetadata | str | None = None,
        **kwargs,
    ) -> HierarchicalClusteringFuture:
        """Fit a hierarchical clustering on sequences, returning a future
        that resolves to a HierarchicalClusteringResult (scipy linkage + leaf order)."""
        # resolve assay_id
        assay_id = (
            assay.assay_id
            if isinstance(assay, AssayMetadata)
            else assay.id
            if isinstance(assay, AssayDataset)
            else assay
        )
        if sequences is not None and assay_id is not None:
            raise InvalidParameterError(
                "Expected only either sequences or assay, not both"
            )
        if sequences is not None:
            n = len(sequences)
            if n < 2:
                raise InvalidParameterError(
                    f"clustering requires at least 2 sequences, got {n}"
                )
            if n > 10000:
                raise InvalidParameterError(
                    f"clustering size cap exceeded: N={n} > 10000"
                )
        # infer feature_type from model type
        feature_type = (
            FeatureType.PLM
            if isinstance(model, EmbeddingModel)
            else FeatureType.SVD
            if isinstance(model, SVDModel)
            else feature_type
        )
        if feature_type is None:
            raise InvalidParameterError(
                "Expected feature_type to be provided if passing str model_id as model"
            )
        if isinstance(feature_type, str):
            feature_type = FeatureType(feature_type)
        if isinstance(reduction, str):
            reduction = ReductionType(reduction)
        # combo validation — mirrors the Go server rule
        _WARD_ONLY_METRICS = {"ward", "centroid", "median"}
        lm_val = (
            linkage_method.value
            if isinstance(linkage_method, LinkageMethod)
            else str(linkage_method)
        )
        m_val = metric.value if isinstance(metric, Metric) else str(metric)
        if lm_val in _WARD_ONLY_METRICS and m_val != "euclidean":
            raise InvalidParameterError(
                f"linkage_method={lm_val!r} requires metric='euclidean', got {m_val!r}"
            )
        # resolve model_id and svd_id
        svd_id: str | None = None
        if feature_type == FeatureType.PLM:
            if reduction is None:
                raise InvalidParameterError(
                    "Expected reduction when using PLM feature_type"
                )
            if isinstance(model, str):
                embeddings_api = getattr(self.session, "embedding", None)
                assert isinstance(embeddings_api, EmbeddingsAPI)
                model = embeddings_api.get_model(model)
            assert isinstance(model, EmbeddingModel), "Expected EmbeddingModel"
            model_id = model.id
        elif feature_type == FeatureType.SVD:
            if reduction is not None:
                raise InvalidParameterError(
                    "Unexpected reduction when using SVD feature_type"
                )
            if isinstance(model, str):
                svd_api = getattr(self.session, "svd", None)
                assert isinstance(svd_api, SVDAPI)
                model = svd_api.get_svd(model)
            assert isinstance(model, SVDModel), "Expected SVDModel"
            model_id = model.id
            svd_id = model.id
        else:
            raise InvalidParameterError(f"Unsupported feature_type: {feature_type}")

        linkage_method = (
            linkage_method.value
            if isinstance(linkage_method, LinkageMethod)
            else str(linkage_method)
        )
        metric = metric.value if isinstance(metric, Metric) else str(metric)
        reduction_str = (
            reduction.value if isinstance(reduction, ReductionType) else reduction
        )

        # Advanced flags such as `force_recompute` are accepted via **kwargs
        # only (intentionally kept out of the typed signature / autocomplete):
        # they bypass the backend result cache and are easy to misuse.
        job = api.clustering_hierarchical_post(
            session=self.session,
            model_id=model_id,
            feature_type=feature_type.value,
            linkage_method=linkage_method,
            metric=metric,
            sequences=sequences,
            assay_id=assay_id,
            reduction=reduction_str,
            svd_id=svd_id,
            **kwargs,
        )
        return HierarchicalClusteringFuture(session=self.session, job=job)

    def get(self, clustering_id: str) -> HierarchicalClusteringFuture:
        """Fetch a clustering job by ID."""
        # Single-method dispatch today; when new methods are added,
        # branch on metadata.method to pick the concrete future type.
        metadata = api.clustering_get(self.session, clustering_id)
        return HierarchicalClusteringFuture(session=self.session, metadata=metadata)

    def list(
        self,
        method: str | None = None,
        page_size: int | None = None,
        page_offset: int | None = None,
    ) -> list[HierarchicalClusteringFuture]:
        """List clustering jobs, optionally filtering by method.

        Pass `page_size` / `page_offset` to paginate."""
        # Single-method dispatch today; when new methods are added,
        # branch on metadata.method to pick the concrete future type.
        return [
            HierarchicalClusteringFuture(session=self.session, metadata=md)
            for md in api.clustering_list_get(
                self.session,
                method=method,
                page_size=page_size,
                page_offset=page_offset,
            )
        ]
