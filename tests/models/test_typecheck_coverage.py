"""Future-returning handle methods have an executed static type assertion."""

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from openprotein import models

HANDLE_TYPES = (models.Model, models.GP, models.SVD, models.UMAP)

CONTRACTS = {
    ("GP", "fit"): "GPFitFuture",
    ("SVD", "fit"): "SVDFitFuture",
    ("SVD", "fit_transform"): "SVDFitFuture",
    ("UMAP", "fit"): "UMAPFitFuture",
    ("UMAP", "fit_transform"): "UMAPFitFuture",
    ("GPModel", "crossvalidate"): "CrossvalidateFuture",
    ("GPModel", "single_site"): "SingleSitePredictionFuture",
    ("AlphaFlow", "sample"): "StructurePredictionFuture",
    ("BioEmu", "sample"): "StructurePredictionFuture",
    ("ESMFlow", "sample"): "StructurePredictionFuture",
    ("Model", "attn"): "AttentionFuture",
    ("Model", "embed"): "EmbeddingFuture",
    ("Model", "fold"): "StructurePredictionFuture",
    (
        "Model",
        "generate",
    ): "SequenceGenerationFuture | StructureDesignFuture | Future",
    ("Model", "indel"): "IndelScoreFuture",
    ("Model", "logits"): "LogitsFuture",
    ("Model", "predict"): "PredictionFuture",
    ("Model", "score"): "LogLikelihoodFuture",
    (
        "Model",
        "single_site",
    ): "SingleSitePredictionFuture | SingleSiteScoreFuture | Future",
    ("PoET", "embed"): "EmbeddingFuture",
    ("PoET", "fit_gp"): "GPFitFuture",
    ("PoET", "fit_svd"): "SVDFitFuture",
    ("PoET", "fit_umap"): "UMAPFitFuture",
    ("PoET", "generate"): "SequenceGenerationFuture",
    ("PoET", "indel"): "IndelScoreFuture",
    ("PoET", "logits"): "LogitsFuture",
    ("PoET", "score"): "LogLikelihoodFuture",
    ("PoET", "single_site"): "SingleSiteScoreFuture",
    ("PoET2", "attn"): "AttentionFuture",
    ("PoET2", "embed"): "EmbeddingFuture",
    ("PoET2", "fit_gp"): "GPFitFuture",
    ("PoET2", "fit_svd"): "SVDFitFuture",
    ("PoET2", "fit_umap"): "UMAPFitFuture",
    ("PoET2", "generate"): "SequenceGenerationFuture",
    ("PoET2", "logits"): "LogitsFuture",
    ("PoET2", "score"): "LogLikelihoodFuture",
    ("PoET2", "single_site"): "SingleSiteScoreFuture",
    ("ProteinLanguageModel", "embed"): "EmbeddingFuture",
    ("ProteinLanguageModel", "fit_gp"): "GPFitFuture",
    ("ProteinLanguageModel", "fit_svd"): "SVDFitFuture",
    ("ProteinLanguageModel", "fit_umap"): "UMAPFitFuture",
    ("ProteinLanguageModel", "single_site"): "SingleSiteScoreFuture",
    ("ProteinMPNN", "embed"): "EmbeddingFuture",
    ("SVDModel", "fit_gp"): "GPFitFuture",
    ("SVDModel", "fit_umap"): "UMAPFitFuture",
    ("SVDModel", "transform"): "EmbeddingFuture",
    ("StructureGenerativeModel", "generate"): "StructureDesignFuture",
    ("StructurePredictionModel", "fold"): "StructurePredictionFuture",
    ("UMAPModel", "transform"): "EmbeddingFuture",
    ("_InverseFoldingModel", "generate"): "SequenceGenerationFuture",
    ("_InverseFoldingModel", "score"): "LogLikelihoodFuture",
    ("_InverseFoldingModel", "single_site"): "SingleSiteScoreFuture",
}


def _type_names(annotation: ast.expr) -> frozenset[str]:
    return frozenset(n.id for n in ast.walk(annotation) if isinstance(n, ast.Name))


def _checked_methods(source: str) -> dict[tuple[str, str], frozenset[str]]:
    tree = ast.parse(source)
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            module = importlib.import_module(node.module)
            for alias in node.names:
                cls = getattr(module, alias.name)
                if inspect.isclass(cls) and issubclass(cls, HANDLE_TYPES):
                    classes[alias.asname or alias.name] = cls
    checked = {}
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        handles = {
            arg.arg: classes[arg.annotation.id]
            for arg in function.args.args
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id in classes
        }
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "assert_type"
                and len(node.args) == 2
            ):
                continue
            call, expected = node.args
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id in handles
            ):
                continue
            method = getattr(handles[call.func.value.id], call.func.attr)
            owner, name = method.__qualname__.rsplit(".", 1)
            checked[owner, name] = _type_names(expected)
    return checked


def test_every_future_returning_handle_method_is_type_checked():
    source = Path(__file__).with_name("typecheck_task_futures.py").read_text()
    checked = _checked_methods(source)
    expected = {
        key: _type_names(ast.parse(value, mode="eval").body)
        for key, value in CONTRACTS.items()
    }
    assert {key: checked.get(key) for key in expected} == expected


def test_exported_handles_keep_their_inherited_method_contracts():
    for name in models.__all__:
        cls = getattr(models, name)
        if not inspect.isclass(cls) or not issubclass(cls, HANDLE_TYPES):
            continue
        for method in {method for _, method in CONTRACTS}:
            owner = next(
                (
                    base.__name__
                    for base in cls.__mro__
                    if (base.__name__, method) in CONTRACTS
                ),
                None,
            )
            if owner is not None:
                assert getattr(cls, method).__qualname__ == f"{owner}.{method}"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("pass", {}),
        ("pass  # Model.embed", {}),
        (
            "assert_type(model.embed(['MKT']), EmbeddingFuture)",
            {("Model", "embed"): frozenset({"EmbeddingFuture"})},
        ),
        (
            "assert_type(model.embed(['MKT']), Future)",
            {("Model", "embed"): frozenset({"Future"})},
        ),
    ],
)
def test_typecheck_coverage_reads_assertions_and_their_expected_types(body, expected):
    source = (
        f"from openprotein.models import Model\ndef check(model: Model):\n    {body}\n"
    )
    assert _checked_methods(source) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("pass", {}),
        ("pass  # SVD.fit", {}),
        (
            "assert_type(estimator.fit(), SVDFitFuture)",
            {("SVD", "fit"): frozenset({"SVDFitFuture"})},
        ),
        (
            "assert_type(estimator.fit(), Future)",
            {("SVD", "fit"): frozenset({"Future"})},
        ),
    ],
)
def test_estimator_coverage_reads_assertions_and_their_expected_types(body, expected):
    source = (
        f"from openprotein.models import SVD\ndef check(estimator: SVD):\n    {body}\n"
    )
    assert _checked_methods(source) == expected
