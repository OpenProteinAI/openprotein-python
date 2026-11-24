"""Where the job machinery shared by models and data jobs lives."""

import ast
import importlib
import subprocess
import sys
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from openprotein import jobs, models
from openprotein.jobs import Future, Job, UnknownJobFuture


@pytest.mark.parametrize(
    "module",
    [
        "openprotein.jobs",
        "openprotein.jobs.schemas",
        "openprotein.models",
        "openprotein.models.schemas",
    ],
)
@pytest.mark.parametrize("name", ["ModelsJob", "MODELS_JOB_TYPE"])
def test_models_jobs_have_no_schema_of_their_own(module, name):
    assert not hasattr(importlib.import_module(module), name)


def _payload(job_type: str) -> dict[str, Any]:
    return {
        "job_id": "m-1",
        "job_type": job_type,
        "status": "SUCCESS",
        "created_date": datetime(2026, 9, 10, 12, 0, 0).isoformat(),
        "args": {"model_id": "poet-2"},
    }


@pytest.mark.parametrize(
    ("job_type", "future_name"),
    [
        ("/models", "Future"),
        ("/models/poet-2/batch/embeddings", "EmbeddingFuture"),
        ("/models/boltzgen", "UnknownJobFuture"),
    ],
)
def test_a_models_job_reloads_as_a_plain_job_routed_by_its_type(job_type, future_name):
    future = Future.create(session=MagicMock(), response=_payload(job_type))

    assert type(future.job) is Job
    assert type(future).__name__ == future_name
    assert isinstance(future, UnknownJobFuture) == (future_name == "UnknownJobFuture")


TASK_FUTURES = [
    "Future",
    "FitFuture",
    "FutureUnit",
    "EmbeddingFuture",
    "LogitsFuture",
    "AttentionFuture",
    "LogLikelihoodFuture",
    "SingleSiteScoreFuture",
    "IndelScoreFuture",
    "SequenceGenerationFuture",
    "PredictionFuture",
    "SingleSitePredictionFuture",
    "StructureDesignFuture",
    "StructurePredictionFuture",
]


@pytest.mark.parametrize("name", TASK_FUTURES)
def test_task_futures_live_in_jobs_futures(name):
    from openprotein.jobs import futures

    future = getattr(futures, name)
    assert getattr(jobs, name) is future
    assert future.__module__ == "openprotein.jobs.futures"
    assert name in jobs.__all__


@pytest.mark.parametrize("name", TASK_FUTURES)
def test_task_futures_are_not_in_models(name):
    assert not hasattr(models, name)
    assert name not in models.__all__


@pytest.mark.parametrize(
    "module",
    [
        "openprotein.models.future",
        "openprotein.models.results",
        "openprotein.models.unit",
        "openprotein.models._dispatch",
    ],
)
def test_the_old_future_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_unit_records_are_read_by_the_jobs_api():
    from openprotein.jobs import _api
    from openprotein.models import _api as model_api

    assert callable(_api.get_unit)
    assert not hasattr(model_api, "get_unit")


@pytest.mark.parametrize("name", ["Unit", "UnitState", "JobOutput"])
def test_the_unit_and_output_records_are_not_public(name):
    from openprotein.jobs import schemas

    assert not hasattr(jobs, name)
    assert name not in jobs.__all__
    assert name not in schemas.__all__
    assert not hasattr(schemas, name)


def test_units_are_handled_as_future_units():
    from openprotein.jobs import futures

    assert jobs.FutureUnit is futures.FutureUnit
    assert not hasattr(futures.FutureUnit, "state")


def test_domain_futures_stay_with_their_domain():
    from openprotein.data import prompts

    assert not hasattr(prompts, "PromptFuture")
    assert not issubclass(prompts.Prompt, jobs.Future)
    assert not hasattr(jobs, "PromptFuture")
    assert not hasattr(jobs, "MSASearchFuture")


def test_a_search_job_reloads_as_a_plain_future():
    future = Future.create(session=MagicMock(), response=_payload("/search/mmseqs2"))

    assert type(future.job) is Job
    assert type(future) is jobs.Future
    assert future.unit(0, output="msa").output == "msa"
    assert find_spec("openprotein.search") is None


FIT_FUTURES = ["GPFitFuture", "SVDFitFuture", "UMAPFitFuture", "CrossvalidateFuture"]


@pytest.mark.parametrize("name", FIT_FUTURES)
def test_fit_futures_live_in_jobs_futures(name):
    from openprotein.jobs import futures

    future = getattr(futures, name)
    assert getattr(jobs, name) is future
    assert future.__module__ == "openprotein.jobs.futures"
    assert name in jobs.__all__
    assert not hasattr(models, name)
    assert name not in models.__all__


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("openprotein.models.gp", "CrossvalidateFuture"),
        ("openprotein.models.gp", "GPFitFuture"),
        ("openprotein.models.svd", "SVDFitFuture"),
        ("openprotein.models.umap", "UMAPFitFuture"),
    ],
)
def test_fit_futures_are_absent_from_their_old_defining_modules(module, name):
    assert not hasattr(importlib.import_module(module), name)


LAZY_ONLY = ("openprotein.models", "openprotein.data", "openprotein.search")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _executed_at_import(body: list[ast.stmt]):
    """Statements that run when the module is imported, function bodies excluded."""
    for node in body:
        yield node
        if isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _executed_at_import(node.body)
            yield from _executed_at_import(node.orelse)
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            for block in (node.body, node.orelse, node.finalbody):
                yield from _executed_at_import(block)
            for handler in node.handlers:
                yield from _executed_at_import(handler.body)
        elif isinstance(node, (ast.With, ast.For, ast.While, ast.ClassDef)):
            yield from _executed_at_import(node.body)
            yield from _executed_at_import(getattr(node, "orelse", []))


def _eager_imports(source: str, module: str) -> list[str]:
    """Modules under LAZY_ONLY that `module` imports when it is imported."""
    package = (
        module.split(".") if module.endswith("__init__") else module.split(".")[:-1]
    )
    package = [p for p in package if p != "__init__"]
    found = []
    for node in _executed_at_import(ast.parse(source).body):
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            parent = ".".join(base + ([node.module] if node.module else []))
            targets = [parent] + [f"{parent}.{alias.name}" for alias in node.names]
        else:
            continue
        found += [
            t
            for t in targets
            if any(t == p or t.startswith(p + ".") for p in LAZY_ONLY)
        ]
    return found


@pytest.mark.parametrize(
    "source",
    [
        "from openprotein.models import Model",
        "from ..models import Model",
        "from openprotein import models",
        "import openprotein.data.prompts",
        "try:\n    pass\nexcept ImportError:\n    from openprotein.models import Model",
        "if x:\n    pass\nelse:\n    from openprotein.search import SearchJob",
        "try:\n    pass\nfinally:\n    from openprotein.models import Model",
        "class A:\n    from openprotein.models import Model",
        "if TYPE_CHECKING:\n    pass\nelse:\n    from openprotein.models import Model",
    ],
)
def test_the_eager_import_check_sees_each_import_form(source):
    assert _eager_imports(source, "openprotein.jobs.futures")


@pytest.mark.parametrize(
    "source",
    [
        "if TYPE_CHECKING:\n    from openprotein.models import Model",
        "def f():\n    from openprotein.models import Model",
        "from openprotein.jobs import Job",
        "from . import _api",
    ],
)
def test_the_eager_import_check_ignores_lazy_and_local_imports(source):
    assert _eager_imports(source, "openprotein.jobs.futures") == []


def test_jobs_imports_models_data_and_search_only_lazily():
    package = Path(jobs.__file__).parent
    eager = {
        str(path.relative_to(package)): found
        for path in sorted(package.rglob("*.py"))
        if (
            found := _eager_imports(
                path.read_text(),
                "openprotein.jobs."
                + ".".join(path.relative_to(package).with_suffix("").parts),
            )
        )
    }
    assert eager == {}


@pytest.mark.parametrize(
    "first",
    [
        "openprotein.jobs",
        "openprotein.jobs.futures",
        "openprotein.models",
        "openprotein.data.prompts",
        "openprotein.predictor",
    ],
)
def test_each_package_imports_first_in_a_fresh_interpreter(first):
    code = (
        f"import {first}\n"
        "import openprotein\n"
        "from openprotein.jobs import EmbeddingFuture, Future\n"
        "from openprotein.jobs._dispatch import create\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
