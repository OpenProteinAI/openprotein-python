import re
from inspect import cleandoc

import pytest

from openprotein import OpenProtein
from openprotein.jobs.futures import (
    Future,
    FutureUnit,
)
from openprotein.models import (
    GPModel,
    MethodDescriptor,
    Model,
    ModelDescription,
    ModelMetadata,
    ModelsAPI,
    StructureGenerationJob,
    SVDModel,
    UMAPModel,
)
from openprotein.models.svd import SVDEmbeddingSource, _SVDRecord


def _own_doc(cls: type, name: str) -> str:
    member = vars(cls)[name]
    if isinstance(member, property):
        member = member.fget
    elif isinstance(member, (classmethod, staticmethod)):
        member = member.__func__
    assert member is not None
    return cleandoc(member.__doc__ or "")


def _attributes(cls: type) -> set[str]:
    doc = cleandoc(vars(cls).get("__doc__") or "")
    assert "Attributes\n----------" in doc
    return set(re.findall(r"^([a-z][a-z0-9_]*)\s*:", doc, flags=re.MULTILINE))


def test_model_identity_is_documented():
    assert "id" in _attributes(Model)


@pytest.mark.parametrize("name", ["metadata", "methods"])
def test_model_discovery_properties_have_own_documentation(name):
    assert _own_doc(Model, name)


def test_models_api_documents_the_session_entrypoint():
    assert "session.models" in cleandoc(ModelsAPI.__doc__ or "")


@pytest.mark.parametrize(
    ("schema", "fields"),
    [
        (ModelDescription, {"summary", "citation_title", "doi"}),
        (MethodDescriptor, {"path", "outputs", "description", "name"}),
        (ModelMetadata, set(ModelMetadata.model_fields)),
    ],
)
def test_model_schema_fields_have_attribute_documentation(schema, fields):
    assert fields <= _attributes(schema)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        (Future, "units"),
        (FutureUnit, "job_id"),
        (FutureUnit, "position"),
        (FutureUnit, "output"),
        (FutureUnit, "status"),
    ],
)
def test_task_provenance_properties_have_own_documentation(owner, name):
    assert _own_doc(owner, name)


def test_task_factory_is_private_in_the_reference():
    assert ":meta private:" in _own_doc(Future, "create")


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        (SVDModel, "embedding"),
        (SVDModel, "requested_n_components"),
        (SVDModel, "delete"),
        (GPModel, "delete"),
        (UMAPModel, "delete"),
    ],
)
def test_fitted_model_operations_have_own_documentation(owner, name):
    assert _own_doc(owner, name)


@pytest.mark.parametrize("schema", [SVDEmbeddingSource, _SVDRecord])
def test_svd_record_fields_have_attribute_documentation(schema):
    assert set(schema.model_fields) == _attributes(schema)


def test_compatibility_design_job_documents_its_type():
    assert "job_type" in _attributes(StructureGenerationJob)


@pytest.mark.parametrize(("name", "label"), [("svd", "SVD"), ("umap", "UMAP")])
def test_session_projection_properties_document_their_api(name, label):
    assert label in _own_doc(OpenProtein, name)
