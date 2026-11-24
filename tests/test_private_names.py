"""Helper types and fold wire models have private names."""

import importlib
from typing import TypeVar

import pytest


def test_estimators_have_no_public_type_variables():
    estimators = importlib.import_module("openprotein.models.estimators")

    assert not hasattr(estimators, "F")
    assert not hasattr(estimators, "H")


def test_jobs_futures_keeps_exactly_its_shipped_type_variables():
    futures = importlib.import_module("openprotein.jobs.futures")
    public = {
        name
        for name, value in vars(futures).items()
        if isinstance(value, TypeVar) and not name.startswith("_")
    }
    assert public == {"F", "V", "K"}


def test_the_fold_wire_models_are_private():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("openprotein.fold.request")
    request = importlib.import_module("openprotein.fold._request")
    assert hasattr(request, "MsaChain")
    assert hasattr(request, "complex_chain_ids")
