import importlib
import inspect
import pkgutil

import pytest

import openprotein


def _is_private(module_name: str) -> bool:
    return any(part.startswith("_") for part in module_name.split("."))


def _public_modules() -> list[str]:
    names = ["openprotein"]
    for info in pkgutil.walk_packages(openprotein.__path__, "openprotein."):
        if not _is_private(info.name):
            names.append(info.name)
    return sorted(names)


@pytest.mark.parametrize("module_name", _public_modules())
def test_public_modules_declare_their_interface(module_name):
    module = importlib.import_module(module_name)
    assert isinstance(getattr(module, "__all__", None), list)
    assert [name for name in module.__all__ if not hasattr(module, name)] == []


@pytest.mark.parametrize("module_name", _public_modules())
def test_public_definitions_are_in_all(module_name):
    module = importlib.import_module(module_name)
    defined = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(value) or inspect.isclass(value))
        and value.__module__ == module_name
    }
    assert defined <= set(module.__all__)


@pytest.mark.parametrize("module_name", _public_modules())
def test_all_names_no_private_module_functions_or_classes(module_name):
    module = importlib.import_module(module_name)
    leaked = set()
    for name in module.__all__:
        value = getattr(module, name)
        if not (inspect.isfunction(value) or inspect.isclass(value)):
            continue
        origin = value.__module__ or ""
        if origin.startswith("openprotein.") and _is_private(origin):
            leaked.add(f"{name} (from {origin})")
    assert not leaked
