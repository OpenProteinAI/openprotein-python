"""SDK code loads assays through ``session.data.Assay``, so it raises no deprecation warning."""

import ast
import pathlib
import warnings
from unittest.mock import MagicMock

import openprotein
from openprotein.data import AssayDataset
from openprotein.models import _inputs

DEPRECATED = {"create", "get", "list", "load_assay"}
BRIDGE = pathlib.Path(openprotein.__file__).parent / "data" / "data.py"


def _deprecated_calls(path: pathlib.Path) -> list[str]:
    calls = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in DEPRECATED
        ):
            continue
        receiver = node.func.value
        named = isinstance(receiver, ast.Name) and receiver.id == "data_api"
        built = (
            isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Name)
            and receiver.func.id == "DataAPI"
        )
        if named or built:
            calls.append(f"{path.name}:{node.lineno} .{node.func.attr}()")
    return calls


def test_no_sdk_module_calls_the_deprecated_assay_verbs():
    root = BRIDGE.parent.parent
    found = [
        call
        for path in sorted(root.rglob("*.py"))
        if path != BRIDGE
        for call in _deprecated_calls(path)
    ]
    assert found == []


def test_an_assay_id_loads_without_a_warning():
    session = MagicMock()
    session.get.return_value.status_code = 200
    session.get.return_value.json.return_value = {
        "assay_name": "n",
        "assay_description": "d",
        "assay_id": "a1",
        "original_filename": "f.csv",
        "created_date": "2023-01-01T00:00:00",
        "num_rows": 1,
        "num_entries": 1,
        "measurement_names": ["y"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assay = _inputs.as_dataset(session, "a1")
    assert isinstance(assay, AssayDataset) and assay.id == "a1"
