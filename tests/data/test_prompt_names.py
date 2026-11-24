import importlib

import pytest

import openprotein.data
from openprotein.data import PromptMetadata

MODULES = ["openprotein.data.prompts", "openprotein.data.prompts.schemas"]


@pytest.mark.parametrize("module", MODULES)
def test_prompt_summary_resolves_to_prompt_metadata_and_warns(module):
    with pytest.warns(FutureWarning) as record:
        resolved = importlib.import_module(module).PromptSummary
    assert resolved is PromptMetadata
    assert [str(w.message) for w in record] == [
        (
            f"`{module}.PromptSummary` is deprecated since 0.18.2 and will be removed "
            "in 0.19.0. Use `openprotein.data.PromptMetadata`."
        )
    ]
    assert record[0].filename == __file__


def test_from_import_of_prompt_summary_warns():
    with pytest.warns(FutureWarning, match="openprotein.data.prompts.PromptSummary"):
        from openprotein.data.prompts import PromptSummary  # noqa: F401


def test_prompt_metadata_is_exported_beside_its_siblings():
    assert {"AssayMetadata", "MSAMetadata", "PromptMetadata"} <= set(
        openprotein.data.__all__
    )
    assert (
        "PromptMetadata" in importlib.import_module("openprotein.data.prompts").__all__
    )
    assert (
        "PromptSummary"
        not in importlib.import_module("openprotein.data.prompts").__all__
    )


@pytest.mark.parametrize("module", MODULES)
def test_an_unknown_name_raises_attribute_error(module):
    with pytest.raises(AttributeError, match="NoSuchName"):
        _ = importlib.import_module(module).NoSuchName
