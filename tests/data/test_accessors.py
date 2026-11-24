"""Stored msas and prompts are reached only through ``session.data.MSA`` and ``session.data.Prompt``."""

import pytest

import openprotein.data
import openprotein.data.msa
import openprotein.data.prompts
from openprotein.data import DataAPI


@pytest.mark.parametrize("name", ["msa", "prompts"])
def test_session_data_has_no_accessor_object(name):
    assert not hasattr(DataAPI, name)


@pytest.mark.parametrize(
    ("module", "name"),
    [
        (openprotein.data, "MSADataAPI"),
        (openprotein.data, "PromptsAPI"),
        (openprotein.data.msa, "MSADataAPI"),
        (openprotein.data.prompts, "PromptsAPI"),
    ],
)
def test_the_accessor_classes_are_gone(module, name):
    assert not hasattr(module, name)
