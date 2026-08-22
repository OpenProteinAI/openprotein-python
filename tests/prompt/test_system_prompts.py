"""Tests for the curated prompts exposed as attributes on ``PromptAPI``.

The prompt IDs are written by hand in ``openprotein/prompt/prompt.py``, so the
checks here are that each attribute resolves the ID it claims, and that the
unversioned attributes stay aliases of their versioned counterparts rather than
drifting onto IDs of their own.
"""

from unittest.mock import MagicMock

import pytest

from openprotein.prompt import prompt as prompt_module
from openprotein.prompt.models import Prompt
from openprotein.prompt.prompt import PromptAPI

# Each curated prompt: the versioned attribute, the unversioned alias that
# should track it, and the ID both must resolve.
CURATED_PROMPTS = [
    ("human_antibody_vh_v1", "human_antibody_vh", prompt_module.HUMAN_ANTIBODY_VH_V1),
    ("human_antibody_vl_v1", "human_antibody_vl", prompt_module.HUMAN_ANTIBODY_VL_V1),
    (
        "human_antibody_vh_vl_v1",
        "human_antibody_vh_vl",
        prompt_module.HUMAN_ANTIBODY_VH_VL_V1,
    ),
    (
        "human_antibody_vl_vh_v1",
        "human_antibody_vl_vh",
        prompt_module.HUMAN_ANTIBODY_VL_VH_V1,
    ),
]


@pytest.fixture
def prompt_api(mock_session: MagicMock):
    """PromptAPI whose GETs return metadata echoing the requested prompt ID."""

    def get(endpoint: str, **kwargs):
        prompt_id = endpoint.rsplit("/", 1)[-1]
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "id": prompt_id,
            "name": "Curated Prompt",
            "created_date": "2023-01-01T00:00:00",
            "num_replicates": 10,
            "status": "SUCCESS",
            "job_id": None,
        }
        return response

    mock_session.get.side_effect = get
    return PromptAPI(mock_session)


@pytest.mark.parametrize("versioned,alias,prompt_id", CURATED_PROMPTS)
def test_versioned_attribute_resolves_its_id(
    prompt_api: PromptAPI,
    mock_session: MagicMock,
    versioned: str,
    alias: str,
    prompt_id: str,
):
    prompt = getattr(prompt_api, versioned)

    assert isinstance(prompt, Prompt)
    assert prompt.id == prompt_id
    mock_session.get.assert_called_once_with(f"v1/prompt/{prompt_id}")


@pytest.mark.parametrize("versioned,alias,prompt_id", CURATED_PROMPTS)
def test_unversioned_alias_tracks_latest_version(
    prompt_api: PromptAPI, versioned: str, alias: str, prompt_id: str
):
    assert getattr(prompt_api, alias).id == prompt_id
    assert getattr(prompt_api, versioned).id == prompt_id


def test_curated_prompt_ids_are_distinct():
    ids = [prompt_id for _, _, prompt_id in CURATED_PROMPTS]

    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("versioned,alias,prompt_id", CURATED_PROMPTS)
def test_curated_attributes_are_documented_properties(
    versioned: str, alias: str, prompt_id: str
):
    # Properties so they read as ``session.prompt.<name>``, documented because
    # those docstrings are the published SDK docs for the curated prompts.
    for name in (versioned, alias):
        attribute = getattr(PromptAPI, name)
        assert isinstance(attribute, property)
        assert attribute.__doc__
