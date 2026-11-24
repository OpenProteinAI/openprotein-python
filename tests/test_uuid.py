import pytest

from openprotein.utils.uuid import _as_id, is_valid_uuid


@pytest.mark.parametrize(
    "value",
    [
        "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
        "9af6889e-504d-8bdd-bd6c-1e58a6925708",
        "886313e1-3b8a-5372-9b90-0c9aee199e5d",
    ],
)
def test_any_uuid_version_is_valid(value):
    assert is_valid_uuid(value)


@pytest.mark.parametrize(
    "value",
    [
        "c9bf9e58",
        "my-prompt",
        "C9BF9E57-1685-4C89-BAFB-FF5AF830BE8A",
        "c9bf9e5716854c89bafbff5af830be8a",
    ],
)
def test_non_canonical_text_is_not_a_uuid(value):
    assert not is_valid_uuid(value)


def test_a_required_version_is_checked():
    assert is_valid_uuid("c9bf9e57-1685-4c89-bafb-ff5af830be8a", version=4)
    assert not is_valid_uuid("9af6889e-504d-8bdd-bd6c-1e58a6925708", version=4)


@pytest.mark.parametrize(
    "value",
    [
        "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
        "C9BF9E57-1685-4C89-BAFB-FF5AF830BE8A",
        b"C9BF9E57-1685-4C89-BAFB-FF5AF830BE8A",
    ],
    ids=["lower", "upper", "bytes"],
)
def test_an_id_in_any_case_is_returned_lower_cased(value):
    assert _as_id(value) == "c9bf9e57-1685-4c89-bafb-ff5af830be8a"


@pytest.mark.parametrize(
    "value",
    [
        "c9bf9e5716854c89bafbff5af830be8a",
        "c9bf9e58",
        "ACDEFGHIKL",
        b"\xff\xfe",
        "c9bf9e57-1685-4c89-bafb-ff5af830be8a ",
        None,
        7,
    ],
)
def test_text_outside_the_hyphenated_form_is_not_an_id(value):
    assert _as_id(value) is None
