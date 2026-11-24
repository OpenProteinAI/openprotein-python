import pytest

from openprotein._deprecation import warn_deprecated


def _deprecated_function():
    warn_deprecated("old.name", "`new.name`", since="0.18.0", removed_in="0.19.0")


def test_warn_deprecated_message_format():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("old.name", "`new.name`", since="0.18.0", removed_in="0.19.0")
    assert str(record[0].message) == (
        "`old.name` is deprecated since 0.18.0 and will be removed in 0.19.0. "
        "Use `new.name`."
    )


def test_warn_deprecated_uses_given_versions():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("a", "`b`", since="0.20.0", removed_in="0.21.0")
    assert str(record[0].message) == (
        "`a` is deprecated since 0.20.0 and will be removed in 0.21.0. Use `b`."
    )


def test_warn_deprecated_category_is_future_warning():
    with pytest.warns(FutureWarning):
        warn_deprecated("old.name", "`new.name`", since="0.18.0", removed_in="0.19.0")


def test_warn_deprecated_points_at_the_caller_of_the_deprecated_function():
    with pytest.warns(FutureWarning) as record:
        _deprecated_function()
    assert record[0].filename == __file__
