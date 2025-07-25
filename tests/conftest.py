from unittest.mock import MagicMock

import pytest

from openprotein.base import APISession


@pytest.fixture
def mock_session():
    """Returns a MagicMock to be used as a session object."""
    session = MagicMock(spec=APISession)
    # To handle cases where the mock is used in a context manager
    session.__enter__.return_value = session
    return session
