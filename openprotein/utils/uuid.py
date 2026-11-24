import re
from uuid import UUID

__all__ = ["is_valid_uuid"]


def is_valid_uuid(uuid_to_test: str, version: int | None = None):
    """
    Check if uuid_to_test is a valid UUID.

     Parameters
    ----------
    uuid_to_test : str
    version : int, optional
        Require this UUID version. Any version is accepted when unset, since
        platform job ids are not all version 4.

     Returns
    -------
    `True` if uuid_to_test is a valid UUID, otherwise `False`.

     Examples
    --------
    >>> is_valid_uuid('c9bf9e57-1685-4c89-bafb-ff5af830be8a')
    True
    >>> is_valid_uuid('9af6889e-504d-8bdd-bd6c-1e58a6925708')
    True
    >>> is_valid_uuid('c9bf9e58')
    False
    """

    try:
        uuid_obj = UUID(uuid_to_test)
    except ValueError:
        return False
    if version is not None and uuid_obj.version != version:
        return False
    return str(uuid_obj) == uuid_to_test


_HYPHENATED_ID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def _as_id(value: object) -> str | None:
    """The lower-cased id when ``value`` is text in the hyphenated 8-4-4-4-12 form.

    Bare hex is not an id: it can be an amino-acid sequence.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str) and _HYPHENATED_ID.fullmatch(value):
        return value.lower()
    return None
