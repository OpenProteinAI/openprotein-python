"""Helper for deprecating public interfaces."""

import warnings


def warn_deprecated(
    name: str,
    use: str,
    *,
    since: str,
    removed_in: str,
    stacklevel: int = 3,
) -> None:
    """
    Warn that a public interface is deprecated.

    The default ``stacklevel`` points the warning at the caller of the
    deprecated function that calls this helper.

    Parameters
    ----------
    name : str
        Name of the deprecated interface.
    use : str
        The replacement, already formatted for display.
    since : str
        Version the interface was deprecated in.
    removed_in : str
        Version the interface is removed in.
    stacklevel : int, default=3
        Stack level passed to `warnings.warn`.
    """
    warnings.warn(
        f"`{name}` is deprecated since {since} and will be removed in {removed_in}. "
        f"Use {use}.",
        FutureWarning,
        stacklevel=stacklevel,
    )
