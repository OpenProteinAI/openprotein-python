"""Shared configuration for E2E tests."""

import os


# Base timeout for E2E tests (defaults to 10 minutes).
BASE_TIMEOUT = int(os.getenv("OPENPROTEIN_E2E_TIMEOUT_SECONDS", 10 * 60))


def scaled_timeout(multiplier: float = 1.0) -> int:
    """Return BASE_TIMEOUT scaled by multiplier (minimum 1 second)."""
    if multiplier <= 0:
        raise ValueError(f"multiplier must be > 0, got {multiplier}")
    return max(1, int(BASE_TIMEOUT * multiplier))

