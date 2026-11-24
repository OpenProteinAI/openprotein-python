"""Reduction types used in OpenProtein."""

from enum import Enum
from typing import Literal

__all__ = ["Reduction", "ReductionType"]


class ReductionType(str, Enum):
    """
    ReductionType is an enumeration of the possible reduction types available.

    Attributes:
        MEAN : Mean reduction takes the mean of the embeddings across the sequence length dimension.
        SUM : Sum reduction takes the sum of the embeddings across the sequence length dimension.
    """

    MEAN = "MEAN"
    SUM = "SUM"


Reduction = Literal["MEAN", "SUM"]
