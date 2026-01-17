"""Reduction types used in OpenProtein."""

from enum import Enum
from typing import Literal


class ReductionType(str, Enum):
    """
    ReductionType is an enumeration of the possible reduction types available.

    Attributes:
        MEAN : Mean reduction takes the mean of the embeddings across the sequence length dimension.
        SUM : Sum reduction takes the sum of the embeddings across the sequence length dimension.
    """

    MEAN = "MEAN"
    SUM = "SUM"


# NOTE: only works with python 3.12+
# Reduction = Literal[*tuple([r.value for r in ReductionType])]
Reduction = Literal["MEAN", "SUM"]
