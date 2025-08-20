"""Reduction types used in OpenProtein."""

from enum import Enum
from typing import Literal


class ReductionType(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"


# NOTE: only works with python 3.12+
# Reduction = Literal[*tuple([r.value for r in ReductionType])]
Reduction = Literal["MEAN", "SUM"]
