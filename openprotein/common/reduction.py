"""Reduction types used in OpenProtein."""

from enum import Enum


class ReductionType(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
