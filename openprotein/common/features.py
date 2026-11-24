"""Feature types used in OpenProtein."""

from enum import Enum
from typing import Literal

__all__ = ["Feature", "FeatureType"]


class FeatureType(str, Enum):
    PLM = "PLM"
    SVD = "SVD"


Feature = Literal["PLM", "SVD"]
