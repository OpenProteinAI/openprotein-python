"""Feature types used in OpenProtein."""

from enum import Enum
from typing import Literal


class FeatureType(str, Enum):

    PLM = "PLM"
    SVD = "SVD"


# NOTE: only works with python 3.12+
# Feature = Literal[*tuple([r.value for r in FeatureType])]
Feature = Literal["PLM", "SVD"]
