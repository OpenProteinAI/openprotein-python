from typing import TypeVar

import numpy as np
import numpy.typing as npt

DType = TypeVar("DType", bound=np.generic)


def readonly_view(x: npt.NDArray[DType]) -> npt.NDArray[DType]:
    v = x.view()
    v.flags.writeable = False
    return v
