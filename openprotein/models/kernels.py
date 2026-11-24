"""Trained Gaussian process kernels, as fitted GPs report them."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = [
    "LinearKernel",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "PeriodicKernel",
    "RBFKernel",
    "RationalQuadraticKernel",
    "TrainedKernel",
]


class _GPHyperparameters(BaseModel):
    """Trained hyperparameters of one property's GP, in data space, as the API serves
    them. Kernel values are listed per subkernel.
    """

    mean: float | None = None
    noise: float | None = None
    lengthscale: list[float | None] = []
    outputscale: list[float | None] = []
    period: list[float | None] = []
    alpha: list[float | None] = []


class TrainedKernel(BaseModel):
    """A trained GP kernel, in data space.

    Each kernel type carries only the hyperparameters that apply to it.

    Attributes
    ----------
    outputscale : float
        Signal variance of the kernel.
    """

    model_config = ConfigDict(frozen=True)

    type: ClassVar[str]
    outputscale: float


class LinearKernel(TrainedKernel):
    """A trained linear kernel."""

    type: ClassVar[str] = "linear"


class _StationaryKernel(TrainedKernel):
    """A trained kernel over distances between embeddings.

    Attributes
    ----------
    lengthscale : float
        Distance in embedding space over which the kernel decays.
    """

    lengthscale: float


class RBFKernel(_StationaryKernel):
    """A trained RBF kernel."""

    type: ClassVar[str] = "rbf"


class Matern12Kernel(_StationaryKernel):
    """A trained Matern 1/2 kernel."""

    type: ClassVar[str] = "matern12"


class Matern32Kernel(_StationaryKernel):
    """A trained Matern 3/2 kernel."""

    type: ClassVar[str] = "matern32"


class Matern52Kernel(_StationaryKernel):
    """A trained Matern 5/2 kernel."""

    type: ClassVar[str] = "matern52"


class PeriodicKernel(_StationaryKernel):
    """A trained periodic kernel.

    Attributes
    ----------
    period : float
        Period length, in embedding space.
    """

    type: ClassVar[str] = "periodic"
    period: float


class RationalQuadraticKernel(_StationaryKernel):
    """A trained rational quadratic kernel.

    Attributes
    ----------
    alpha : float
        Scale-mixture parameter.
    """

    type: ClassVar[str] = "rational_quadratic"
    alpha: float


_TRAINED_KERNELS: dict[str, type[TrainedKernel]] = {
    kernel.type: kernel
    for kernel in (
        LinearKernel,
        RBFKernel,
        Matern12Kernel,
        Matern32Kernel,
        Matern52Kernel,
        PeriodicKernel,
        RationalQuadraticKernel,
    )
}


def _sole_hyperparameters(
    by_property: dict[str, _GPHyperparameters],
) -> _GPHyperparameters:
    """The hyperparameters of a GP trained on one property."""
    if len(by_property) != 1:
        raise ValueError(
            f"expected hyperparameters for one property, got {len(by_property)}"
        )
    return next(iter(by_property.values()))


def _trained_kernel(
    kernel_type: str, hyperparameters: _GPHyperparameters
) -> TrainedKernel:
    """Build the trained kernel of ``kernel_type`` from its served hyperparameters."""
    kernel = _TRAINED_KERNELS.get(kernel_type.lower())
    if kernel is None:
        raise ValueError(f"unknown kernel type {kernel_type!r}")
    values = {}
    for name in ("outputscale", "lengthscale", "period", "alpha"):
        per_subkernel = getattr(hyperparameters, name)
        applies = name in kernel.model_fields
        if len(per_subkernel) > 1 or (applies and not per_subkernel):
            raise ValueError(
                f"expected one {kernel.type} kernel, got {len(per_subkernel)} {name} values"
            )
        if applies:
            values[name] = per_subkernel[0]
    return kernel(**values)
