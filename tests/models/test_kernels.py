"""Trained GP kernels are defined in openprotein.models."""

import importlib
import warnings

import pytest

KERNELS = [
    "TrainedKernel",
    "LinearKernel",
    "RBFKernel",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "PeriodicKernel",
    "RationalQuadraticKernel",
]


@pytest.mark.parametrize("name", KERNELS)
def test_a_kernel_is_one_class_from_models_and_predictor(name):
    from openprotein import models
    from openprotein.models import kernels

    kernel = getattr(kernels, name)
    assert kernel.__module__ == "openprotein.models.kernels"
    assert getattr(models, name) is kernel
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predictor = importlib.import_module("openprotein.predictor")
        assert getattr(predictor, name) is kernel


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("openprotein.predictor", "GPHyperparameters"),
        ("openprotein.predictor.schemas", "GPHyperparameters"),
        ("openprotein.predictor.schemas", "sole_hyperparameters"),
        ("openprotein.predictor.schemas", "trained_kernel"),
        ("openprotein.predictor.api", "predictor_get_hyperparameters"),
        ("openprotein.models.kernels", "GPHyperparameters"),
        ("openprotein.models.kernels", "sole_hyperparameters"),
        ("openprotein.models.kernels", "trained_kernel"),
        ("openprotein.models.gp", "GPHyperparameters"),
        ("openprotein.models.gp", "sole_hyperparameters"),
        ("openprotein.models.gp", "trained_kernel"),
        ("openprotein.predictor.api", "GPHyperparameters"),
        ("openprotein.predictor.models", "GPHyperparameters"),
        ("openprotein.predictor.models", "sole_hyperparameters"),
        ("openprotein.predictor.models", "trained_kernel"),
    ],
)
def test_hyperparameter_helpers_are_private(module, name):
    assert not hasattr(importlib.import_module(module), name)


def test_the_kernel_spec_types_stay_in_predictor():
    from openprotein.predictor import Kernel, KernelType, schemas

    assert KernelType is schemas.KernelType
    assert Kernel is schemas.Kernel
    assert KernelType.__module__ == Kernel.__module__ == "openprotein.predictor.schemas"


def test_a_trained_kernel_type_equals_its_kernel_type():
    from openprotein.models.kernels import PeriodicKernel, RBFKernel
    from openprotein.predictor import KernelType

    assert RBFKernel.type == KernelType.RBF
    assert PeriodicKernel.type == KernelType.PERIODIC
