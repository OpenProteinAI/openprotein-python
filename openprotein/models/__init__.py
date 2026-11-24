"""Make the ModelsAPI class available on the package."""

import importlib
from typing import Any

from openprotein import _deprecation

from .ensemble import (
    AlphaFlow,
    AlphaFlowInput,
    BioEmu,
    BioEmuInput,
    ESMFlow,
    ESMFlowInput,
    StructureEnsemblePredictionModel,
)
from .estimators import GP, SVD, UMAP
from .folding import (
    AlphaFold2,
    Boltz1,
    Boltz2,
    ESMFold,
    ESMFold2,
    MiniFold,
    OpenDDE,
    Protenix,
    ProtenixV2,
    RosettaFold3,
    StructurePredictionModel,
)
from .foundational import (
    ESM1,
    ESM2,
    ESMC,
    ESMIF1,
    AbLang2,
    PoET,
    PoET2,
    ProteinLanguageModel,
    ProteinMPNN,
    ProtSeq,
    ProtT5,
    RotaProt,
)
from .generative import BoltzGen, RFdiffusion, StructureGenerativeModel
from .gp import GPModel
from .kernels import (
    LinearKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    PeriodicKernel,
    RationalQuadraticKernel,
    RBFKernel,
    TrainedKernel,
)
from .model import Model
from .models import ModelsAPI
from .schemas import (
    MethodDescriptor,
    ModelDescription,
    ModelMetadata,
)
from .structure_generation import (
    BoltzGenFuture,
    RFdiffusionFuture,
    StructureGenerationFuture,
    StructureGenerationJob,
)
from .svd import SVDModel
from .umap import UMAPModel

__all__ = [
    "ESM1",
    "ESM2",
    "ESMC",
    "ESMIF1",
    "GP",
    "SVD",
    "UMAP",
    "AbLang2",
    "AlphaFlow",
    "AlphaFlowInput",
    "AlphaFold2",
    "BioEmu",
    "BioEmuInput",
    "Boltz1",
    "Boltz2",
    "BoltzGen",
    "BoltzGenFuture",
    "ESMFlow",
    "ESMFlowInput",
    "ESMFold",
    "ESMFold2",
    "GPModel",
    "LinearKernel",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "MethodDescriptor",
    "MiniFold",
    "Model",
    "ModelDescription",
    "ModelMetadata",
    "ModelsAPI",
    "OpenDDE",
    "PeriodicKernel",
    "PoET",
    "PoET2",
    "ProtSeq",
    "ProtT5",
    "ProteinLanguageModel",
    "ProteinMPNN",
    "Protenix",
    "ProtenixV2",
    "RBFKernel",
    "RFdiffusion",
    "RFdiffusionFuture",
    "RationalQuadraticKernel",
    "RosettaFold3",
    "RotaProt",
    "SVDModel",
    "StructureEnsemblePredictionModel",
    "StructureGenerationFuture",
    "StructureGenerationJob",
    "StructureGenerativeModel",
    "StructurePredictionModel",
    "TrainedKernel",
    "UMAPModel",
]

_RENAMED = {
    "BoltzGenModel": "BoltzGen",
    "ESMIF1Model": "ESMIF1",
    "ProteinMPNNModel": "ProteinMPNN",
    "RFdiffusionModel": "RFdiffusion",
}
_MOVED = {
    **dict.fromkeys(
        (
            "AlphaFold2Model",
            "Boltz1Model",
            "Boltz1xModel",
            "Boltz2Model",
            "ESMFold2FastModel",
            "ESMFold2Model",
            "ESMFoldModel",
            "MiniFoldModel",
            "ProtenixModel",
            "ProtenixV2Model",
            "RosettaFold3Model",
        ),
        "openprotein.fold",
    ),
    **dict.fromkeys(
        ("ESMModel", "OpenProteinModel", "PoET2Model", "PoETModel"),
        "openprotein.embeddings",
    ),
}


def __getattr__(name: str) -> Any:
    if name == "SolubleMPNNModel":
        _deprecation.warn_deprecated(
            "openprotein.models.SolubleMPNNModel",
            "`session.models.solublempnn`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        from .foundational.proteinmpnn import _SolubleMPNN

        return _SolubleMPNN
    if name in _RENAMED:
        new = _RENAMED[name]
        module = "openprotein.models"
    elif name in _MOVED:
        new = name
        module = _MOVED[name]
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    _deprecation.warn_deprecated(
        f"openprotein.models.{name}",
        f"`{module}.{new}`",
        since="0.18.0",
        removed_in="0.19.0",
    )
    return getattr(importlib.import_module(module), new)
