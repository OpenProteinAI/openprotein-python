"""Make the ModelsAPI class available on the package."""

# Reference other modules
from ..embeddings import ESMModel, OpenProteinModel, PoET2Model, PoETModel
from ..fold import (
    AlphaFold2Model,
    Boltz1Model,
    Boltz1xModel,
    Boltz2Model,
    ESMFold2FastModel,
    ESMFold2Model,
    ESMFoldModel,
    MiniFoldModel,
    ProtenixModel,
    ProtenixV2Model,
    RosettaFold3Model,
)
from .foundation.boltzgen import BoltzGenFuture, BoltzGenModel
from .foundation.esmif1 import ESMIF1Model
from .foundation.proteinmpnn import ProteinMPNNModel
from .foundation.rfdiffusion import RFdiffusionFuture, RFdiffusionModel
from .models import ModelsAPI
from .structure_generation import (
    StructureGenerationFuture,
    StructureGenerationJob,
)
