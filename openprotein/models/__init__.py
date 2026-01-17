"""Make the ModelsAPI class available on the package."""

# Reference other modules
from ..embeddings import ESMModel, OpenProteinModel, PoET2Model, PoETModel
from ..fold import (
    AlphaFold2Model,
    Boltz1Model,
    Boltz1xModel,
    Boltz2Model,
    ESMFoldModel,
    MiniFoldModel,
    RosettaFold3Model,
)
from .foundation.boltzgen import BoltzGenFuture, BoltzGenJob, BoltzGenModel
from .foundation.proteinmpnn import ProteinMPNNModel
from .foundation.rfdiffusion import RFdiffusionFuture, RFdiffusionJob, RFdiffusionModel
from .models import ModelsAPI
