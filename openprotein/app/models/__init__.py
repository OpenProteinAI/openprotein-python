"""
OpenProtein app-level models providing service-level functionality.

isort:skip_file
"""

from .assaydata import AssayDataPage, AssayDataset, AssayMetadata

# workflow system
from .futures import Future, MappedFuture, StreamingFuture
from .train import CVFuture, TrainFuture
from .predict import PredictionResultFuture as WorkflowPredictionResultFuture
from .design import DesignFuture as WorkflowDesignFuture

# poet system
from .align import MSAFuture, PromptFuture
from .deprecated.poet import PoetGenerateFuture, PoetScoreFuture, PoetSingleSiteFuture

# distributed system
from .embeddings import (
    EmbeddingModel,
    EmbeddingResultFuture,
    EmbeddingsScoreResultFuture,
    ESMModel,
    OpenProteinModel,
    PoETModel,
)
from .svd import SVDModel
from .umap import UMAPModel
from .fold import AlphaFold2Model, ESMFoldModel, FoldModel, FoldResultFuture
from .predictor import PredictionResultFuture, PredictorModel
from .designer import DesignFuture
