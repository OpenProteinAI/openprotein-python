"""
OpenProtein schemas for interacting with the system.

isort:skip_file
"""

from .job import Job, JobStatus, JobType
from .assaydata import AssayDataPage, AssayMetadata
from .train import (
    CVItem as WorkflowCVItem,
    CVJob as WorkflowCVJob,
    TrainJob as WorkflowTrainJob,
    TrainStep as WorkflowTrainStep,
)
from .predict import (
    PredictJob as WorkflowPredictJob,
    PredictSingleSiteJob as WorkflowPredictSingleSiteJob,
)
from .design import (
    ModelCriterion,
    Criterion,
    DesignJobCreate,
    DesignMetadata,
    DesignResults,
    DesignStep,
    DesignJob,
)
from .deprecated.poet import (
    PoetScoreJob,
    PoetSSPJob,
    PoetScoreResult,
    PoetSSPResult,
    PoetGenerateJob,
)
from .align import MSAJob, MSASamplingMethod, PoetInputType, PromptJob, PromptPostParams
from .embeddings import (
    ModelMetadata,
    ModelDescription,
    TokenInfo,
    ReductionType,
    EmbeddedSequence,
    EmbeddingsJob,
    AttnJob,
    LogitsJob,
    ScoreJob,
    ScoreSingleSiteJob,
    GenerateJob,
)
from .fold import FoldJob
from .svd import (
    SVDMetadata,
    FitJob,
    EmbeddingsJob as SVDEmbeddingsJob,
)
from .predictor import (
    Constraints,
    CVJob,
    FeatureType,
    Kernel,
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictorArgs,
    PredictorMetadata,
    PredictSingleSiteJob,
    TrainJob,
)
