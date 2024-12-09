"""
OpenProtein schemas for interacting with the system.

isort:skip_file
"""

from .job import Job, JobStatus, JobType
from .assaydata import AssayDataPage, AssayMetadata
from .design import (
    ModelCriterion,
    NMutationCriterion,
    n_mutations,
    Subcriterion,
    Criterion,
    Criteria,
)
from .align import MSAJob, MSASamplingMethod, AlignType, PromptJob, PromptPostParams
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
from .features import FeatureType
from .svd import (
    SVDMetadata,
    FitJob as SVDFitJob,
    EmbeddingsJob as SVDEmbeddingsJob,
)
from .umap import (
    UMAPMetadata,
    FitJob as UMAPFitJob,
    EmbeddingsJob as UMAPEmbeddingsJob,
)
from .predictor import (
    Constraints,
    CVJob,
    Kernel,
    PredictJob,
    PredictMultiJob,
    PredictMultiSingleSiteJob,
    PredictorArgs,
    PredictorMetadata,
    PredictSingleSiteJob,
    TrainJob,
)
from .designer import (
    Design,
    DesignJob,
    DesignAlgorithm,
    DesignConstraint,
)

from .deprecated import *
