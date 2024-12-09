"""
Deprecated OpenProtein schemas.

isort:skip_file
"""

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
    DesignJobCreate as WorkflowDesignJobCreate,
    DesignJob as WorkflowDesignJob,
    Design as WorkflowDesign,
)
