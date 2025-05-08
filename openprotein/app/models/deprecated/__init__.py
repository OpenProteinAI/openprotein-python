"""
Deprecated OpenProtein application objects.

isort:skip_file
"""

# workflow apis
from .train import TrainFuture as WorkflowTrainFuture
from .predict import PredictionResultFuture as WorkflowPredictionResultFuture
from .design import DesignFuture as WorkflowDesignFuture
