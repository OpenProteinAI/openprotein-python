"""
Embeddings module for using protein language models on OpenProtein.

isort:skip_file
"""

from .embeddings import EmbeddingsAPI
from .models import EmbeddingModel
from .openprotein import OpenProteinModel
from .esm import ESMModel
from .poet import PoETModel
from .poet2 import PoET2Model
from .schemas import (
    EmbeddedSequence,
    EmbeddingsJob,
    AttnJob,
    LogitsJob,
    ScoreJob,
    ScoreIndelJob,
    ScoreSingleSiteJob,
    GenerateJob,
)
from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)
