"""
OpenProtein's embeddings module.

isort:skip_file
"""

from openprotein.app import EmbeddingsAPI
from openprotein.app.models import (
    ESMModel,
    OpenProteinModel,
    PoETModel,
    SVDModel,
    UMAPModel,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
    EmbeddingsGenerateFuture,
)
