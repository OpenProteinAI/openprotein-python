"""
SVD module for OpenProtein for reducing embeddings.

isort:skip_file
"""

from .schemas import SVDMetadata, SVDFitJob, SVDEmbeddingsJob
from .models import SVDModel, SVDEmbeddingsResultFuture
from .svd import SVDAPI
