"""
OpenProtein Python client.

A pythonic interface for interacting with our cutting-edge protein engineering platform.

isort:skip_file
"""

from openprotein._version import __version__
from openprotein.app import (
    AssayDataAPI,
    JobsAPI,
    TrainingAPI,
    DesignAPI,
    AlignAPI,
    EmbeddingsAPI,
    FoldAPI,
    SVDAPI,
    UMAPAPI,
    PredictorAPI,
    DesignerAPI,
)
from openprotein.app.models import Future
from openprotein.base import APISession


class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    _data = None
    _jobs = None
    _train = None
    _design = None
    _align = None
    _embedding = None
    _svd = None
    _umap = None
    _fold = None
    _predictor = None
    _designer = None

    def wait(self, future: Future, *args, **kwargs):
        return future.wait(*args, **kwargs)

    wait_until_done = wait

    def load_job(self, job_id):
        return self.jobs.get(job_id=job_id)

    @property
    def data(self) -> AssayDataAPI:
        """
        The data submodule gives access to functionality for uploading and accessing user data.
        """
        if self._data is None:
            self._data = AssayDataAPI(self)
        return self._data

    @property
    def jobs(self) -> JobsAPI:
        """
        The jobs submodule gives access to functionality for listing jobs and checking their status.
        """
        if self._jobs is None:
            self._jobs = JobsAPI(self)
        return self._jobs

    @property
    def train(self) -> TrainingAPI:
        """
        The train submodule gives access to functionality for training and validating ML models.
        """
        if self._train is None:
            self._train = TrainingAPI(self)
        return self._train

    @property
    def design(self) -> DesignAPI:
        """
        The design submodule gives access to functionality for designing new sequences using models from train.
        """
        if self._design is None:
            self._design = DesignAPI(self)
        return self._design

    @property
    def align(self) -> AlignAPI:
        """
        The Align submodule gives access to the sequence alignment capabilities by building MSAs and prompts that can be used with PoET.
        """
        if self._align is None:
            self._align = AlignAPI(self)
        return self._align

    @property
    def embedding(self) -> EmbeddingsAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._embedding is None:
            self._embedding = EmbeddingsAPI(self)
        return self._embedding

    @property
    def svd(self) -> SVDAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._svd is None:
            self._svd = SVDAPI(self, self.embedding)
        return self._svd

    @property
    def umap(self) -> UMAPAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._umap is None:
            self._umap = UMAPAPI(self)
        return self._umap

    @property
    def predictor(self) -> PredictorAPI:
        """
        The predictor submodule gives access to training and predicting with predictors built on top of embeddings.
        """
        if self._predictor is None:
            self._predictor = PredictorAPI(self, self.embedding, self.svd)
        return self._predictor

    @property
    def designer(self) -> DesignerAPI:
        """
        The designer submodule gives access to functionality for designing new sequences using models from predictor train.
        """
        if self._designer is None:
            self._designer = DesignerAPI(self)
        return self._designer

    @property
    def fold(self) -> FoldAPI:
        """
        The fold submodule gives access to functionality for folding sequences and returning PDBs.
        """
        if self._fold is None:
            self._fold = FoldAPI(self)
        return self._fold


connect = OpenProtein
