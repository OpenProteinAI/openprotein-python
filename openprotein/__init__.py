"""
OpenProtein Python client.

A pythonic interface for interacting with our cutting-edge protein engineering platform.

isort:skip_file
"""

from typing import TYPE_CHECKING
import warnings

from openprotein._version import __version__
from openprotein.data import DataAPI
from openprotein.errors import DeprecationError
from openprotein.jobs import JobsAPI
from openprotein.align import AlignAPI
from openprotein.prompt import PromptAPI
from openprotein.embeddings import EmbeddingsAPI
from openprotein.fold import FoldAPI
from openprotein.models import ModelsAPI
from openprotein.svd import SVDAPI
from openprotein.umap import UMAPAPI
from openprotein.predictor import PredictorAPI
from openprotein.design import DesignAPI
from openprotein.jobs import Future
from openprotein.base import APISession


class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    _data = None
    _jobs = None
    _align = None
    _prompt = None
    _embeddings = None
    _svd = None
    _umap = None
    _fold = None
    _predictor = None
    _design = None
    _models = None

    def wait(self, future: Future, *args, **kwargs):
        return future.wait(*args, **kwargs)

    wait_until_done = wait

    def load_job(self, job_id):
        return self.jobs.get(job_id=job_id)

    @property
    def data(self) -> DataAPI:
        """
        The data submodule gives access to functionality for uploading and accessing user data.
        """
        if self._data is None:
            self._data = DataAPI(self)
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
    def align(self) -> AlignAPI:
        """
        The Align submodule gives access to the sequence alignment capabilities by building MSAs and prompts that can be used with PoET.
        """
        if self._align is None:
            self._align = AlignAPI(self)
        return self._align

    @property
    def prompt(self) -> PromptAPI:
        """
        The Align submodule gives access to the sequence alignment capabilities by building MSAs and prompts that can be used with PoET.
        """
        if self._prompt is None:
            self._prompt = PromptAPI(self)
        return self._prompt

    @property
    def embedding(self) -> EmbeddingsAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._embeddings is None:
            self._embeddings = EmbeddingsAPI(self)
        return self._embeddings

    embeddings = embedding

    @property
    def svd(self) -> SVDAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._svd is None:
            self._svd = SVDAPI(
                session=self,
            )
        return self._svd

    @property
    def umap(self) -> UMAPAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._umap is None:
            self._umap = UMAPAPI(
                session=self,
            )
        return self._umap

    @property
    def predictor(self) -> PredictorAPI:
        """
        The predictor submodule gives access to training and predicting with predictors built on top of embeddings.
        """
        if self._predictor is None:
            self._predictor = PredictorAPI(
                session=self,
            )
        return self._predictor

    @property
    def design(self) -> DesignAPI:
        """
        The designer submodule gives access to functionality for designing new sequences using models from predictor train.
        """
        if self._design is None:
            self._design = DesignAPI(
                session=self,
            )
        return self._design

    @property
    def fold(self) -> FoldAPI:
        """
        The fold submodule gives access to functionality for folding sequences and returning PDBs.
        """
        if self._fold is None:
            self._fold = FoldAPI(self)
        return self._fold

    @property
    def models(self) -> "ModelsAPI":
        """
        The models submodule provides a unified entry point to all protein models.
        """
        if self._models is None:
            self._models = ModelsAPI(self)
        return self._models


connect = OpenProtein
