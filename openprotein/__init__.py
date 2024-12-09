"""
OpenProtein Python client.

A pythonic interface for interacting with our cutting-edge protein engineering platform.

isort:skip_file
"""

from typing import TYPE_CHECKING
import warnings

from openprotein._version import __version__
from openprotein.app import (
    DataAPI,
    JobsAPI,
    AlignAPI,
    EmbeddingsAPI,
    FoldAPI,
    SVDAPI,
    UMAPAPI,
    PredictorAPI,
    DesignAPI,
)
from openprotein.app.models import Future
from openprotein.base import APISession

if TYPE_CHECKING:
    from openprotein.app.deprecated import TrainingAPI, DesignAPI


class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    _data = None
    _jobs = None
    _align = None
    _embeddings = None
    _svd = None
    _umap = None
    _fold = None
    _predictor = None
    _designer = None
    _deprecated = None

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
    def train(self):
        raise AttributeError(
            "Access to deprecated train module is under the deprecated property, i.e. session.deprecated.train"
        )

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
            self._svd = SVDAPI(self, self.embeddings)
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
            self._predictor = PredictorAPI(self, self.embeddings, self.svd)
        return self._predictor

    @property
    def design(self) -> DesignAPI:
        """
        The designer submodule gives access to functionality for designing new sequences using models from predictor train.
        """
        if self._designer is None:
            self._designer = DesignAPI(self)
        return self._designer

    @property
    def fold(self) -> FoldAPI:
        """
        The fold submodule gives access to functionality for folding sequences and returning PDBs.
        """
        if self._fold is None:
            self._fold = FoldAPI(self)
        return self._fold

    @property
    def deprecated(self) -> "Deprecated":

        if self._deprecated is None:
            warnings.warn(
                "Support for deprecated APIs will be dropped in the future! Read the documentation to migrate to the updated APIs."
            )
            self._deprecated = self.Deprecated(self)
        return self._deprecated

    class Deprecated:

        _train = None
        _design = None

        def __init__(self, session: APISession):
            self.session = session

        @property
        def train(self) -> "TrainingAPI":
            """
            The train submodule gives access to functionality for training and validating ML models.
            """
            from openprotein.app.deprecated import TrainingAPI

            if self._train is None:
                self._train = TrainingAPI(self.session)
            return self._train

        @property
        def design(self) -> "DesignAPI":
            """
            The design submodule gives access to functionality for designing new sequences using models from train.
            """
            from openprotein.app.deprecated import DesignAPI

            if self._design is None:
                self._design = DesignAPI(self.session)
            return self._design


connect = OpenProtein
