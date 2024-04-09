from openprotein._version import __version__

from openprotein.base import APISession
from openprotein.api.jobs import JobsAPI, Job
from openprotein.api.data import DataAPI
from openprotein.api.align import AlignAPI
from openprotein.api.embedding import EmbeddingAPI
from openprotein.api.train import TrainingAPI
from openprotein.api.design import DesignAPI
from openprotein.api.fold import FoldAPI
from openprotein.api.jobs import load_job


class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    _embedding = None
    _fold = None
    _align = None
    _jobs = None
    _data = None
    _train = None
    _design = None

    def wait(self, job: Job, *args, **kwargs):
        return job.wait(self, *args, **kwargs)

    wait_until_done = wait

    def load_job(self, job_id):
        return load_job(self, job_id)

    @property
    def jobs(self) -> JobsAPI:
        """
        The jobs submodule gives access to functionality for listing jobs and checking their status.
        """
        if self._jobs is None:
            self._jobs = JobsAPI(self)
        return self._jobs

    @property
    def data(self) -> DataAPI:
        """
        The data submodule gives access to functionality for uploading and accessing user data.
        """
        if self._data is None:
            self._data = DataAPI(self)
        return self._data

    @property
    def train(self) -> TrainingAPI:
        """
        The train submodule gives access to functionality for training and validating ML models.
        """
        if self._train is None:
            self._train = TrainingAPI(self)
        return self._train

    @property
    def align(self) -> AlignAPI:
        """
        The PoET submodule gives access to the PoET generative model and MSA and prompt creation interfaces.
        """
        if self._align is None:
            self._align = AlignAPI(self)
        return self._align

    @property
    def embedding(self) -> EmbeddingAPI:
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        if self._embedding is None:
            self._embedding = EmbeddingAPI(self)
        return self._embedding

    @property
    def design(self) -> DesignAPI:
        """
        The design submodule gives access to functionality for designing new sequences using models from train.
        """
        if self._design is None:
            self._design = DesignAPI(self)
        return self._design

    @property
    def fold(self) -> FoldAPI:
        """
        The fold submodule gives access to functionality for folding sequences and returning PDBs.
        """
        if self._fold is None:
            self._fold = FoldAPI(self)
        return self._fold


connect = OpenProtein
