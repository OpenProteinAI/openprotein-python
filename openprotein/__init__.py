from openprotein._version import __version__

from openprotein.base import APISession
from openprotein.api.jobs import JobsAPI, Job
from openprotein.api.data import DataAPI
from openprotein.api.poet import PoetAPI
from openprotein.api.embedding import EmbeddingAPI
from openprotein.api.train import TrainingAPI
from openprotein.api.design import DesignAPI
from openprotein.api.predict import PredictAPI
class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    def wait(self, job: Job, *args, **kwargs):
        return job.wait(self, *args, **kwargs)
    
    wait_until_done = wait

    @property
    def jobs(self):
        """
        The jobs submodule gives access to functionality for listing jobs and checking their status.
        """
        return JobsAPI(self)
    
    @property
    def data(self):
        """
        The data submodule gives access to functionality for uploading and accessing user data. 
        """
        return DataAPI(self)
    
    @property 
    def train(self):
        """
        The train submodule gives access to functionality for training and validating ML models. 
        """
        return TrainingAPI(self)

    @property
    def poet(self):
        """
        The PoET submodule gives access to the PoET generative model and MSA and prompt creation interfaces.
        """
        return PoetAPI(self)
    
    @property
    def embedding(self):
        """
        The embedding submodule gives access to protein embedding models and their inference endpoints.
        """
        return EmbeddingAPI(self)

    @property 
    def predict(self):
        """
        The predict submodule gives access to sequence predictions using models from train. 
        """
        return PredictAPI(self)
    
    @property
    def design(self):
        """
        The design submodule gives access to functionality for designing new sequences using models from train. 
        """
        return DesignAPI(self)

connect = OpenProtein
