from openprotein._version import __version__

from openprotein.base import APISession
from openprotein.api.jobs import JobsAPI
from openprotein.api.data import DataAPI
from openprotein.api.poet import PoetAPI


class OpenProtein(APISession):
    """
    The base class for accessing OpenProtein API functionality.
    """

    @property
    def jobs(self):
        """
        The jobs submodule gives access to functionality for listing jobs and checking their status.
        """
        return JobsAPI(self)
    
    @property
    def data(self):
        return DataAPI(self)

    @property
    def poet(self):
        """
        The PoET submodule gives access to the PoET generative model and MSA and prompt creation interfaces.
        """
        return PoetAPI(self)

connect = OpenProtein
