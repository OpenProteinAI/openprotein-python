from openprotein._version import __version__

from openprotein.base import APISession
from openprotein.api.jobs import JobsAPI
from openprotein.api.prots2prot import Prots2ProtAPI


class OpenProtein(APISession):
    @property
    def jobs(self):
        return JobsAPI(self)

    @property
    def prots2prot(self):
        return Prots2ProtAPI(self)

connect = OpenProtein
