from openprotein.api import align
from openprotein.base import APISession
from openprotein.schemas import Job, PoetInputType


class AlignFuture:
    session: APISession
    job: Job

    def get_input(self, input_type: PoetInputType):
        """See child function docs."""
        return align.get_input(self.session, self.job, input_type)

    def get_seed(self):
        """See child function docs."""
        return align.get_seed(self.session, self.job)

    @property
    def id(self):
        return self.job.job_id
