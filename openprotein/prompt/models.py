from openprotein.base import APISession
from openprotein.jobs import Future, JobsAPI
from openprotein.protein import Protein

from . import api
from .schemas import PromptJob, PromptMetadata, QueryMetadata


class Prompt(Future):
    """Prompt which contains a set of sequences and/or structures used to condition the PoET models."""

    metadata: PromptMetadata
    session: APISession
    job: PromptJob | None

    def __init__(
        self,
        session: APISession,
        job: PromptJob | None = None,
        metadata: PromptMetadata | None = None,
        num_replicates: int | None = None,
    ):
        """
        Initialize a new Prompt instance.

        Parameters
        ----------
        session : APISession
            An APISession object used for interacting with the API.
        job: PromptJob | None
            A PromptJob containing information about the optional prompt job.
        metadata : PromptMetadata
            A PromptMetadata object containing metadata for the prompt.
        """
        """Initializes with either job get or svd metadata get."""
        if metadata is None:
            # use job to fetch metadata
            if job is None:
                raise ValueError("Expected prompt metadata or job")
            # if no num_replicates, we need an api call to get the info
            if num_replicates is None:
                metadata = api.get_prompt_metadata(
                    session=session, prompt_id=job.job_id
                )
            # else we can just build the metadata from the job
            else:
                metadata = PromptMetadata(
                    id=job.job_id,
                    name=job.job_id,
                    description=None,
                    created_date=job.created_date,
                    num_replicates=num_replicates,
                    job_id=job.job_id,
                    status=job.status,
                )
        self.metadata = metadata
        self.session = session
        if self.metadata.job_id is not None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = PromptJob.create(jobs_api.get_job(job_id=self.metadata.job_id))
            super().__init__(session, job)

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    def get(self) -> list[list[Protein]]:
        context = api.get_prompt(session=self.session, prompt_id=str(self.id))
        return context

    def _wait_job(self, **kwargs):
        if self.job is None:
            return None
        return super()._wait_job(**kwargs)

    @property
    def id(self):
        return self.metadata.id

    @property
    def name(self):
        return self.metadata.name

    @property
    def description(self):
        return self.metadata.description

    @property
    def created_date(self):
        return self.metadata.created_date

    @property
    def num_replicates(self):
        return self.metadata.num_replicates

    @property
    def status(self):
        if self.job is not None:
            return super().status
        return self.metadata.status


class Query:
    """Query containing a sequence/structure used to query the PoET-2 model which opens up new workflows."""

    metadata: QueryMetadata
    session: APISession

    def __init__(self, session: APISession, metadata: QueryMetadata):
        """
        Initialize a new Query instance.

        Parameters
        ----------
        session : APISession
            An APISession object used for interacting with the API.
        metadata : QueryMetadata
            A QueryMetadata object containing metadata for the query.
        """
        self.metadata = metadata
        self.session = session

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    def get(self) -> Protein:
        query = api.get_query(session=self.session, query_id=str(self.id))
        return query

    @property
    def id(self):
        return self.metadata.id

    @property
    def created_date(self):
        return self.metadata.created_date
