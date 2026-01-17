from openprotein.base import APISession
from openprotein.jobs import Future, JobsAPI
from openprotein.molecules import Complex, Protein

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
        """
        Retrieve the prompt as a list of :py:class:`~openprotein.molecules.Protein`.

        Returns:
            list of list of Protein representing the prompt context
        """
        context = api.get_prompt(session=self.session, prompt_id=str(self.id))
        return context

    def _wait_job(self, **kwargs):
        if self.job is None:
            return None
        return super()._wait_job(**kwargs)

    @property
    def id(self):
        """The unique identifier of the prompt."""
        return self.metadata.id

    @property
    def name(self):
        """The name of the prompt."""
        return self.metadata.name

    @property
    def description(self):
        """The description of the prompt."""
        return self.metadata.description

    @property
    def created_date(self):
        """The timestamp when the prompt was created."""
        return self.metadata.created_date

    @property
    def num_replicates(self):
        """The number of replicates in the prompt for an ensemble prompt."""
        return self.metadata.num_replicates

    @property
    def status(self):
        """The status of the prompt if sampling from an :py:class:`~openprotein.align.MSAFuture`."""
        if self.job is not None:
            return super().status
        return self.metadata.status


class Query:
    """
    Query containing a sequence/structure used to query the design models which opens up new workflows.

    Create a query with a masked sequence using :py:meth:`~openprotein.molecules.Protein.mask_sequence_at` for :py:class:`~openprotein.embeddings.PoET2Model` to run inverse folding.

    Create a query with a masked structure using :py:meth:`~openprotein.molecules.Protein.mask_structure_at` for :py:class:`~openprotein.models.RFdiffusionModel` to run inverse folding.
    """

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

    def get(self) -> Protein | Complex:
        """
        Retrieve the query as a :py:class:`~openprotein.molecules.Protein` or  :py:class:`~openprotein.molecules.Complex`.


        Returns:
            Protein or Complex representing the query
        """
        query = api.get_query(session=self.session, query_id=str(self.id))
        return query

    @property
    def id(self):
        """The unique identifier of the query."""
        return self.metadata.id

    @property
    def created_date(self):
        """The timestamp when the query was created."""
        return self.metadata.created_date
