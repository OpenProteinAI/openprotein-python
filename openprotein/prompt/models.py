from openprotein import config
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
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
        Initialize a new Prompt instance from a job or metadata.

        Parameters
        ----------
        session : APISession
            An APISession object used for interacting with the API.
        job : PromptJob | None
            A PromptJob containing information about the optional prompt job.
        metadata : PromptMetadata | None
            A PromptMetadata object containing metadata for the prompt. If
            omitted, metadata is fetched (or constructed) from ``job``.
        num_replicates : int | None
            If provided alongside ``job``, build metadata locally without an
            extra API call.
        """
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
        self.job = None  # default for uploaded
        if self.metadata.job_id is not None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = PromptJob.create(jobs_api.get_job(job_id=self.metadata.job_id))
            self.job = job

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    def _get(self, verbose: bool = False, **kwargs) -> list[list[Protein | Complex]]:
        """
        Retrieve the prompt as context entries.

        Single-chain entries collapse to :py:class:`~openprotein.molecules.Protein`;
        multichain entries are returned as :py:class:`~openprotein.molecules.Complex`.
        Use :py:meth:`get_as_complexes` for a uniform ``Complex`` return, or
        :py:meth:`get_as_proteins` for a uniform ``Protein`` return (the latter
        raises if any entry is multichain).

        Returns:
            list of list of Protein or Complex representing the prompt context
        """
        context = api.get_prompt(session=self.session, prompt_id=str(self.id))
        return context

    def get_as_complexes(self) -> list[list[Complex]]:
        """
        Retrieve the prompt context with every entry as a :py:class:`Complex`.

        Single-chain entries are wrapped as ``Complex({"A": protein})`` so the
        return type is uniform regardless of chain count.
        """
        context = api.get_prompt(session=self.session, prompt_id=str(self.id))
        return [
            [
                (
                    entry
                    if isinstance(entry, Complex)
                    else Complex({"A": entry}, name=entry.name)
                )
                for entry in entries
            ]
            for entries in context
        ]

    def get_as_proteins(self) -> list[list[Protein]]:
        """
        Retrieve the prompt context with every entry as a :py:class:`Protein`.

        Raises :py:class:`InvalidParameterError` if any entry is multichain — use
        :py:meth:`get_as_complexes` instead when multichain entries may be present.
        """
        context = api.get_prompt(session=self.session, prompt_id=str(self.id))
        result: list[list[Protein]] = []
        for entries in context:
            row: list[Protein] = []
            for entry in entries:
                if isinstance(entry, Complex):
                    raise InvalidParameterError(
                        "prompt contains a multichain entry; use get_as_complexes()"
                    )
                row.append(entry)
            result.append(row)
        return result

    def _wait_job(
        self,
        interval: float = config.POLLING_INTERVAL,
        timeout: int | None = None,
        verbose: bool = False,
    ):
        if self.job is None:
            return None
        return super()._wait_job(interval, timeout, verbose)

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

        Single-chain queries collapse to :py:class:`Protein`; multichain queries are
        returned as :py:class:`Complex`. For a uniform return type, see
        :py:meth:`get_as_complex` or :py:meth:`get_as_protein`.

        Returns:
            Protein or Complex representing the query
        """
        query = api.get_query(session=self.session, query_id=str(self.id))
        return query

    def get_as_complex(self) -> Complex:
        """
        Retrieve the query as a :py:class:`Complex`.

        A single-chain :py:class:`Protein` result is wrapped as
        ``Complex({"A": protein})`` so the return type is uniform.
        """
        query = api.get_query(session=self.session, query_id=str(self.id))
        if isinstance(query, Protein):
            return Complex({"A": query}, name=query.name)
        return query

    def get_as_protein(self) -> Protein:
        """
        Retrieve the query as a :py:class:`Protein`.

        Raises :py:class:`InvalidParameterError` if the query is multichain — use
        :py:meth:`get_as_complex` instead when multichain queries may be present.
        """
        query = api.get_query(session=self.session, query_id=str(self.id))
        if isinstance(query, Complex):
            raise InvalidParameterError("query is multichain; use get_as_complex()")
        return query

    @property
    def id(self):
        """The unique identifier of the query."""
        return self.metadata.id

    @property
    def created_date(self):
        """The timestamp when the query was created."""
        return self.metadata.created_date
