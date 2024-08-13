from typing import Optional, Dict, List, Union, Literal, Any
from enum import Enum

from openprotein.base import APISession
from openprotein.api.jobs import AsyncJobFuture
from openprotein.schemas import JobType
import openprotein.config as config
from openprotein.jobs import JobType, Job

from openprotein.errors import APIError
from openprotein.futures import FutureFactory, FutureBase
from openprotein.pydantic import BaseModel, Field, validator
from datetime import datetime
import re


class DesignMetadata(BaseModel):
    y_mu: Optional[float] = None
    y_var: Optional[float] = None


class DesignSubscore(BaseModel):
    score: float
    metadata: DesignMetadata


class DesignStep(BaseModel):
    step: int
    sample_index: int
    sequence: str
    # scores: List[int]
    # subscores_metadata: List[List[DesignSubscore]]
    __initial_scores: List[float] = Field(
        ..., alias="scores"
    )  # renaming 'scores' to 'initial_scores'  # noqa: E501
    scores: List[List[DesignSubscore]] = Field(
        ..., alias="subscores_metadata"
    )  # renaming 'subscores_metadata' to 'scores'  # noqa: E501
    # umap1: float
    # umap2: float


class DesignResults(BaseModel):
    status: str
    job_id: str
    job_type: str
    created_date: datetime
    start_date: datetime
    end_date: Optional[datetime]
    assay_id: str
    num_rows: int
    result: List[DesignStep]


class DirectionEnum(str, Enum):
    gt = ">"
    lt = "<"
    eq = "="


class Criterion(BaseModel):
    target: float
    weight: float
    direction: str


class ModelCriterion(BaseModel):
    criterion_type: Literal["model"]
    model_id: str
    measurement_name: str
    criterion: Criterion


class NMutationCriterion(BaseModel):
    criterion_type: Literal["n_mutations"]
    # sequences: Optional[List[str]]


CriterionItem = Union[ModelCriterion, NMutationCriterion]


class DesignConstraint:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.mutations = self.initialize(sequence)

    def initialize(self, sequence: str) -> Dict[int, List[str]]:
        """Initialize with no changes allowed to the sequence."""
        return {i: [aa] for i, aa in enumerate(sequence, start=1)}

    def allow(
        self, positions: Union[int, List[int]], amino_acids: Union[List[str], str]
    ) -> None:
        """Allow specific amino acids at given positions."""
        if isinstance(positions, int):
            positions = [positions]
        if isinstance(amino_acids, str):
            amino_acids = list(amino_acids)

        for position in positions:
            if position in self.mutations:
                self.mutations[position].extend(amino_acids)
            else:
                self.mutations[position] = amino_acids

    def remove(
        self, positions: Union[int, List[int]], amino_acids: Union[List[str], str]
    ) -> None:
        """Remove specific amino acids from being allowed at given positions."""
        if isinstance(positions, int):
            positions = [positions]
        if isinstance(amino_acids, str):
            amino_acids = list(amino_acids)

        for position in positions:
            if position in self.mutations:
                for aa in amino_acids:
                    if aa in self.mutations[position]:
                        self.mutations[position].remove(aa)

    def as_dict(self) -> Dict[int, List[str]]:
        """Convert the internal mutations representation into a dictionary."""
        return self.mutations


class DesignJobCreate(BaseModel):
    assay_id: str
    criteria: List[List[CriterionItem]]
    num_steps: Optional[int] = 8
    pop_size: Optional[int] = None
    n_offsprings: Optional[int] = None
    crossover_prob: Optional[float] = None
    crossover_prob_pointwise: Optional[float] = None
    mutation_average_mutations_per_seq: Optional[int] = None
    allowed_tokens: Optional[Union[DesignConstraint, Dict[int, List[str]]]] = None

    class Config:
        arbitrary_types_allowed = True

    @validator("allowed_tokens", pre=True)
    def ensure_dict(cls, v):
        if isinstance(v, DesignConstraint):
            return v.as_dict()
        return v


def _validate_mutation_dict(d: dict, amino_acids: str = "ACDEFGHIKLMNPQRSTVWY"):
    validated = {}
    for k, v in d.items():
        _ = [i for i in v if i in amino_acids]
        validated[k] = _
    return validated


def mutation_regex(
    constraints: str,
    amino_acids: Union[List[str], str] = "ACDEFGHIKLMNPQRSTVWY",
    verbose: bool = False,
) -> dict:
    """
    Parses a constraint string for sequence and return a mutation dict.

    Syntax supported:
    * [AC] - position must be A or C ONLY
    * X - position can be any amino acid
    * A - position will always be A
    * [^ACD] - anything except A, C or D
    * X{3} - 3 consecutive positions of any residue
    * A{3} -  3 consecutive positions of A

    Parameters
    ----------
    constraints: A string representing the constraints on the protein sequence.
    amino_acids: A list or string of all possible amino acids.
    verbose: control verbosity

    Returns
    -------
    dict : mutation dict
    """
    if isinstance(amino_acids, str):
        amino_acids = list(amino_acids)
    constraints_dict = {}

    constraints_dict = {}
    pos = 1

    pattern = re.compile(
        r"(\[[^\]]*\])|(\{[A-Z]+\})|([A-Z]\{\d+\})|([A-Z]\{\d+,\d*\})|(X\{\d+\})|([A-Z])|(X)"
    )

    for match in pattern.finditer(constraints):
        token = match.group()
        if verbose:
            print(f"parsed: {token}")

        if token.startswith("[") and token.endswith("]"):
            if "^" in token:
                # Negation
                excluded = set(token[2:-1])
                options = [aa for aa in amino_acids if aa not in excluded]
            else:
                # Specific options
                options = list(token[1:-1])
            constraints_dict[pos] = options
            pos += 1
        elif token.startswith("{") and token.endswith("}"):
            # Ranges of positions or exact repetitions for specific amino acids
            options = list(token[1:-1])
            constraints_dict[pos] = options
            pos += 1
        elif "{" in token and "X" not in token:
            # Ranges of positions or exact repetitions for specific amino acids
            base, range_part = token.split("{")
            if "," in range_part:
                # Range specified, handle similarly to previous versions
                start, end = map(int, range_part[:-1].split(","))
                for _ in range(start, end + 1):
                    constraints_dict[pos] = [base]
                    pos += 1
            else:
                # Exact repetition specified
                count = int(range_part[:-1])
                for _ in range(count):
                    constraints_dict[pos] = [base]
                    pos += 1
        elif token.startswith("X{") and token.endswith("}"):
            # Fixed number of wildcard positions
            num = int(token[2:-1])
            for _ in range(num):
                constraints_dict[pos] = list(amino_acids)
                pos += 1
        elif token == "X":
            # Any amino acid
            constraints_dict[pos] = list(amino_acids)
            pos += 1
        else:
            # Specific amino acid
            constraints_dict[pos] = [token]
            pos += 1

    return _validate_mutation_dict(constraints_dict)


def position_mutation(
    positions: List, amino_acids: Union[str, List] = "ACDEFGHIKLMNPQRSTVWY"
):
    if isinstance(amino_acids, list):
        amino_acids = "".join(amino_acids)
    return {k: list(amino_acids) for k in positions}


def nochange(sequence: str):
    return {k + 1: [v] for k, v in enumerate(sequence)}


def keep_cys(sequence: str):
    return {k + 1: [v] for k, v in enumerate(sequence) if v == "C"}


def create_design_job(session: APISession, design_job: DesignJobCreate):
    """
    Send a POST request for protein design job.

    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    design_job : DesignJobCreate
        The details of the design job to be created, with the following parameters:
        - assay_id: The ID for the assay.
        - criteria: A list of CriterionItem lists for evaluating the design.
        - num_steps: The number of steps in the genetic algo. Default is 8.
        - pop_size: The population size for the genetic algo. Default is None.
        - n_offsprings: The number of offspring for the genetic algo. Default is None.
        - crossover_prob: The crossover probability for the genetic algo. Default is None.
        - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
        - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
        - allowed_tokens: A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Default is None.

    Returns
    -------
    Job
        The created job as a Job instance.
    """
    params = design_job.dict(exclude_none=True)
    # print(f"sending design: {params}")
    response = session.post("v1/workflow/design/genetic-algorithm", json=params)

    return FutureFactory.create_future(session=session, response=response)


def get_design_results(
    session: APISession,
    job_id: str,
    step: Optional[int] = None,
    page_size: Optional[int] = None,
    page_offset: Optional[int] = None,
) -> DesignResults:
    """
    Retrieves the results of a Design job.

    This function retrieves the results of a Design job by making a GET request to design..

    Parameters
    ----------
    session : APISession
        APIsession with auth
    job_id : str
        The ID of the job whose results are to be retrieved.
    step: int
        The step to retrieve. -1 indicates the last step.
    page_size : Optional[int], default is None
        The number of results to be returned per page. If None, all results are returned.
    page_offset : Optional[int], default is None
        The number of results to skip. If None, defaults to 0.

    Returns
    -------
    DesignJob
        The job object representing the Design job.

    Raises
    ------
    HTTPError
        If the GET request does not succeed.
    """
    endpoint = f"v1/workflow/design/{job_id}"
    params = {}
    if page_size is not None:
        params["page_size"] = page_size
    if page_offset is not None:
        params["page_offset"] = page_offset
    if step is not None:
        params["step"] = step

    response = session.get(endpoint, params=params)

    return DesignResults(**response.json())


class DesignFutureMixin:
    session: APISession
    job: Job

    def get_results(
        self,
        step: Optional[int] = None,
        page_size: Optional[int] = None,
        page_offset: Optional[int] = None,
    ) -> DesignResults:
        """
        Retrieves the results of a Design job.

        This function retrieves the results of a Design job by making a GET request to design..

        Parameters
        ----------
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        DesignJob
            The job object representing the Design job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return get_design_results(
            self.session,
            job_id=self.job.job_id,
            step=step,
            page_size=page_size,
            page_offset=page_offset,
        )


class DesignFuture(DesignFutureMixin, AsyncJobFuture, FutureBase):
    """Future Job for manipulating results"""

    job_type = [JobType.workflow_design]

    def __init__(self, session: APISession, job: Job, page_size=1000):
        super().__init__(session, job)
        self.page_size = page_size

    def __str__(self) -> str:
        return str(self.job)

    def __repr__(self) -> str:
        return repr(self.job)

    def _fmt_results(self, results) -> List[Dict]:
        return [i.dict() for i in results]

    @property
    def id(self):
        return self.job.job_id

    def get(self, step: Optional[int] = None, verbose: bool = False) -> List[Dict]:
        """
        Get all the results of the design job.

        Args:
            verbose (bool, optional): If True, print verbose output. Defaults False.

        Raises:
            APIError: If there is an issue with the API request.

        Returns:
            List: A list of predict objects representing the results.
        """
        page = self.page_size

        results = []
        num_returned = page
        offset = 0

        while num_returned >= page:
            try:
                response = self.get_results(
                    page_offset=offset, step=step, page_size=page
                )
                results += response.result
                num_returned = len(response.result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
        return self._fmt_results(results)


class DesignAPI:
    """interface for calling Design endpoints"""

    session: APISession

    def __init__(self, session: APISession):
        self.session = session

    def create_design_job(self, design_job: DesignJobCreate) -> DesignFuture:
        """
        Start a protein design job based on your assaydata, a trained ML model and Criteria (specified here).

        Parameters
        ----------
        design_job : DesignJobCreate
            The details of the design job to be created, with the following parameters:
            - assay_id: The ID for the assay.
            - criteria: A list of CriterionItem lists for evaluating the design.
            - num_steps: The number of steps in the genetic algo. Default is 8.
            - pop_size: The population size for the genetic algo. Default is None.
            - n_offsprings: The number of offspring for the genetic algo. Default is None.
            - crossover_prob: The crossover probability for the genetic algo. Default is None.
            - crossover_prob_pointwise: The pointwise crossover probability for the genetic algo. Default is None.
            - mutation_average_mutations_per_seq: The average number of mutations per sequence. Default is None.
            - allowed_tokens: A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Default is None.

        Returns
        -------
        DesignFuture
            The created job as a DesignFuture instance.
        """
        return create_design_job(self.session, design_job)

    def get_design_results(
        self,
        job_id: str,
        step: Optional[int] = None,
        page_size: Optional[int] = None,
        page_offset: Optional[int] = None,
    ) -> DesignResults:
        """
        Retrieves the results of a Design job.

        Parameters
        ----------
        job_id : str
            The ID for the design job
        step: int
            The design step to retrieve, if None: retrieve all.
        page_size : Optional[int], default is None
            The number of results to be returned per page. If None, all results are returned.
        page_offset : Optional[int], default is None
            The number of results to skip. If None, defaults to 0.

        Returns
        -------
        DesignJob
            The job object representing the Design job.

        Raises
        ------
        HTTPError
            If the GET request does not succeed.
        """
        return get_design_results(
            self.session,
            step=step,
            job_id=job_id,
            page_size=page_size,
            page_offset=page_offset,
        )
