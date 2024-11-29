import re
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_serializer,
)

from .job import Job, JobType


class SubscoreMetadata(BaseModel):
    y_mu: float | None = None
    y_var: float | None = None


class DesignSubscore(BaseModel):
    score: float
    metadata: SubscoreMetadata


class DesignResult(BaseModel):
    step: int
    sample_index: int
    sequence: str
    # scores: List[int]
    # subscores_metadata: List[List[DesignSubscore]]
    # scores: list[float]
    scores: list[list[DesignSubscore]] = Field(..., alias="subscores_metadata")
    # umap1: float
    # umap2: float


class DesignJob(Job):
    job_type: Literal[JobType.workflow_design]


class Design(DesignJob):
    assay_id: str
    num_rows: int
    result: list[DesignResult]


class CriterionType(str, Enum):
    model = "model"
    n_mutations = "n_mutations"


class Subcriterion(BaseModel):

    criterion_type: CriterionType

    def __and__(self, other: "Subcriterion | Criterion | Any") -> "Criterion":
        """Returns a Criterion with the two subcriteria AND-ed."""
        others = []
        if isinstance(other, Subcriterion):
            others = [other]
        elif isinstance(other, Criterion):
            others = other.root
        else:
            raise ValueError(
                f"Expected to chain only with criterion or subcriterion, got {type(other)}"
            )
        return Criterion([self] + others)  # type: ignore - doesnt like Self

    def __or__(self, other: "Subcriterion | Criterion | Any") -> "Criteria":
        """Returns a Criteria with the two subcriteria OR-ed."""
        if isinstance(other, Criterion):
            pass
        elif isinstance(other, Subcriterion):
            other = Criterion([other])
        else:
            raise ValueError(
                f"Expected to chain only with criterion or subcriterion, got {type(other)}"
            )
        return Criteria([Criterion([self]), other])


class ModelCriterion(Subcriterion):

    class Criterion(BaseModel):
        class DirectionEnum(str, Enum):
            gt = ">"
            lt = "<"
            eq = "="

        weight: float = 1.0
        direction: DirectionEnum | None = None
        target: float | None = None

    criterion_type: CriterionType = CriterionType.model
    model_id: str
    measurement_name: str
    criterion: Criterion = Criterion()

    model_config = ConfigDict(protected_namespaces=())

    def __mul__(self, weight: float) -> "ModelCriterion":
        self.criterion.weight = weight
        return self

    def __lt__(self, other: float) -> "ModelCriterion":
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.lt
        return self

    def __gt__(self, other: float) -> "ModelCriterion":
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.gt
        return self

    def __eq__(self, other: float) -> "ModelCriterion":
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.eq
        return self

    __rmul__ = __mul__

    @model_serializer(mode="wrap")
    def validate_criterion_before_serialize(self, handler):
        if (
            self.criterion is None
            or self.criterion.direction is None
            or self.criterion.target is None
        ):
            raise ValueError("Expected direction and target to be set")
        return handler(self)


class NMutationCriterion(Subcriterion):
    criterion_type: CriterionType = CriterionType.n_mutations
    sequences: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def remove_empty_sequences(self, handler):
        d = handler(self)
        if not d["sequences"]:
            del d["sequences"]
        return d


n_mutations = NMutationCriterion


class Criterion(RootModel):
    root: list[ModelCriterion | NMutationCriterion | Subcriterion]

    def __and__(self, other: "Criterion | Subcriterion") -> "Criterion":
        """Returns a Criteria with the other criterion OR-ed with itself."""
        others = []

        if isinstance(other, Subcriterion):
            others = [other]
        elif isinstance(other, Criterion):
            others = other.root

        return Criterion(self.root + others)

    def __or__(self, other: "Criterion | Subcriterion") -> "Criteria":
        """Returns a Criteria with the other criterion OR-ed with itself."""

        if isinstance(other, Criterion):
            pass
        elif isinstance(other, Subcriterion):
            other = Criterion([other])

        return Criteria([self, other])


class Criteria(RootModel):
    root: list[Criterion]

    def __or__(self, other: "Criterion | Subcriterion | Criteria") -> "Criteria":
        """Returns a Criteria with the other criteria OR-ed with itself."""
        if isinstance(other, Criteria):
            pass
        if isinstance(other, Criterion):
            other = Criteria([other])
        elif isinstance(other, Subcriterion):
            other = Criteria([Criterion([other])])

        return Criteria(self.root + other.root)


class DesignConstraint:
    def __init__(self, sequence: str):
        self.sequence = sequence
        self.mutations = self.initialize(sequence)

    def initialize(self, sequence: str) -> dict[int, list[str]]:
        """Initialize with no changes allowed to the sequence."""
        return {i: [aa] for i, aa in enumerate(sequence, start=1)}

    def allow(self, positions: int | list[int], amino_acids: list[str] | str) -> None:
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

    def remove(self, positions: int | list[int], amino_acids: list[str] | str) -> None:
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

    def as_dict(self) -> dict[int, list[str]]:
        """Convert the internal mutations representation into a dictionary."""
        return self.mutations


class DesignJobCreate(BaseModel):
    assay_id: str
    criteria: Criteria
    num_steps: int = 8
    pop_size: int | None = None
    n_offsprings: int | None = None
    crossover_prob: float | None = None
    crossover_prob_pointwise: float | None = None
    mutation_average_mutations_per_seq: int | None = None
    allowed_tokens: DesignConstraint | dict[int, list[str]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("allowed_tokens", mode="before")
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
    amino_acids: list[str] | str = "ACDEFGHIKLMNPQRSTVWY",
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
    positions: list, amino_acids: str | list = "ACDEFGHIKLMNPQRSTVWY"
):
    if isinstance(amino_acids, list):
        amino_acids = "".join(amino_acids)
    return {k: list(amino_acids) for k in positions}


def no_change(sequence: str):
    return {k + 1: [v] for k, v in enumerate(sequence)}


def keep_cys(sequence: str):
    return {k + 1: [v] for k, v in enumerate(sequence) if v == "C"}
