from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_serializer


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

    def initialize(self, sequence: str) -> dict[int, set[str]]:
        """Initialize with no changes allowed to the sequence."""
        return {i: {aa} for i, aa in enumerate(sequence, start=1)}

    def allow(
        self,
        amino_acids: list[str] | str | None = None,
        positions: int | list[int] | None = None,
    ) -> None:
        """Allow specific amino acids at given positions."""
        if isinstance(positions, int):
            positions = [positions]
        elif positions is None:
            positions = [i + 1 for i in range(len(self.sequence))]
        if isinstance(amino_acids, str):
            amino_acids = list(amino_acids)
        elif amino_acids is None:
            amino_acids = list(self.sequence)

        for position in positions:
            if position in self.mutations:
                for aa in amino_acids:
                    self.mutations[position].add(aa)
            else:
                self.mutations[position] = set(amino_acids)

    def remove(
        self,
        amino_acids: list[str] | str | None = None,
        positions: int | list[int] | None = None,
    ) -> None:
        """Remove specific amino acids from being allowed at given positions."""
        if isinstance(positions, int):
            positions = [positions]
        elif positions is None:
            positions = [i + 1 for i in range(len(self.sequence))]
        if isinstance(amino_acids, str):
            amino_acids = list(amino_acids)
        elif amino_acids is None:
            amino_acids = list(self.sequence)

        for position in positions:
            if position in self.mutations:
                for aa in amino_acids:
                    if aa in self.mutations[position]:
                        self.mutations[position].remove(aa)

    def as_dict(self) -> dict[int, list[str]]:
        """Convert the internal mutations representation into a dictionary."""
        return {i: list(aa) for i, aa in self.mutations.items()}
