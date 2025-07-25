"""Schemas for the OpenProtein design system."""

from collections import namedtuple
from datetime import datetime
from enum import Enum
from typing import Any, Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, RootModel, model_serializer

from openprotein.jobs import Job, JobStatus, JobType


class CriterionType(str, Enum):
    """
    Enum representing the types of criteria.

    Attributes
    ----------
    model : str
        Criterion type for model-based criteria.
    n_mutations : str
        Criterion type for mutation count-based criteria.
    """

    model = "model"
    n_mutations = "n_mutations"


class Subcriterion(BaseModel):
    """
    Base class for subcriteria.

    Attributes
    ----------
    criterion_type : CriterionType
        The type of the criterion.
    """

    criterion_type: CriterionType

    def __and__(self, other: "Subcriterion | Criterion | Any") -> "Criterion":
        """
        Combine this subcriterion with another using logical AND.

        Parameters
        ----------
        other : Subcriterion or Criterion or Any
            The other subcriterion or criterion to combine.

        Returns
        -------
        Criterion
            A new Criterion with the two subcriteria AND-ed.

        Raises
        ------
        ValueError
            If `other` is not a Subcriterion or Criterion.
        """
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
        """
        Combine this subcriterion with another using logical OR.

        Parameters
        ----------
        other : Subcriterion or Criterion or Any
            The other subcriterion or criterion to combine.

        Returns
        -------
        Criteria
            A new Criteria with the two subcriteria OR-ed.

        Raises
        ------
        ValueError
            If `other` is not a Subcriterion or Criterion.
        """
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
    """
    Subcriterion for model-based criteria.

    Attributes
    ----------
    criterion_type : CriterionType
        The type of the criterion (always 'model').
    model_id : str
        The identifier of the model.
    measurement_name : str
        The name of the measurement.
    criterion : ModelCriterion.Criterion
        The criterion details.
    """

    class Criterion(BaseModel):
        """
        Inner class representing the details of a model criterion.

        Attributes
        ----------
        weight : float
            The weight of the criterion.
        direction : DirectionEnum or None
            The direction of the comparison.
        target : float or None
            The target value for the criterion.
        """

        class DirectionEnum(str, Enum):
            """
            Enum for direction of comparison.

            Attributes
            ----------
            gt : str
                Greater than.
            lt : str
                Less than.
            eq : str
                Equal to.
            """

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
        """
        Set the weight of the criterion.

        Parameters
        ----------
        weight : float
            The weight to set.

        Returns
        -------
        ModelCriterion
            The updated ModelCriterion.
        """
        self.criterion.weight = weight
        return self

    def __lt__(self, other: float) -> "ModelCriterion":
        """
        Set the criterion to less than a target value.

        Parameters
        ----------
        other : float
            The target value.

        Returns
        -------
        ModelCriterion
            The updated ModelCriterion.
        """
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.lt
        return self

    def __gt__(self, other: float) -> "ModelCriterion":
        """
        Set the criterion to greater than a target value.

        Parameters
        ----------
        other : float
            The target value.

        Returns
        -------
        ModelCriterion
            The updated ModelCriterion.
        """
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.gt
        return self

    def __eq__(self, other: float) -> "ModelCriterion":
        """
        Set the criterion to equal a target value.

        Parameters
        ----------
        other : float
            The target value.

        Returns
        -------
        ModelCriterion
            The updated ModelCriterion.
        """
        self.criterion.target = other
        self.criterion.direction = ModelCriterion.Criterion.DirectionEnum.eq
        return self

    __rmul__ = __mul__

    @model_serializer(mode="wrap")
    def validate_criterion_before_serialize(self, handler):
        """
        Validate the criterion before serialization.

        Parameters
        ----------
        handler : callable
            The serialization handler.

        Returns
        -------
        Any
            The serialized object.

        Raises
        ------
        ValueError
            If direction or target is not set.
        """
        if (
            self.criterion is None
            or self.criterion.direction is None
            or self.criterion.target is None
        ):
            raise ValueError("Expected direction and target to be set")
        return handler(self)


class NMutationCriterion(Subcriterion):
    """
    Subcriterion for mutation count-based criteria.

    Attributes
    ----------
    criterion_type : CriterionType
        The type of the criterion (always 'n_mutations').
    sequences : list of str
        List of sequences.
    """

    criterion_type: CriterionType = CriterionType.n_mutations
    sequences: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def remove_empty_sequences(self, handler):
        """
        Remove empty sequences before serialization.

        Parameters
        ----------
        handler : callable
            The serialization handler.

        Returns
        -------
        dict
            The serialized object with empty sequences removed.
        """
        d = handler(self)
        if not d["sequences"]:
            del d["sequences"]
        return d


n_mutations = NMutationCriterion


class Criterion(RootModel):
    """
    Class representing a logical AND of subcriteria.

    Attributes
    ----------
    root : list of Subcriterion
        The list of subcriteria.
    """

    root: list[ModelCriterion | NMutationCriterion | Subcriterion]

    def __and__(self, other: "Criterion | Subcriterion") -> "Criterion":
        """
        Combine this criterion with another using logical AND.

        Parameters
        ----------
        other : Criterion or Subcriterion
            The other criterion or subcriterion to combine.

        Returns
        -------
        Criterion
            A new Criterion with the two criteria AND-ed.
        """
        others = []

        if isinstance(other, Subcriterion):
            others = [other]
        elif isinstance(other, Criterion):
            others = other.root

        return Criterion(self.root + others)

    def __or__(self, other: "Criterion | Subcriterion") -> "Criteria":
        """
        Combine this criterion with another using logical OR.

        Parameters
        ----------
        other : Criterion or Subcriterion
            The other criterion or subcriterion to combine.

        Returns
        -------
        Criteria
            A new Criteria with the two criteria OR-ed.
        """
        if isinstance(other, Criterion):
            pass
        elif isinstance(other, Subcriterion):
            other = Criterion([other])

        return Criteria([self, other])


class Criteria(RootModel):
    """
    Class representing a logical OR of criteria.

    Attributes
    ----------
    root : list of Criterion
        The list of criteria.
    """

    root: list[Criterion]

    def __or__(self, other: "Criterion | Subcriterion | Criteria") -> "Criteria":
        """
        Combine this criteria with another using logical OR.

        Parameters
        ----------
        other : Criterion or Subcriterion or Criteria
            The other criterion, subcriterion, or criteria to combine.

        Returns
        -------
        Criteria
            A new Criteria with the two criteria OR-ed.
        """
        if isinstance(other, Criteria):
            pass
        if isinstance(other, Criterion):
            other = Criteria([other])
        elif isinstance(other, Subcriterion):
            other = Criteria([Criterion([other])])

        return Criteria(self.root + other.root)


class DesignConstraint:
    """
    Class for managing design constraints on a sequence.

    Attributes
    ----------
    sequence : str
        The sequence to constrain.
    mutations : dict of int to set of str
        Allowed amino acids at each position.
    """

    def __init__(self, sequence: str):
        """
        Initialize the design constraint.

        Parameters
        ----------
        sequence : str
            The sequence to constrain.
        """
        self.sequence = sequence
        self.mutations = self.initialize(sequence)

    def initialize(self, sequence: str) -> dict[int, set[str]]:
        """
        Initialize with no changes allowed to the sequence.

        Parameters
        ----------
        sequence : str
            The sequence to constrain.

        Returns
        -------
        dict of int to set of str
            Allowed amino acids at each position.
        """
        return {i: {aa} for i, aa in enumerate(sequence, start=1)}

    def allow(
        self,
        amino_acids: list[str] | str | None = None,
        positions: int | list[int] | None = None,
    ) -> None:
        """
        Allow specific amino acids at given positions.

        Parameters
        ----------
        amino_acids : list of str or str or None, optional
            Amino acids to allow. If None, allows all amino acids in the sequence.
        positions : int or list of int or None, optional
            Positions to allow amino acids at. If None, allows at all positions.
        """
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
        """
        Remove specific amino acids from being allowed at given positions.

        Parameters
        ----------
        amino_acids : list of str or str or None, optional
            Amino acids to remove. If None, removes all amino acids in the sequence.
        positions : int or list of int or None, optional
            Positions to remove amino acids from. If None, removes from all positions.
        """
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
        """
        Convert the internal mutations representation into a dictionary.

        Returns
        -------
        dict of int to list of str
            Allowed amino acids at each position.
        """
        return {i: list(aa) for i, aa in self.mutations.items()}


class DesignAlgorithm(str, Enum):
    """
    Enum representing design algorithms.

    Attributes
    ----------
    genetic_algorithm : str
        Genetic algorithm.
    """

    genetic_algorithm = "genetic-algorithm"


class Design(BaseModel):
    """
    Class representing a design.

    Attributes
    ----------
    id : str
        The design identifier.
    status : JobStatus
        The status of the design job.
    progress_counter : int
        The progress counter.
    created_date : datetime
        The creation date.
    algorithm : DesignAlgorithm
        The design algorithm used.
    num_rows : int
        The number of rows.
    num_steps : int
        The number of steps.
    assay_id : str
        The assay identifier.
    criteria : Criteria
        The design criteria.
    allowed_tokens : dict of str to list of str or None
        Allowed tokens for the design.
    pop_size : int
        Population size.
    n_offsprings : int
        Number of offsprings (GA parameter).
    crossover_prob : float
        Crossover probability (GA parameter).
    crossover_prob_pointwise : float
        Pointwise crossover probability (GA parameter).
    mutation_average_mutations_per_seq : int
        Average number of mutations per sequence (GA parameter).
    """

    id: str
    status: JobStatus
    progress_counter: int
    created_date: datetime
    algorithm: DesignAlgorithm
    num_rows: int
    num_steps: int
    assay_id: str
    criteria: Criteria
    allowed_tokens: dict[str, list[str]] | None
    pop_size: int
    n_offsprings: int
    crossover_prob: float
    crossover_prob_pointwise: float
    mutation_average_mutations_per_seq: int

    def is_done(self):
        """
        Check if the design job is done.

        Returns
        -------
        bool
            True if the job is done, False otherwise.
        """
        return self.status.done()


class DesignJob(Job):
    """
    Class representing a design job.

    Attributes
    ----------
    job_type : Literal[JobType.designer]
        The type of the job (always 'designer').
    """

    job_type: Literal[JobType.designer]


class DesignResult(NamedTuple):
    step: int
    sample_index: int
    sequence: str
    scores: np.ndarray
    subscores: np.ndarray
    means: np.ndarray
    vars: np.ndarray
