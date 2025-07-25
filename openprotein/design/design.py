"""Design API providing the interface to design novel proteins based on a your design criteria."""

from openprotein.base import APISession
from openprotein.data import AssayDataset, DataAPI
from openprotein.jobs import JobsAPI

from . import api
from .future import DesignFuture
from .schemas import Criteria, Criterion, DesignConstraint, Subcriterion


class DesignAPI:
    """Design API providing the interface to design novel proteins based on your design criteria."""

    def __init__(
        self,
        session: APISession,
    ):
        self.session = session

    def list_designs(self) -> list[DesignFuture]:
        """
        List all designs.

        Returns
        -------
        list of DesignFuture
            A list of DesignFuture objects representing all designs.
        """
        return [
            DesignFuture(
                session=self.session,
                metadata=m,
            )
            for m in api.designs_list(session=self.session)
        ]

    def get_design(self, design_id: str) -> DesignFuture:
        """
        Retrieve a specific design by its ID.

        Parameters
        ----------
        design_id : str
            ID of the design to retrieve.

        Returns
        -------
        DesignFuture
            A future object representing the design job and its results.
        """
        return DesignFuture(
            session=self.session,
            metadata=api.design_get(session=self.session, design_id=design_id),
        )

    def create_genetic_algorithm_design(
        self,
        assay: AssayDataset,
        criteria: Criteria | Subcriterion | Criterion,
        num_steps: int = 25,
        pop_size: int = 1024,
        n_offsprings: int = 5120,
        crossover_prob: float = 1.0,
        crossover_prob_pointwise: float = 0.2,
        mutation_average_mutations_per_seq: int = 1,
        allowed_tokens: DesignConstraint | dict[int, list[str]] = {},
    ) -> DesignFuture:
        """
        Start a protein design job using a genetic algorithm based on assay data, a trained ML model, and specified criteria.

        Parameters
        ----------
        assay : AssayDataset
            The AssayDataset to design from.
        criteria : Criteria or Subcriterion or Criterion
            Criteria for evaluating the design.
        num_steps : int, optional
            The number of steps in the genetic algorithm. Default is 25.
        pop_size : int, optional
            The population size for the genetic algorithm. Default is 1024.
        n_offsprings : int, optional
            The number of offspring for the genetic algorithm. Default is 5120.
        crossover_prob : float, optional
            The crossover probability for the genetic algorithm. Default is 1.0.
        crossover_prob_pointwise : float, optional
            The pointwise crossover probability for the genetic algorithm. Default is 0.2.
        mutation_average_mutations_per_seq : int, optional
            The average number of mutations per sequence. Default is 1.
        allowed_tokens : DesignConstraint or dict of int to list of str, optional
            A dict of positions and allowed tokens (e.g. {1: ['G', 'L']}) designating how mutations may occur. Defaults to empty dict.

        Returns
        -------
        DesignFuture
            A future object representing the design job and its results.
        """
        return DesignFuture.create(
            session=self.session,
            job=api.designer_create_genetic_algorithm(
                self.session,
                assay_id=assay.id,
                criteria=criteria,
                num_steps=num_steps,
                pop_size=pop_size,
                n_offsprings=n_offsprings,
                crossover_prob=crossover_prob,
                crossover_prob_pointwise=crossover_prob_pointwise,
                mutation_average_mutations_per_seq=mutation_average_mutations_per_seq,
                allowed_tokens=allowed_tokens,
            ),
        )

    def create_design_job(
        self,
        *args,
    ):
        raise AttributeError(
            "create_design_job belongs to the deprecated design interface. Use create_genetic_algorithm_design instead in the new design interface."
        )

    def get_design_results(self, *args):
        raise AttributeError(
            "get_design_results belongs to the deprecated design interface. Use get_design and wait instead in the new design interface."
        )
