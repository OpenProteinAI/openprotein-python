from openprotein.api import designer
from openprotein.app.models import AssayDataset, DesignFuture
from openprotein.base import APISession
from openprotein.schemas import Criteria, Criterion, DesignConstraint, Subcriterion


class DesignAPI:
    """Interface for calling Designer endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def list_designs(self) -> list[DesignFuture]:
        """List designs."""
        return [
            DesignFuture(session=self.session, metadata=m)
            for m in designer.designs_list(session=self.session)
        ]

    def get_design(self, design_id: str) -> DesignFuture:
        """
        Get design.

        Args
        ____
        design_id: str
            ID of design to retrieve

        Returns
        _______
        DesignFuture:
            A future object representing the design job and its results.
        """
        return DesignFuture(
            session=self.session,
            metadata=designer.design_get(session=self.session, design_id=design_id),
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
        Start a protein design job based on your assaydata, a trained ML model and Criteria (specified here).

        Args
        ----------
        assay: AssayDataset
            The AssayDataset to design from.
        criteria: Criteria | Subcriterion | Criterion
            Criteria for evaluating the design. Use documentation for syntactic sugar to easily create.
        num_steps: int
            The number of steps in the genetic algo. Default is 8.
        pop_size: int
            The population size for the genetic algo. Default is 1024.
        n_offsprings: int
            The number of offspring for the genetic algo. Default is 5120.
        crossover_prob: float
            The crossover probability for the genetic algo. Default is 1.0.
        crossover_prob_pointwise: float
            The pointwise crossover probability for the genetic algo. Default is 0.2.
        mutation_average_mutations_per_seq: int
            The average number of mutations per sequence. Default is 1.
        allowed_tokens: DesignConstraint | dict[int, list[str]]
            A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Defaults to empty dict.

        Returns
        -------
        DesignFuture:
            A future object representing the design job and its results.
        """
        return DesignFuture.create(
            session=self.session,
            job=designer.designer_create_genetic_algorithm(
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
