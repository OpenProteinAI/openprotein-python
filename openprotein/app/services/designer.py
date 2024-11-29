from openprotein.api import designer
from openprotein.app.models import AssayDataset, DesignFuture
from openprotein.base import APISession
from openprotein.schemas import Criteria, Criterion, DesignConstraint, Subcriterion


class DesignerAPI:
    """interface for calling Designer endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def list_designs(self) -> list[DesignFuture]:
        return [
            DesignFuture(session=self.session, metadata=m)
            for m in designer.designs_list(session=self.session)
        ]

    def get_design(self, design_id: str) -> DesignFuture:
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

        Parameters
        ----------
        design_job : DesignJobCreate
            The details of the design job to be created, with the following parameters:
            - assay_id: The ID for the assay.
            - criteria: Criteria for evaluating the design.
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
