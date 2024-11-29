from typing import Generator

from openprotein.api import assaydata, designer
from openprotein.api import job as job_api
from openprotein.base import APISession
from openprotein.schemas import Criteria, Design, DesignJob
from openprotein.schemas.designer import DesignAlgorithm

from .assaydata import AssayDataset
from .futures import Future, StreamingFuture


class DesignFuture(StreamingFuture, Future):
    """Future Job for manipulating results"""

    job: DesignJob

    def __init__(
        self,
        session: APISession,
        job: DesignJob | None = None,
        metadata: Design | None = None,
    ):
        """Initializes with either job get or design get."""
        self._design_assay = None
        if metadata is None:
            if job is None:
                raise ValueError("Expected design metadata or job")
            metadata = designer.design_get(session=session, design_id=job.job_id)
        self._metadata = metadata
        if job is None:
            job = DesignJob.create(job_api.job_get(session=session, job_id=metadata.id))
        super().__init__(session, job)

    @property
    def id(self):
        return self._metadata.id

    @property
    def assay(self) -> AssayDataset:
        if self._design_assay is None:
            self._design_assay = self.get_assay()
        return self._design_assay

    @property
    def algorithm(self) -> DesignAlgorithm:
        return self._metadata.algorithm

    @property
    def criteria(self) -> Criteria:
        return self._metadata.criteria

    @property
    def num_steps(self):
        return self._metadata.num_steps

    @property
    def num_rows(self):
        return self._metadata.num_rows

    @property
    def allowed_tokens(self) -> dict[str, list[str]] | None:
        return self._metadata.allowed_tokens

    @property
    def pop_size(self) -> int:
        return self._metadata.pop_size

    @property
    def n_offsprings(self) -> int:
        return self._metadata.n_offsprings

    @property
    def crossover_prob(self) -> float:
        return self._metadata.crossover_prob

    @property
    def crossover_prob_pointwise(self) -> float:
        return self._metadata.crossover_prob_pointwise

    @property
    def mutation_average_mutations_per_seq(self) -> int:
        return self._metadata.mutation_average_mutations_per_seq

    @property
    def metadata(self):
        self._refresh_metadata()
        return self._metadata

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = designer.design_get(
                session=self.session, design_id=self._metadata.id
            )

    def delete(self) -> bool:
        """
        Delete this design.
        """
        return designer.design_delete(session=self.session, design_id=self.id)

    def stream(self) -> Generator:
        stream = designer.designer_get_design_results(
            session=self.session, design_id=self.id
        )
        return designer.decode_design_results_stream(data=stream)

    def get_assay(self) -> AssayDataset:
        """
        Get assay used for design job.

        Returns
        -------
            AssayDataset: Assay dataset used for design.
        """
        return AssayDataset(
            session=self.session,
            metadata=assaydata.get_assay_metadata(
                self.session, self._metadata.assay_id
            ),
        )
