from openprotein.api import assaydata
from openprotein.api import job as job_api
from openprotein.api import predictor, svd
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import PredictorMetadata, TrainJob

from ..assaydata import AssayDataset
from ..embeddings import EmbeddingModel
from ..futures import Future
from ..svd import SVDModel
from .predict import PredictionResultFuture


class PredictorModel(Future):
    """
    Class providing predict endpoint for fitted predictor models.

    Also implements a Future that waits for train job.
    """

    job: TrainJob

    def __init__(
        self,
        session: APISession,
        job: TrainJob | None = None,
        metadata: PredictorMetadata | None = None,
    ):
        """Initializes with either job get or svd metadata get."""
        self._training_assay = None
        # initialize the metadata
        if metadata is None:
            if job is None:
                raise ValueError("Expected predictor metadata or job")
            metadata = predictor.predictor_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            job = TrainJob.create(job_api.job_get(session=session, job_id=metadata.id))
        # getter initializes job if not provided
        super().__init__(session, job)

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self):
        return self.metadata.id

    @property
    def reduction(self):
        return self.metadata.model_spec.features.reduction

    @property
    def sequence_length(self):
        if (constraints := self.metadata.model_spec.constraints) is not None:
            return constraints.sequence_length
        return None

    @property
    def training_assay(self) -> AssayDataset:
        if self._training_assay is None:
            self._training_assay = self.get_assay()
        return self._training_assay

    @property
    def training_properties(self) -> list[str]:
        return self.metadata.training_dataset.properties

    @property
    def metadata(self):
        self._refresh_metadata()
        return self._metadata

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = predictor.predictor_get(self.session, self._metadata.id)

    def get_model(self) -> EmbeddingModel | SVDModel | None:
        """Fetch embeddings model"""
        if (features := self._metadata.model_spec.features) and (
            model_id := features.model_id
        ) is None:
            return None
        elif features.type.upper() == "PLM":
            model = EmbeddingModel.create(session=self.session, model_id=model_id)
        elif features.type.upper() == "SVD":
            model = SVDModel(
                session=self.session,
                metadata=svd.svd_get(session=self.session, svd_id=model_id),
            )
        else:
            raise ValueError(f"Unexpected feature type {features.type}")
        return model

    @property
    def model(self) -> EmbeddingModel | SVDModel | None:
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this SVD model.
        """
        return predictor.predictor_delete(self.session, self.id)

    def get(self, verbose: bool = False):
        # overload for Future
        return self

    def get_assay(self) -> AssayDataset:
        """
        Get assay used for train job.

        Returns
        -------
            list[bytes]: list of sequences
        """
        return AssayDataset(
            session=self.session,
            metadata=assaydata.get_assay_metadata(
                self.session, self._metadata.training_dataset.assay_id
            ),
        )

    def predict(self, sequences: list[bytes] | list[str]) -> PredictionResultFuture:
        if self.sequence_length is not None:
            for sequence in sequences:
                sequence = sequence if isinstance(sequence, str) else sequence.decode()
                if len(sequence) != self.sequence_length:
                    raise InvalidParameterError(
                        f"Expected sequence to predict to be of length {self.sequence_length}"
                    )
        return PredictionResultFuture.create(
            session=self.session,
            job=predictor.predictor_predict_post(
                session=self.session, predictor_id=self.id, sequences=sequences
            ),
        )
