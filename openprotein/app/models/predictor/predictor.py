from openprotein.api import assaydata
from openprotein.api import job as job_api
from openprotein.api import predictor, svd
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError
from openprotein.schemas import Criterion, ModelCriterion, PredictorMetadata, TrainJob

from ..assaydata import AssayDataset
from ..embeddings import EmbeddingModel
from ..futures import Future
from ..svd import SVDModel
from .predict import PredictionResultFuture
from .validate import CVResultFuture


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
        """Initializes with either job get or predictor get."""
        self._training_assay = None
        # initialize the metadata
        if metadata is None:
            if job is None:
                raise ValueError("Expected predictor metadata or job")
            metadata = predictor.predictor_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            job = TrainJob.create(job_api.job_get(session=session, job_id=metadata.id))
        super().__init__(session, job)

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    def __or__(self, model: "PredictorModel") -> "PredictorModelGroup":
        if self.sequence_length is not None:
            if model.sequence_length != self.sequence_length:
                raise ValueError(
                    "Expected sequence lengths to either match or be None."
                )
        return PredictorModelGroup(
            session=self.session,
            models=[self, model],
            sequence_length=self.sequence_length or model.sequence_length,
            check_sequence_length=False,
        )

    def __lt__(self, target: float) -> ModelCriterion:
        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.lt
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    def __gt__(self, target: float) -> ModelCriterion:
        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.gt
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    def __eq__(self, target: float) -> ModelCriterion:
        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.eq
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    class InvalidMultitaskModelToCriterion(Exception): ...

    @property
    def id(self):
        return self._metadata.id

    @property
    def reduction(self):
        return self._metadata.model_spec.features.reduction

    @property
    def sequence_length(self):
        if (constraints := self._metadata.model_spec.constraints) is not None:
            return constraints.sequence_length
        return None

    @property
    def training_assay(self) -> AssayDataset:
        if self._training_assay is None:
            self._training_assay = self.get_assay()
        return self._training_assay

    @property
    def training_properties(self) -> list[str]:
        return self._metadata.training_dataset.properties

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
        Delete this predictor model.
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
            AssayDataset: Assay dataset used for train job.
        """
        return AssayDataset(
            session=self.session,
            metadata=assaydata.get_assay_metadata(
                self.session, self._metadata.training_dataset.assay_id
            ),
        )

    def crossvalidate(self, n_splits: int | None = None) -> CVResultFuture:
        return CVResultFuture.create(
            session=self.session,
            job=predictor.predictor_crossvalidate_post(
                session=self.session,
                predictor_id=self.id,
                n_splits=n_splits,
            ),
        )

    def predict(self, sequences: list[bytes] | list[str]) -> PredictionResultFuture:
        if self.sequence_length is not None:
            for sequence in sequences:
                # convert to string to check token length
                sequence = sequence if isinstance(sequence, str) else sequence.decode()
                if len(sequence) != self.sequence_length:
                    raise InvalidParameterError(
                        f"Expected sequences to predict to be of length {self.sequence_length}"
                    )
        return PredictionResultFuture.create(
            session=self.session,
            job=predictor.predictor_predict_post(
                session=self.session, predictor_id=self.id, sequences=sequences
            ),
        )

    def single_site(self, sequence: bytes | str) -> PredictionResultFuture:
        if self.sequence_length is not None:
            # convert to string to check token length
            seq = sequence if isinstance(sequence, str) else sequence.decode()
            if len(seq) != self.sequence_length:
                raise InvalidParameterError(
                    f"Expected sequence to predict to be of length {self.sequence_length}"
                )
        return PredictionResultFuture.create(
            session=self.session,
            job=predictor.predictor_predict_single_site_post(
                session=self.session, predictor_id=self.id, base_sequence=sequence
            ),
        )


class PredictorModelGroup(Future):
    """
    Class providing predict endpoint for fitted predictor models.

    Also implements a Future that waits for train job.
    """

    __models__: list[PredictorModel]

    def __init__(
        self,
        session: APISession,
        models: list[PredictorModel],
        sequence_length: int | None = None,
        check_sequence_length: bool = True,  # turn off checking - prevent n^2 operation when chaining many
    ):
        if len(models) == 0:
            raise ValueError("Expected at least one model to group")
        # calculate and check sequence length compatibility
        if check_sequence_length:
            for m in models:
                if m.sequence_length is not None:
                    if sequence_length is None:
                        sequence_length = m.sequence_length
                    elif sequence_length != m.sequence_length:
                        raise ValueError(
                            "Expected sequence lengths of all models to either match or be None."
                        )
        self.sequence_length = sequence_length
        self.session = session
        self.__models__ = models

    def __str__(self) -> str:
        return repr(self.__models__)

    def __repr__(self) -> str:
        return repr(self.__models__)

    def __or__(self, model: PredictorModel) -> "PredictorModelGroup":
        if self.sequence_length is not None:
            if model.sequence_length != self.sequence_length:
                raise ValueError(
                    "Expected sequence lengths to either match or be None."
                )
        return PredictorModelGroup(
            session=self.session,
            models=self.__models__ + [model],
            sequence_length=self.sequence_length or model.sequence_length,
            check_sequence_length=False,
        )

    def predict(self, sequences: list[bytes] | list[str]) -> PredictionResultFuture:
        if self.sequence_length is not None:
            for sequence in sequences:
                # convert to string to check token length
                sequence = sequence if isinstance(sequence, str) else sequence.decode()
                if len(sequence) != self.sequence_length:
                    raise InvalidParameterError(
                        f"Expected sequences to predict to be of length {self.sequence_length}"
                    )
        return PredictionResultFuture.create(
            session=self.session,
            job=predictor.predictor_predict_multi_post(
                session=self.session,
                predictor_ids=[m.id for m in self.__models__],
                sequences=sequences,
            ),
        )

    def single_site(self, sequence: bytes | str) -> PredictionResultFuture:
        if self.sequence_length is not None:
            # convert to string to check token length
            seq = sequence if isinstance(sequence, str) else sequence.decode()
            if len(seq) != self.sequence_length:
                raise InvalidParameterError(
                    f"Expected sequence to predict to be of length {self.sequence_length}"
                )
        return PredictionResultFuture.create(
            session=self.session,
            job=predictor.predictor_predict_single_site_post(
                session=self.session, predictor_id=self.id, base_sequence=sequence
            ),
        )

    def get(self, verbose: bool = False):
        # overload for Future
        return self
