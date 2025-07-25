"""Predictor models for making predictions on new sequences."""

from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.data import AssayDataset, DataAPI
from openprotein.embeddings import EmbeddingModel, EmbeddingsAPI
from openprotein.errors import InvalidParameterError
from openprotein.jobs import Future, JobsAPI, JobType
from openprotein.svd import SVDAPI, SVDModel

from . import api
from .prediction import PredictionResultFuture
from .schemas import (
    PredictorEnsembleJob,
    PredictorMetadata,
    PredictorTrainJob,
    PredictorType,
)
from .validate import CVResultFuture

if TYPE_CHECKING:
    from openprotein.design import ModelCriterion


class PredictorModel(Future):
    """
    Class providing predict endpoint for fitted predictor models.

    Also implements a Future that waits for train job.
    """

    job: PredictorTrainJob | None

    def __init__(
        self,
        session: APISession,
        job: PredictorTrainJob | PredictorEnsembleJob | None = None,
        metadata: PredictorMetadata | None = None,
    ):
        """
        Construct a predictor model.

        Takes in either a train job, or the predictor metadata.

        :meta private:
        """
        self._training_assay = None

        # initialize the predictor metadata
        if metadata is None:
            if job is None or job.job_id is None:
                raise ValueError("Expected predictor metadata or job")
            metadata = api.predictor_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            if metadata.model_spec.type != PredictorType.ENSEMBLE:
                jobs_api = getattr(session, "jobs", None)
                assert isinstance(jobs_api, JobsAPI)
                job = PredictorTrainJob.create(jobs_api.get_job(job_id=metadata.id))
            else:
                job = PredictorEnsembleJob(
                    created_date=self._metadata.created_date,
                    status=self._metadata.status,
                    job_type=JobType.predictor_train,
                )
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

    def __lt__(self, target: float) -> "ModelCriterion":
        from openprotein.design import ModelCriterion

        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.lt
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    def __gt__(self, target: float) -> "ModelCriterion":
        from openprotein.design import ModelCriterion

        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.gt
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    def __eq__(self, target: float) -> "ModelCriterion":
        from openprotein.design import ModelCriterion

        if len(self.training_properties) == 1:
            return ModelCriterion(
                model_id=self.id,
                measurement_name=self.training_properties[0],
                criterion=ModelCriterion.Criterion(
                    target=target, direction=ModelCriterion.Criterion.DirectionEnum.eq
                ),
            )
        raise self.InvalidMultitaskModelToCriterion()

    class InvalidMultitaskModelToCriterion(Exception):
        """
        Exception raised when trying to create model criterion from multitask predictor.

        :meta private:
        """

    @property
    def id(self):
        """ID of predictor."""
        return self._metadata.id

    @property
    def reduction(self):
        """The reduction of th embeddings used to train the predictor, if any."""
        return (
            self._metadata.model_spec.features.reduction
            if self._metadata.model_spec.features is not None
            else None
        )

    @property
    def sequence_length(self):
        """The sequence length constraint on the predictor, if any."""
        if (constraints := self._metadata.model_spec.constraints) is not None:
            return constraints.sequence_length
        return None

    @property
    def training_assay(self) -> AssayDataset:
        """The assay the predictor was trained on."""
        if self._training_assay is None:
            self._training_assay = self.get_assay()
        return self._training_assay

    @property
    def training_properties(self) -> list[str]:
        """The list of properties the predictor was trained on."""
        return self._metadata.training_dataset.properties

    @property
    def metadata(self):
        """The predictor metadata."""
        self._refresh_metadata()
        return self._metadata

    def _refresh_metadata(self):
        if not self._metadata.is_done():
            self._metadata = api.predictor_get(self.session, self._metadata.id)

    def get_model(self) -> EmbeddingModel | SVDModel | None:
        """Retrieve the embeddings or SVD model used to create embeddings to train on."""
        if (
            (features := self._metadata.model_spec.features)
            and (model_id := features.model_id) is None
            or features is None
        ):
            return None
        elif features.type.upper() == "PLM":
            model = EmbeddingModel.create(session=self.session, model_id=model_id)
        elif features.type.upper() == "SVD":
            svd_api = getattr(self.session, "svd", None)
            assert isinstance(svd_api, SVDAPI)
            model = svd_api.get_svd(svd_id=model_id)
        else:
            raise ValueError(f"Unexpected feature type {features.type}")
        return model

    @property
    def model(self) -> EmbeddingModel | SVDModel | None:
        """The embeddings or SVD model used to create embeddings to train on."""
        return self.get_model()

    def delete(self) -> bool:
        """
        Delete this predictor model.
        """
        return api.predictor_delete(self.session, self.id)

    def get(self, verbose: bool = False):
        """
        Returns the train loss curves.
        """
        return self.metadata.traingraphs

    def get_assay(self) -> AssayDataset:
        """
        Get assay used for train job.

        Returns
        -------
            AssayDataset: Assay dataset used for train job.
        """
        data_api = getattr(self.session, "data", None)
        assert isinstance(data_api, DataAPI)
        return data_api.get(assay_id=self._metadata.training_dataset.assay_id)

    def crossvalidate(self, n_splits: int | None = None) -> CVResultFuture:
        """
        Run a crossvalidation on the trained predictor.
        """
        return CVResultFuture.create(
            session=self.session,
            job=api.predictor_crossvalidate_post(
                session=self.session,
                predictor_id=self.id,
                n_splits=n_splits,
            ),
        )

    def predict(self, sequences: list[bytes] | list[str]) -> PredictionResultFuture:
        """
        Make predictions about the trained properties for a list of sequences.
        """
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
            job=api.predictor_predict_post(
                session=self.session, predictor_id=self.id, sequences=sequences
            ),
        )

    def single_site(self, sequence: bytes | str) -> PredictionResultFuture:
        """
        Compute the single-site mutated predictions of a base sequence.
        """
        if self.sequence_length is not None:
            # convert to string to check token length
            seq = sequence if isinstance(sequence, str) else sequence.decode()
            if len(seq) != self.sequence_length:
                raise InvalidParameterError(
                    f"Expected sequence to predict to be of length {self.sequence_length}"
                )
        return PredictionResultFuture.create(
            session=self.session,
            job=api.predictor_predict_single_site_post(
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
        """
        Make predictions about the trained properties for a list of sequences.
        """
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
            job=api.predictor_predict_multi_post(
                session=self.session,
                predictor_ids=[m.id for m in self.__models__],
                sequences=sequences,
            ),
        )

    def single_site(self, sequence: bytes | str) -> PredictionResultFuture:
        """
        Compute the single-site mutated predictions of a base sequence.
        """
        if self.sequence_length is not None:
            # convert to string to check token length
            seq = sequence if isinstance(sequence, str) else sequence.decode()
            if len(seq) != self.sequence_length:
                raise InvalidParameterError(
                    f"Expected sequence to predict to be of length {self.sequence_length}"
                )
        return PredictionResultFuture.create(
            session=self.session,
            job=api.predictor_predict_single_site_post(
                session=self.session, predictor_id=self.id, base_sequence=sequence
            ),
        )

    def get(self, verbose: bool = False):
        """
        Returns the predictor model.

        :meta private:
        """
        return self

    def delete(self):
        return api.predictor_delete(session=self.session, predictor_id=self.id)
