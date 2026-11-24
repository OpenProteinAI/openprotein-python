"""Predictor models for making predictions on new sequences."""

from functools import cached_property
from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.data import AssayDataset, DataAPI
from openprotein.embeddings import EmbeddingModel
from openprotein.errors import InvalidParameterError
from openprotein.jobs import Future, JobsAPI, JobType
from openprotein.models.kernels import (
    TrainedKernel,
    _GPHyperparameters,
    _sole_hyperparameters,
    _trained_kernel,
)
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

__all__ = ["PredictorModel", "PredictorModelGroup"]


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
        self._hyperparameters: _GPHyperparameters | None = None

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
    def name(self):
        """The name of the predictor."""
        return self._metadata.name

    @property
    def description(self):
        """The description of the predictor."""
        return self._metadata.description

    def rename(self, name: str | None = None, description: str | None = None):
        """Not supported: a predictor's name is set when it is trained."""
        raise NotImplementedError("a predictor's name is set when it is trained")

    @property
    def reduction(self):
        """The reduction of the embeddings used to train the predictor, if any."""
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

    def _get(self, verbose: bool = False):
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
        return data_api.Assay(id=self._metadata.training_dataset.assay_id)

    def crossvalidate(self, n_splits: int | None = None) -> CVResultFuture:
        """
        Run a crossvalidation on the trained predictor.

        Parameters
        ----------
        n_splits : int, optional
            Number of folds.

        Returns
        -------
        CVResultFuture
            Future for the crossvalidation job.
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

    def _get_hyperparameters(self) -> _GPHyperparameters:
        if self._hyperparameters is None:
            self._hyperparameters = _sole_hyperparameters(
                api._predictor_get_hyperparameters(
                    session=self.session, predictor_id=self.id
                )
            )
        return self._hyperparameters

    # IPython can evaluate this descriptor to complete the concrete kernel's fields.
    @cached_property
    def kernel(self) -> TrainedKernel:
        """
        The trained kernel of this GP predictor, in data space.

        The first access, including runtime tab completion, may make a request.
        Failed reads are retried on subsequent access.

        Its class follows the kernel type and carries only the hyperparameters that
        apply to it: every kernel has an ``outputscale``, every kernel but linear a
        ``lengthscale``, periodic a ``period`` and rational quadratic an ``alpha``.

        Raises
        ------
        HTTPError
            404 for an ensemble, or a GP trained before hyperparameters were recorded.

        Examples
        --------
        >>> predictor.kernel
        PeriodicKernel(outputscale=1.92, lengthscale=0.83, period=2.31)
        >>> predictor.kernel.lengthscale
        0.83
        """
        hyperparameters = self._get_hyperparameters()
        kernel_spec = self._metadata.model_spec.kernel
        assert kernel_spec is not None
        try:
            return _trained_kernel(kernel_spec.type, hyperparameters)
        except ValueError:
            self._hyperparameters = None
            raise

    @property
    def mean(self) -> float | None:
        """
        Constant mean of the GP prior, in the units of the trained property.

        Raises
        ------
        HTTPError
            404 for an ensemble, or a GP trained before hyperparameters were recorded.
        """
        return self._get_hyperparameters().mean

    @property
    def noise(self) -> float | None:
        """
        Observation noise variance, in the units of the trained property squared.

        Raises
        ------
        HTTPError
            404 for an ensemble, or a GP trained before hyperparameters were recorded.
        """
        return self._get_hyperparameters().noise

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

    def _get(self, verbose: bool = False):
        """
        Returns the predictor model.

        :meta private:
        """
        return self

    def delete(self):
        return api.predictor_delete(session=self.session, predictor_id=self.id)
