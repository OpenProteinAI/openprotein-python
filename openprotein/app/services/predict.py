import logging

from openprotein.api import predict
from openprotein.app.models import PredictFuture, TrainFuture
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError

logger = logging.getLogger(__name__)


class PredictService:
    """interface for calling Predict endpoints"""

    def __init__(self, session: APISession):
        """
        Initialize a new instance of the PredictService class.

        Parameters
        ----------
        session : APISession
            APIsession with auth
        """
        self.session = session

    def create_predict_job(
        self,
        sequences: list[str],
        train_job: TrainFuture,
        model_ids: list[str] | None = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for a given list of sequences and a trained model.

        Parameters
        ----------
        sequences : List
            The list of sequences to be used for the Predict job.
        train_job : Any
            The train job object representing the trained model.
        model_ids : List[str], optional
            The list of model ids to be used for Predict. Default is None.

        Returns
        -------
        PredictFuture
            The job object representing the Predict job.

        Raises
        ------
        InvalidParameterError
            If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
        InvalidParameterError
            If BOTH train_job and model_ids are specified
        InvalidParameterError
            If NEITHER train_job or model_ids is specified
        APIError
            If the backend refuses the job (due to sequence length or invalid inputs)
        """
        if train_job.assay.sequence_length is not None:
            if any([train_job.assay.sequence_length != len(s) for s in sequences]):
                raise InvalidParameterError(
                    f"Predict sequences length {len(sequences[0])}  != training assaydata ({train_job.assay.sequence_length})"
                )
        if not train_job.done():
            logger.warning(
                f"Potential error: Training job has status {train_job.status}"
            )
            # raise InvalidParameterError(
            #    f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            # )

        return PredictFuture.create(
            session=self.session,
            job=predict.create_predict_job(
                session=self.session,
                sequences=sequences,
                train_job_id=train_job.id,
                model_ids=model_ids,
            ),
        )

    def create_predict_single_site(
        self,
        sequence: str,
        train_job: TrainFuture,
        model_ids: list[str] | None = None,
    ) -> PredictFuture:
        """
        Creates a new Predict job for single site mutation analysis with a trained model.

        Parameters
        ----------
        sequence : str
            The sequence for single site analysis.
        train_job : Any
            The train job object representing the trained model.
        model_ids : List[str], optional
            The list of model ids to be used for Predict. Default is None.

        Returns
        -------
        PredictFuture
            The job object representing the Predict job.

        Raises
        ------
        InvalidParameterError
            If the sequences are not of the same length as the assay data or if the train job has not completed successfully.
        InvalidParameterError
            If BOTH train_job and model_ids are specified
        InvalidParameterError
            If NEITHER train_job or model_ids is specified
        APIError
            If the backend refuses the job (due to sequence length or invalid inputs)
        """
        if train_job.assay is not None:
            if train_job.assay.sequence_length is not None:
                if any([train_job.assay.sequence_length != len(sequence)]):
                    raise InvalidParameterError(
                        f"Predict sequences length {len(sequence)}  != training assaydata ({train_job.assay.sequence_length})"
                    )
        train_job.refresh()
        if not train_job.done():
            logger.warning(
                f"Potential error: Training job has status {train_job.status}"
            )
            # raise InvalidParameterError(
            #    f"train job has status {train_job.status.value}, Predict requires status SUCCESS"
            # )

        return PredictFuture.create(
            session=self.session,
            job=predict.create_predict_single_site(
                session=self.session,
                sequence=sequence,
                train_job_id=train_job.id,
                model_ids=model_ids,
            ),
        )
