from openprotein.api import train
from openprotein.app.models import AssayDataset, TrainFuture
from openprotein.base import APISession
from openprotein.errors import InvalidParameterError


class TrainingAPI:
    """API interface for calling Train endpoints"""

    def __init__(
        self,
        session: APISession,
    ):
        self.session = session
        self.assay = None

    def create_training_job(
        self,
        assaydataset: AssayDataset,
        measurement_name: str | list[str],
        model_name: str = "",
        force_preprocess: bool = False,
    ) -> TrainFuture:
        """
        Create a training job on your data.

        This function validates the inputs, formats the data, and sends the job.

        Parameters
        ----------
        assaydataset : AssayDataset
            An AssayDataset object from which the assay_id is extracted.
        measurement_name : str or List[str]
            The name(s) of the measurement(s) to be used in the training job.
        model_name : str, optional
            The name to give the model.
        force_preprocess : bool, optional
            If set to True, preprocessing is forced even if data already exists.

        Returns
        -------
        TrainFuture
            A TrainFuture Job

        Raises
        ------
        InvalidParameterError
            If the `assaydataset` is not an AssayDataset object,
            If any measurement name provided does not exist in the AssayDataset,
            or if the AssayDataset has fewer than 3 data points.
        HTTPError
            If the request to the server fails.
        """
        if isinstance(measurement_name, str):
            measurement_name = [measurement_name]

        # ensure measurements exist in dataset
        for measurement in measurement_name:
            if measurement not in assaydataset.measurement_names:
                raise InvalidParameterError(f"No {measurement} in measurement names")

        # ensure assaydataset is large enough
        if assaydataset.shape[0] < 3:
            raise InvalidParameterError(
                "Assaydata must have >=3 data points for training!"
            )

        return TrainFuture.create(
            session=self.session,
            job=train.create_train_job(
                session=self.session,
                assay_id=assaydataset.id,
                measurement_name=measurement_name,
                model_name=model_name,
                force_preprocess=force_preprocess,
            ),
        )

    def _create_training_job_br(
        self,
        assaydataset: AssayDataset,
        measurement_name: str | list[str],
        model_name: str = "",
        force_preprocess: bool = False,
    ) -> TrainFuture:
        """Same as create_training_job."""
        return TrainFuture.create(
            session=self.session,
            job=train._create_train_job_br(
                session=self.session,
                assay_id=assaydataset.id,
                measurement_name=measurement_name,
                model_name=model_name,
                force_preprocess=force_preprocess,
            ),
        )

    def _create_training_job_gp(
        self,
        assaydataset: AssayDataset,
        measurement_name: str | list[str],
        model_name: str = "",
        force_preprocess: bool = False,
    ) -> TrainFuture:
        """Same as create_training_job."""
        return TrainFuture.create(
            session=self.session,
            job=train._create_train_job_gp(
                session=self.session,
                assay_id=assaydataset.id,
                measurement_name=measurement_name,
                model_name=model_name,
                force_preprocess=force_preprocess,
            ),
        )

    def get_training_results(self, job_id: str) -> TrainFuture:
        """
        Get training results (e.g. loss etc).

        Parameters
        ----------
        job_id : str
            job_id to get


        Returns
        -------
        TrainFuture
            A TrainFuture Job
        """
        train_job = train.get_training_results(self.session, job_id)
        return TrainFuture.create(
            session=self.session, job=train_job, traingraph=train_job.traingraph
        )
