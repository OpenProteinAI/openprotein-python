# Store for Model and Future classes
from openprotein.jobs import job_get, ResultsParser
from typing import Optional, Any


class FutureBase:
    """Base class for all Future classes.

    This class needs to be directly inherited for class discovery."""

    # overridden by subclasses
    job_type: Optional[Any] = None

    @classmethod
    def get_job_type(cls):
        """Return the job type associated with this Future class."""

        if isinstance(cls.job_type, str):
            return [cls.job_type]
        return cls.job_type


class FutureFactory:
    """Factory class for creating Future instances based on job_type."""

    @staticmethod
    def create_future(
        session, job_id: Optional[str] = None, response: Optional[dict] = None, **kwargs
    ):
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Parameters:
        - job: The job object containing the job_type attribute.
        - session: sess for API interactions.
        - **kwargs: Additional keyword arguments to pass to the Future class constructor.

        Returns:
        - An instance of the appropriate Future class.
        """

        # parse job
        if job_id:
            job = job_get(session, job_id)
        else:
            if "job" not in kwargs:
                job = ResultsParser.parse_obj(response)
            else:
                job = kwargs.pop("job")

        # Dynamically discover all subclasses of FutureBase
        future_classes = FutureBase.__subclasses__()
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Find the Future class that matches the job type
        for future_class in future_classes:
            if job.job_type in future_class.get_job_type():
                return future_class(session=session, job=job, **kwargs)  # type: ignore

        raise ValueError(f"Unsupported job type: {job.job_type}")
