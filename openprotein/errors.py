from pydantic import BaseModel
from requests import Response

__all__ = [
    "APIError",
    "AuthError",
    "DeprecationError",
    "HTTPError",
    "InvalidJob",
    "InvalidParameterError",
    "JobFailedException",
    "JobOutputsNotReady",
    "MissingParameterError",
    "NoJobError",
    "NotPrepared",
    "RawAPIError",
    "TimeoutException",
]


_UPGRADE_MESSAGE = "If this issue persists, try upgrading the client: pip install --upgrade openprotein-python"


class InvalidParameterError(Exception):
    """Raised when a parameter has an invalid value."""

    def __init__(self, message="Invalid parameter"):
        self.message = message
        super().__init__(self.message)


class MissingParameterError(Exception):
    """Raised when a required parameter is not provided."""

    def __init__(self, message="Required parameter is missing"):
        self.message = message
        super().__init__(self.message)


class RawAPIError(BaseModel):
    detail: str


class APIError(Exception):
    """Raised when the API returns an error."""

    def __init__(self, message: str):
        self.message = f"{message}\n{_UPGRADE_MESSAGE}"
        super().__init__(self.message)


class HTTPError(APIError):
    def __init__(self, response: Response):
        self.response = response
        self.status_code = response.status_code
        self.text = response.text
        self.url = response.url
        message = (
            f"Status code {self.status_code}\non resource: {self.url}\n{self.text}"
        )
        super().__init__(message)


class JobOutputsNotReady(APIError):
    """A job's named output is not complete yet."""

    def __init__(self, message: str):
        self.message = message
        Exception.__init__(self, message)


class AuthError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message="Invalid authorization"):
        self.message = message
        super().__init__(self.message)


class InvalidJob(Exception):
    """Raised when a job cannot be found or is not valid for the request."""

    def __init__(self, message="No such job"):
        self.message = message
        super().__init__(self.message)


class NotPrepared(Exception):
    """A job's units are not listed yet."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        super().__init__(f"job {job_id} has not been prepared yet")


class TimeoutException(Exception):
    """Raised when waiting on a job or request times out."""

    def __init__(self, message="Request timed out!"):
        self.message = message
        super().__init__(self.message)


class JobFailedException(Exception):
    """Raised when retrieving results from a job that has failed."""

    def __init__(
        self,
        job_id: str,
        failure_message: str | None = None,
    ):
        self.job_id = job_id
        self.failure_message = failure_message
        message = f"Job {job_id} failed"
        if failure_message:
            message = f"{message}: {failure_message}"
        self.message = message
        super().__init__(self.message)


class NoJobError(AttributeError):
    """A stored row that no job made, such as an upload, has no job fields."""


class DeprecationError(Exception):
    """DeprecationError used for flagging to the user to not use this interface anymore."""

    def __init__(self, message="This API is deprecated and no longer supported"):
        self.message = message
        super().__init__(self.message)
