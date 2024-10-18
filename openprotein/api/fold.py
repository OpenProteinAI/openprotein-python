from openprotein.api.embedding import ModelMetadata
from openprotein.base import APISession
from openprotein.schemas import FoldJob
from pydantic import TypeAdapter

PATH_PREFIX = "v1/fold"


def fold_models_list_get(session: APISession) -> list[str]:
    """
    List available fold models.

    Args:
        session (APISession): API session

    Returns:
        List[str]: list of model names.
    """

    endpoint = PATH_PREFIX + "/models"
    response = session.get(endpoint)
    result = response.json()
    return result


def fold_model_get(session: APISession, model_id: str) -> ModelMetadata:
    endpoint = PATH_PREFIX + f"/models/{model_id}"
    response = session.get(endpoint)
    result = response.json()
    return ModelMetadata(**result)


def fold_get_sequences(session: APISession, job_id: str) -> list[bytes]:
    """
    Get results associated with the given request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to fetch

    Returns
    -------
    sequences : List[bytes]
    """
    endpoint = PATH_PREFIX + f"/{job_id}/sequences"
    response = session.get(endpoint)
    return TypeAdapter(list[bytes]).validate_python(response.json())


def fold_get_sequence_result(
    session: APISession, job_id: str, sequence: bytes | str
) -> bytes:
    """
    Get encoded result for a sequence from the request ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    job_id : str
        job ID to retrieve results from
    sequence : bytes
        sequence to retrieve results for

    Returns
    -------
    result : bytes
    """
    if isinstance(sequence, bytes):
        sequence = sequence.decode()
    endpoint = PATH_PREFIX + f"/{job_id}/{sequence}"
    response = session.get(endpoint)
    return response.content


def fold_models_esmfold_post(
    session: APISession,
    sequences: list[bytes],
    num_recycles: int | None = None,
) -> FoldJob:
    """
    POST a request for structure prediction using ESMFold. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    sequences : List[bytes]
        sequences to request results for
    num_recycles : Optional[int]
        number of recycles for structure prediction

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + "/models/esmfold"

    sequences_unicode = [(s if isinstance(s, str) else s.decode()) for s in sequences]
    body: dict = {
        "sequences": sequences_unicode,
    }
    if num_recycles is not None:
        body["num_recycles"] = num_recycles

    response = session.post(endpoint, json=body)
    return FoldJob.model_validate(response.json())


def fold_models_alphafold2_post(
    session: APISession,
    msa_id: str,
    num_recycles: int | None = None,
    num_models: int = 1,
    num_relax: int = 0,
) -> FoldJob:
    """
    POST a request for structure prediction using AlphaFold2. Returns a Job object referring to this request
    that can be used to retrieve results later.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa_id : str
        ID of MSA to use for structure prediction. The first sequence in the MSA is the query sequence.
    num_recycles : Optional, int.
        Number of recycles for structure prediction. Default to nil which lets the system decide.
    num_models : int
        Number of models to predict. Defaults to 1.
    num_relax : int
        Number of relaxation iterations to run. Defaults to 0.

    Returns
    -------
    job : Job
    """
    endpoint = PATH_PREFIX + "/models/alphafold2"

    body = {
        "msa_id": msa_id,
        "num_models": num_models,
        "num_relax": num_relax,
    }
    if num_recycles is not None:
        body["num_recycles"] = num_recycles

    response = session.post(endpoint, json=body)
    # GET endpoint for AF2 expects the query sequence (first sequence) within the MSA
    # since we don't know what the is, leave the sequence out of the future to be retrieved when calling get()
    return FoldJob.model_validate(response.json())
