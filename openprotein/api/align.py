import io
import random
from typing import BinaryIO, Iterator

import requests

import openprotein.config as config
from openprotein.base import APISession
from openprotein.csv import csv_stream
from openprotein.errors import APIError, InvalidParameterError, MissingParameterError
from openprotein.schemas import AlignType, Job, MSASamplingMethod


def get_align_job_inputs(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> requests.Response:
    """
    Get MSA and related data for an align job.

    Returns either the original user seed (RAW), the generated MSA or the prompt.

    Specify prompt_index to retreive the specific prompt for each replicate when input_type is PROMPT.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : int or str
        The job identifier.
    input_type : PoetInputType
        The type of MSA data.
    prompt_index : Optional[int]
        The replicate number for the prompt (input_type=-PROMPT only)

    Returns
    -------
    requests.Response
        The response from the server.
    """
    endpoint = "v1/align/inputs"

    params = {"job_id": job_id, "msa_type": input_type}
    if prompt_index is not None:
        params["replicate"] = prompt_index

    response = session.get(endpoint, params=params, stream=True)
    return response


def get_input(
    session: APISession,
    job: Job,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> Iterator[list[str]]:
    """
    Get input data for a given job.

    Parameters
    ----------
    session : APISession
        The API session.
    job : Job
        The job for which to retrieve data.
    input_type : PoetInputType
        The type of MSA data.
    prompt_index : Optional[int]
        The replicate number for the prompt (input_type=-PROMPT only)

    Returns
    -------
    csv.reader
        A CSV reader for the response data.
    """
    job_id = job.job_id
    response = get_align_job_inputs(
        session=session, job_id=job_id, input_type=input_type, prompt_index=prompt_index
    )
    return csv_stream(response)


def get_seed(session: APISession, job: Job) -> Iterator[list[str]]:
    """
    Get the seed for a given MSA job.

    Parameters
    ----------
    session : APISession
        The API session.
    job : Job
        The job for which to retrieve the seed.

    Returns
    -------
    Iterator[list[str]]
        A CSV reader for the seed sequence.
    """
    return get_input(session=session, job=job, input_type=AlignType.INPUT)


def get_msa(session: APISession, job: Job) -> Iterator[list[str]]:
    """
    Get the generated MSA (Multiple Sequence Alignment) for a given job.

    Parameters
    ----------
    session : APISession
        The API session.
    job : Job
        The job for which to retrieve the MSA.

    Returns
    -------
    Iterator[list[str]]
        A CSV reader for the MSA data.
    """
    return get_input(session=session, job=job, input_type=AlignType.MSA)


def msa_post(
    session: APISession,
    msa_file: BinaryIO | None = None,
    seed: str | bytes | None = None,
) -> Job:
    """
    Create an MSA.

    Either via a seed sequence (which will trigger MSA creation) or a ready-to-use MSA (via msa_file).

    Note that seed and msa_file are mutually exclusive, and one or the other must be set.

    Parameters
    ----------
    session : APISession

    msa_file : BinaryIO, Optional
        Ready-made MSA file. Defaults to None.
    seed : str | bytes, optional
        Seed sequence to trigger MSA job. Defaults to None.

    Raises
    ------
    Exception
        If msa_file and seed are both None.

    Returns
    -------
    Job
        Job details.
    """
    if (msa_file is None and seed is None) or (
        msa_file is not None and seed is not None
    ):
        raise MissingParameterError("seed OR msa_file must be provided.")
    endpoint = "v1/align/msa"

    is_seed = False
    if seed is not None:
        seed = seed.encode() if isinstance(seed, str) else seed
        msa_file = io.BytesIO(b"\n".join([b">seed", seed]))
        is_seed = True

    params = {"is_seed": is_seed}
    files = {"msa_file": msa_file}

    response = session.post(endpoint, files=files, params=params)
    return Job.model_validate(response.json())


# TODO - document the `ep` and `op` parameters
def mafft_post(
    session: APISession,
    sequence_file: BinaryIO,
    auto: bool = True,
    ep: float | None = None,
    op: float | None = None,
) -> Job:
    """
    Align sequences using the `mafft` algorithm. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

    Set auto to True to automatically attempt the best params. Leave a parameter as None to use system defaults.

    Parameters
    ----------
    session : APISession
    sequence_file : BinaryIO
        Sequences to align in fasta or csv format.
    auto : bool = True, optional
        Set to true to automatically set algorithm parameters.
    ep : float, optional
        mafft parameter
    op : float, optional
        mafft parameter

    Returns
    -------
    Job
        Job details.
    """
    endpoint = "v1/align/mafft"

    files = {"file": sequence_file}
    params: dict = {"auto": auto}
    if ep is not None:
        params["ep"] = ep
    if op is not None:
        params["op"] = op

    response = session.post(endpoint, files=files, params=params)
    return Job.model_validate(response.json())


# TODO - document the `clustersize` and `iterations` parameters
def clustalo_post(
    session: APISession,
    sequence_file: BinaryIO,
    clustersize: int | None = None,
    iterations: int | None = None,
) -> Job:
    """
    Align sequences using the `clustal omega` algorithm. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

    Leave a parameter as None to use system defaults.

    Parameters
    ----------
    session : APISession
    sequence_file : BinaryIO
        Sequences to align in fasta or csv format.
    clustersize : int, optional
        clustal omega parameter
    iterations : int, optional
        clustal omega parameter

    Returns
    -------
    Job
        Job details.
    """
    endpoint = "v1/align/clustalo"

    files = {"file": sequence_file}
    params = {}
    if clustersize is not None:
        params["clustersize"] = clustersize
    if iterations is not None:
        params["iterations"] = iterations

    response = session.post(endpoint, files=files, params=params)
    return Job.model_validate(response.json())


def abnumber_post(
    session: APISession,
    sequence_file: BinaryIO,
    scheme: str = "imgt",
) -> Job:
    """
    Align antibody using `AbNumber`. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

    The antibody numbering scheme can be specified from `imgt` (default), `chothia`, `kabat`, or `aho`.

    Parameters
    ----------
    session : APISession
    sequence_file : BinaryIO
        Sequences to align in fasta or csv format.
    scheme : str = 'imgt'
        Antibody numbering scheme. Can be one of 'imgt', 'chothia', 'kabat', or 'aho'

    Returns
    -------
    Job
        Job details.
    """
    endpoint = "v1/align/abnumber"

    valid_schemes = ["imgt", "chothia", "kabat", "aho"]
    if scheme not in valid_schemes:
        raise Exception(
            f"Antibody numbering {scheme} not recognized. Must be one of {valid_schemes}."
        )

    files = {"file": sequence_file}
    params = {"scheme": scheme}

    response = session.post(endpoint, files=files, params=params)
    return Job.model_validate(response.json())


# TODO - implement support for getting the antibody numbering from an `AbNumber` job
def antibody_schema_get(session: APISession, job_id: str):
    """
    Return the antibody numbering for an `AbNumber` job.
    """
    raise NotImplementedError()


def prompt_post(
    session: APISession,
    msa_id: str,
    num_sequences: int | None = None,
    num_residues: int | None = None,
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT,
    homology_level: float = 0.8,
    max_similarity: float = 1.0,
    min_similarity: float = 0.0,
    always_include_seed_sequence: bool = False,
    num_ensemble_prompts: int = 1,
    random_seed: int | None = None,
) -> Job:
    """
    Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET Jobs.

    The MSA is specified by msa_id and created in msa_post.

    Parameters
    ----------
    session : APISession
        An instance of APISession to manage interactions with the API.
    msa_id : str
        The ID of the Multiple Sequence Alignment to use for the prompt.
    num_sequences : int, optional
        Maximum number of sequences in the prompt. Must be <100.
    num_residues : int, optional
        Maximum number of residues (tokens) in the prompt. Must be less than 24577.
    method : MSASamplingMethod, optional
        Method to use for MSA sampling. Defaults to NEIGHBORS_NONGAP_NORM_NO_LIMIT.
    homology_level : float, optional
        Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Defaults to 0.8.
    max_similarity : float, optional
        Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 1.0.
    min_similarity : float, optional
        Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Defaults to 0.0.
    always_include_seed_sequence : bool, optional
        Whether to always include the seed sequence in the MSA. Defaults to False.
    num_ensemble_prompts : int, optional
        Number of ensemble jobs to run. Defaults to 1.
    random_seed : int, optional
        Seed for random number generation. Defaults to a random number between 0 and 2**32-1.

    Raises
    ------
    InvalidParameterError
        If provided parameter values are not in the allowed range.
    MissingParameterError
        If both or none of 'num_sequences', 'num_residues' is specified.

    Returns
    -------
    Job
    """
    endpoint = "v1/align/prompt"

    if not (0 <= homology_level <= 1):
        raise InvalidParameterError("The 'homology_level' must be between 0 and 1.")
    if not (0 <= max_similarity <= 1):
        raise InvalidParameterError("The 'max_similarity' must be between 0 and 1.")
    if not (0 <= min_similarity <= 1):
        raise InvalidParameterError("The 'min_similarity' must be between 0 and 1.")

    if num_residues is None and num_sequences is None:
        num_residues = 12288

    if (num_sequences is None and num_residues is None) or (
        num_sequences is not None and num_residues is not None
    ):
        raise MissingParameterError(
            "Either 'num_sequences' or 'num_residues' must be set, but not both."
        )

    if num_sequences is not None and not (0 <= num_sequences < 100):
        raise InvalidParameterError("The 'num_sequences' must be between 0 and 100.")

    if num_residues is not None and not (0 <= num_residues < 24577):
        raise InvalidParameterError("The 'num_residues' must be between 0 and 24577.")

    if random_seed is None:
        random_seed = random.randrange(2**32)

    params = {
        "msa_id": msa_id,
        "msa_method": method,
        "homology_level": homology_level,
        "max_similarity": max_similarity,
        "min_similarity": min_similarity,
        "force_include_first": always_include_seed_sequence,
        "replicates": num_ensemble_prompts,
        "seed": random_seed,
    }
    if num_sequences is not None:
        params["max_msa_sequences"] = num_sequences
    if num_residues is not None:
        params["max_msa_tokens"] = num_residues

    response = session.post(endpoint, params=params)
    return Job.model_validate(response.json())
