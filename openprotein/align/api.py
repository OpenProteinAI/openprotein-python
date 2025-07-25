"""Align REST API interface for making HTTP calls to our align backend."""

import io
import random
from typing import BinaryIO, Iterator

import requests

from openprotein import csv, fasta
from openprotein.base import APISession
from openprotein.errors import APIError, InvalidParameterError, MissingParameterError
from openprotein.jobs import Job

from .schemas import AbNumberScheme, AlignType, MSASamplingMethod


def get_align_job_inputs(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> requests.Response:
    """
    Retrieve MSA and related data for an alignment job.

    Depending on `input_type`, returns either the original user seed (RAW), the generated MSA, or the prompt.
    If `input_type` is PROMPT, specify `prompt_index` to retrieve the specific prompt for each replicate.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.
    input_type : AlignType
        The type of MSA data to retrieve.
    prompt_index : int or None, optional
        The replicate number for the prompt (only used if `input_type` is PROMPT).

    Returns
    -------
    requests.Response
        The response object from the server.
    """
    endpoint = "v1/align/inputs"

    params = {"job_id": job_id, "msa_type": input_type}
    if prompt_index is not None:
        params["replicate"] = prompt_index

    response = session.get(endpoint, params=params, stream=True)
    return response


def get_input(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> Iterator[tuple[str, str]]:
    """
    Retrieve input data for a given alignment job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.
    input_type : AlignType
        The type of MSA data to retrieve.
    prompt_index : int or None, optional
        The replicate number for the prompt (only used if `input_type` is PROMPT).

    Returns
    -------
    Iterator[tuple[str, str]]
        An iterator over the name, sequence of the response.
    """
    response = get_align_job_inputs(
        session=session, job_id=job_id, input_type=input_type, prompt_index=prompt_index
    )
    if response.headers.get("Content-Type") == "text/x-fasta":
        return fasta.parse_stream(response.iter_lines(decode_unicode=True))
    else:
        # take first two columns only
        return (
            (row[0], row[1])
            for row in csv.parse_stream(response.iter_lines(decode_unicode=True))
        )


def get_seed(session: APISession, job_id: str) -> str:
    """
    Retrieve the seed sequence for a given MSA job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.

    Returns
    -------
    str
        The seed sequence.
    """
    # HACK for some reason this returns a csv
    r = get_input(session=session, job_id=job_id, input_type=AlignType.INPUT)
    seed = next(r)[1]
    return seed


def get_msa(session: APISession, job_id: str) -> Iterator[tuple[str, str]]:
    """
    Retrieve the generated MSA (Multiple Sequence Alignment) for a given job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.

    Returns
    -------
    Iterator[tuple[str, str]]
        An iterator over the name, sequence of the MSA.
    """
    return get_input(session=session, job_id=job_id, input_type=AlignType.MSA)


def msa_post(
    session: APISession,
    msa_file: BinaryIO | None = None,
    seed: str | bytes | None = None,
) -> Job:
    """
    Create an MSA job.

    Either a seed sequence (which will trigger MSA creation) or a ready-to-use MSA (via `msa_file`) must be provided.
    `seed` and `msa_file` are mutually exclusive.

    Parameters
    ----------
    session : APISession
        The API session.
    msa_file : BinaryIO or None, optional
        Ready-made MSA file. Defaults to None.
    seed : str or bytes or None, optional
        Seed sequence to trigger MSA job. Defaults to None.

    Raises
    ------
    MissingParameterError
        If neither or both of `msa_file` and `seed` are provided.

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


def mafft_post(
    session: APISession,
    sequence_file: BinaryIO,
    auto: bool = True,
    ep: float | None = None,
    op: float | None = None,
) -> Job:
    """
    Align sequences using the MAFFT algorithm.

    Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.
    Set `auto` to True to automatically attempt the best parameters. Leave a parameter as None to use system defaults.

    Parameters
    ----------
    session : APISession
        The API session.
    sequence_file : BinaryIO
        Sequences to align in FASTA or CSV format.
    auto : bool, optional
        Set to True to automatically set algorithm parameters. Default is True.
    ep : float or None, optional
        MAFFT parameter. Default is None.
    op : float or None, optional
        MAFFT parameter. Default is None.

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


def clustalo_post(
    session: APISession,
    sequence_file: BinaryIO,
    clustersize: int | None = None,
    iterations: int | None = None,
) -> Job:
    """
    Align sequences using the Clustal Omega algorithm.

    Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.
    Leave a parameter as None to use system defaults.

    Parameters
    ----------
    session : APISession
        The API session.
    sequence_file : BinaryIO
        Sequences to align in FASTA or CSV format.
    clustersize : int or None, optional
        Clustal Omega parameter. Default is None.
    iterations : int or None, optional
        Clustal Omega parameter. Default is None.

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
    scheme: AbNumberScheme | str = AbNumberScheme.IMGT,
) -> Job:
    """
    Align antibody sequences using AbNumber.

    Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.
    The antibody numbering scheme can be specified.

    Parameters
    ----------
    session : APISession
        The API session.
    sequence_file : BinaryIO
        Sequences to align in FASTA or CSV format.
    scheme : AbNumberScheme, optional
        Antibody numbering scheme. Default is IMGT.

    Returns
    -------
    Job
        Job details.
    """
    endpoint = "v1/align/abnumber"

    if isinstance(scheme, str):
        if scheme not in {value.value for value in AbNumberScheme}:
            raise InvalidParameterError(f"Antibody numbering {scheme} not recognized")

    files = {"file": sequence_file}
    params = {"scheme": scheme if isinstance(scheme, str) else scheme.value}

    response = session.post(endpoint, files=files, params=params)
    return Job.model_validate(response.json())


def antibody_schema_get(session: APISession, job_id: str):
    """
    Retrieve the antibody numbering for an AbNumber job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Returns
    -------
    None
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
    Create a protein sequence prompt from a linked MSA (Multiple Sequence Alignment) for PoET jobs.

    The MSA is specified by `msa_id` and created in `msa_post`.

    Parameters
    ----------
    session : APISession
        The API session.
    msa_id : str
        The ID of the Multiple Sequence Alignment to use for the prompt.
    num_sequences : int or None, optional
        Maximum number of sequences in the prompt. Must be less than 100.
    num_residues : int or None, optional
        Maximum number of residues (tokens) in the prompt. Must be less than 24577.
    method : MSASamplingMethod, optional
        Method to use for MSA sampling. Default is NEIGHBORS_NONGAP_NORM_NO_LIMIT.
    homology_level : float, optional
        Level of homology for sequences in the MSA (neighbors methods only). Must be between 0 and 1. Default is 0.8.
    max_similarity : float, optional
        Maximum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Default is 1.0.
    min_similarity : float, optional
        Minimum similarity between sequences in the MSA and the seed. Must be between 0 and 1. Default is 0.0.
    always_include_seed_sequence : bool, optional
        Whether to always include the seed sequence in the MSA. Default is False.
    num_ensemble_prompts : int, optional
        Number of ensemble jobs to run. Default is 1.
    random_seed : int or None, optional
        Seed for random number generation. Default is a random number between 0 and 2**32-1.

    Raises
    ------
    InvalidParameterError
        If provided parameter values are not in the allowed range.
    MissingParameterError
        If both or neither of `num_sequences` and `num_residues` are specified.

    Returns
    -------
    Job
        Job details.
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
