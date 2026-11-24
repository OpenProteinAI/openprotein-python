"""Align REST API interface for making HTTP calls to our align backend."""

import functools
import io
import random
from collections.abc import Callable, Iterator
from types import FunctionType
from typing import Any, BinaryIO

import requests

from openprotein import _deprecation, csv, fasta
from openprotein.base import APISession, _silence_server_warnings
from openprotein.data.msa import MSADownloadFormat
from openprotein.data.msa._api import msa_download
from openprotein.errors import InvalidParameterError, MissingParameterError
from openprotein.jobs import Job, JobType
from openprotein.jobs.api import job_get
from openprotein.prompt.schemas import PromptMetadata

from ._files import read_file_content, read_sequences
from .schemas import (
    AbNumberScheme,
    AlignType,
    MSASamplingMethod,
    Pairing,
    UnnumberedPolicy,
)

__all__ = [
    "abnumber_post",
    "antibody_schema_get",
    "clustalo_post",
    "get_align_job_inputs",
    "get_input",
    "get_msa",
    "get_seed",
    "mafft_post",
    "msa_post",
    "prompt_post",
]


_MSA_REPLACEMENT = (
    "`MSAFuture.wait()` for the generated MSA, or "
    "`session.data.MSA(id=...).to_string()`"
)
_PROMPT_REPLACEMENT = (
    "`Prompt.get()` for the replicates of a sampled prompt, indexed by replicate"
)
_SEED_REPLACEMENT = (
    "`openprotein.align.api.get_seed` for an MSA job's seed; the input of other "
    "align jobs has no read of its own, so keep your own copy of the submitted "
    "sequences"
)
_FUTURE_SEED_REPLACEMENT = "`MSAFuture.get_seed()` for the seed"


def _inputs_replacement(input_type: AlignType, *, on_future: bool = False) -> str:
    """The replacement to name when the read of ``input_type`` is deprecated.

    ``on_future`` selects the text for reads through an ``MSAFuture``.
    """
    if input_type == AlignType.MSA:
        return _MSA_REPLACEMENT
    if input_type == AlignType.PROMPT:
        return _PROMPT_REPLACEMENT
    return _FUTURE_SEED_REPLACEMENT if on_future else _SEED_REPLACEMENT


def _get_align_job_inputs(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> requests.Response:
    endpoint = "v1/align/inputs"

    params = {"job_id": job_id, "msa_type": input_type}
    if prompt_index is not None:
        params["replicate"] = prompt_index

    with _silence_server_warnings():
        return session.get(endpoint, params=params, stream=True)


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

    .. deprecated:: 0.18.0
        For ``AlignType.MSA``, use `MSAFuture.wait()` for the generated MSA, or
        `session.data.MSA(id=...).to_string()`. For ``AlignType.PROMPT``, use
        `Prompt.get()` for the replicates of a sampled prompt, indexed by
        replicate. For ``AlignType.INPUT``, use
        `openprotein.align.api.get_seed` for an MSA job's seed; the input of other
        align jobs has no read of its own, so keep your own copy of the submitted
        sequences.

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
    _deprecation.warn_deprecated(
        "openprotein.align.api.get_align_job_inputs",
        _inputs_replacement(input_type),
        since="0.18.0",
        removed_in="0.19.0",
    )
    return _get_align_job_inputs(session, job_id, input_type, prompt_index)


def _get_input(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> Iterator[tuple[str, str]]:
    response = _get_align_job_inputs(
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


def get_input(
    session: APISession,
    job_id: str,
    input_type: AlignType,
    prompt_index: int | None = None,
) -> Iterator[tuple[str, str]]:
    """
    Retrieve input data for a given alignment job.

    .. deprecated:: 0.18.0
        For ``AlignType.MSA``, use `MSAFuture.wait()` for the generated MSA, or
        `session.data.MSA(id=...).to_string()`. For ``AlignType.PROMPT``, use
        `Prompt.get()` for the replicates of a sampled prompt, indexed by
        replicate. For ``AlignType.INPUT``, use
        `openprotein.align.api.get_seed` for an MSA job's seed; the input of other
        align jobs has no read of its own, so keep your own copy of the submitted
        sequences.

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
    _deprecation.warn_deprecated(
        "openprotein.align.api.get_input",
        _inputs_replacement(input_type),
        since="0.18.0",
        removed_in="0.19.0",
    )
    return _get_input(session, job_id, input_type, prompt_index)


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
    endpoint = "v1/align/seed"
    response = session.get(endpoint, params={"msa_id": job_id})
    return response.json()["sequence"]


_FASTA_MSA_JOB_TYPES = {
    JobType.mafft.value,
    JobType.clustalo.value,
    JobType.abnumber.value,
}


def _msa_format(job_type: str) -> MSADownloadFormat:
    """The format an MSA is read in: FASTA for alignments, a3m for searches."""
    return "fasta" if job_type in _FASTA_MSA_JOB_TYPES else "a3m"


def _read_msa(
    session: APISession, msa_id: str, format: MSADownloadFormat
) -> Iterator[tuple[str, str]]:
    """Read a stored MSA in `format`, which must be "a3m" or "fasta"."""
    content = msa_download(session, msa_id, format=format)
    return fasta.parse_stream(iter(content.decode().splitlines()))


def get_msa(session: APISession, job_id: str) -> Iterator[tuple[str, str]]:
    """
    Retrieve the generated MSA (Multiple Sequence Alignment) for a given job.

    The MSA is requested as a3m for a search or upload job and as FASTA for a
    MAFFT, Clustal Omega or AbNumber job, whatever format it is stored in.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier, which is also the MSA identifier.

    Returns
    -------
    Iterator[tuple[str, str]]
        An iterator over the name, sequence of the MSA.

    Raises
    ------
    HTTPError
        If the job is not found or its MSA is not stored.
    """
    job = job_get(session, job_id)
    return _read_msa(session, job_id, _msa_format(job.job_type))


def _msa_post(session: APISession, msa_file: BinaryIO) -> Job:
    endpoint = "v1/align/msa"
    with _silence_server_warnings():
        response = session.post(
            endpoint, files={"msa_file": msa_file}, params={"is_seed": False}
        )
    return Job.model_validate(response.json())


def msa_post(
    session: APISession,
    msa_file: BinaryIO | None = None,
    seed: str | bytes | None = None,
) -> Job:
    """
    Create an MSA job from a ready-made MSA or from a seed sequence.

    Exactly one of `msa_file` and `seed` must be provided.

    .. deprecated:: 0.18.0
        Use `session.data.MSA.create` for a ready-made MSA, or
        `session.align.create_msa` for a seed.

    Parameters
    ----------
    session : APISession
        The API session.
    msa_file : BinaryIO or None, optional
        Ready-made MSA as FASTA, a3m or a headerless CSV of names and sequences.
        Defaults to None.
    seed : str or bytes or None, optional
        Seed sequence to build an MSA from. Defaults to None.

    Raises
    ------
    MissingParameterError
        If neither or both of `msa_file` and `seed` are provided.

    Returns
    -------
    Job
        Job details.
    """
    _deprecation.warn_deprecated(
        "openprotein.align.api.msa_post",
        "`session.data.MSA.create` for a ready-made MSA, "
        "or `session.align.create_msa` for a seed",
        since="0.18.0",
        removed_in="0.19.0",
    )
    if (msa_file is None) == (seed is None):
        raise MissingParameterError("seed OR msa_file must be provided.")
    if msa_file is not None:
        return _msa_post(session, msa_file)

    assert seed is not None
    seed = seed.encode() if isinstance(seed, str) else seed
    with _silence_server_warnings():
        response = session.post(
            "v1/align/msa",
            files={"msa_file": io.BytesIO(b"\n".join([b">seed", seed]))},
            params={"is_seed": True},
        )
    return Job.model_validate(response.json())


def _msa_mmseqs_search_post(
    session: APISession, seed: str | bytes, pairing: Pairing = Pairing.BOTH
) -> Job:
    """
    Launch a ColabFold MMseqs2 homology search to build an MSA for a query sequence.

    Posts the query to the dedicated search endpoint. The query may be a single
    sequence, or `:`-separated for a complex.

    Parameters
    ----------
    session : APISession
        The API session.
    seed : str or bytes
        Query (seed) sequence to search.
    pairing : Pairing, optional
        Cross-chain pairing strategy for a complex query (default `both`); has no
        effect on a single-chain query.

    Returns
    -------
    Job
        Job details for the submitted search.
    """
    endpoint = "v1/align/search/mmseqs2/msa"

    query = seed.decode() if isinstance(seed, bytes) else seed
    data = {"query": query, "pairing": pairing.value}

    response = session.post(endpoint, data=data)
    return Job.model_validate(response.json())


def _is_sequence_file(value: Any) -> bool:
    """Whether ``value`` is a file, its content, or a ``(name, file)`` tuple."""
    if isinstance(value, tuple):
        content = value[1] if len(value) > 1 else None
        return hasattr(content, "read") or isinstance(content, (bytes, bytearray))
    return hasattr(value, "read") or isinstance(value, (bytes, bytearray, str))


def _accepts_sequence_file(
    *file_options: str,
) -> Callable[[FunctionType], Callable[..., Job]]:
    """Let an align POST helper also take a deprecated ``sequence_file``.

    The file, given first or as ``sequence_file=``, is read into sequences and
    names. ``file_options`` names the options that may follow it positionally.
    """

    def decorate(post: FunctionType) -> Callable[..., Job]:
        name = post.__name__

        @functools.wraps(post)
        def wrapper(
            session: APISession, *args: Any, sequence_file: Any = None, **kwargs: Any
        ) -> Job:
            if sequence_file is None and args and _is_sequence_file(args[0]):
                sequence_file, args = args[0], args[1:]
            if sequence_file is None:
                return post(session, *args, **kwargs)
            _deprecation.warn_deprecated(
                f"openprotein.align.api.{name}(sequence_file=...)",
                f"`openprotein.align.api.{name}` with sequences and names",
                since="0.18.0",
                removed_in="0.19.0",
            )
            if len(args) > len(file_options):
                raise TypeError(
                    f"{name}() takes at most {len(file_options)} options after "
                    "the sequence file"
                )
            for option, value in zip(file_options, args):
                if option in kwargs:
                    raise TypeError(f"{name}() got multiple values for {option!r}")
                kwargs[option] = value
            sequences, names = read_sequences(read_file_content(sequence_file))
            return post(session, sequences, names, **kwargs)

        return wrapper

    return decorate


def _body(sequences: list[str], names: list[str] | None) -> dict:
    """The request body the align endpoints take."""
    body: dict = {"sequences": sequences}
    if names is not None:
        if len(names) != len(sequences):
            raise InvalidParameterError(
                "names and sequences must be the same length, "
                f"not {len(names)} and {len(sequences)}"
            )
        body["names"] = names
    return body


@_accepts_sequence_file("auto", "ep", "op")
def mafft_post(
    session: APISession,
    sequences: list[str],
    names: list[str] | None = None,
    auto: bool = True,
    ep: float | None = None,
    op: float | None = None,
) -> Job:
    """
    Align sequences using the MAFFT algorithm.

    Set `auto` to True to automatically attempt the best parameters. Leave a parameter as None to use system defaults.

    .. deprecated:: 0.18.0
        ``sequence_file``, a FASTA or headerless CSV file in place of
        ``sequences`` and ``names``, is removed in 0.19.0. Pass ``sequences``
        and ``names`` instead.

    Parameters
    ----------
    session : APISession
        The API session.
    sequences : list[str]
        Unaligned sequences, in the order they should be aligned. Use `:` to
        separate the chains of a multi-chain sequence.
    names : list[str] or None, optional
        Row labels for the aligned output, same length as `sequences`. Omit them
        and the aligner labels the rows itself.
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

    params: dict = {"auto": auto}
    if ep is not None:
        params["ep"] = ep
    if op is not None:
        params["op"] = op

    response = session.post(endpoint, json=_body(sequences, names), params=params)
    return Job.model_validate(response.json())


@_accepts_sequence_file("clustersize", "iterations")
def clustalo_post(
    session: APISession,
    sequences: list[str],
    names: list[str] | None = None,
    clustersize: int | None = None,
    iterations: int | None = None,
) -> Job:
    """
    Align sequences using the Clustal Omega algorithm.

    Leave a parameter as None to use system defaults.

    .. deprecated:: 0.18.0
        ``sequence_file``, a FASTA or headerless CSV file in place of
        ``sequences`` and ``names``, is removed in 0.19.0. Pass ``sequences``
        and ``names`` instead.

    Parameters
    ----------
    session : APISession
        The API session.
    sequences : list[str]
        Unaligned sequences, in the order they should be aligned. Use `:` to
        separate the chains of a multi-chain sequence.
    names : list[str] or None, optional
        Row labels for the aligned output, same length as `sequences`. Omit them
        and the aligner labels the rows itself.
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

    params = {}
    if clustersize is not None:
        params["clustersize"] = clustersize
    if iterations is not None:
        params["iterations"] = iterations

    response = session.post(endpoint, json=_body(sequences, names), params=params)
    return Job.model_validate(response.json())


@_accepts_sequence_file("scheme", "drop_minority_chains", "unnumbered_policy")
def abnumber_post(
    session: APISession,
    sequences: list[str],
    names: list[str] | None = None,
    scheme: AbNumberScheme | str = AbNumberScheme.IMGT,
    drop_minority_chains: bool = False,
    unnumbered_policy: UnnumberedPolicy | str = UnnumberedPolicy.ERROR,
) -> Job:
    """
    Align antibody sequences using AbNumber.

    The antibody numbering scheme can be specified.

    .. deprecated:: 0.18.0
        ``sequence_file``, a FASTA or headerless CSV file in place of
        ``sequences`` and ``names``, is removed in 0.19.0. Pass ``sequences``
        and ``names`` instead.

    Parameters
    ----------
    session : APISession
        The API session.
    sequences : list[str]
        Unaligned sequences, in the order they should be aligned. Use `:` to
        separate the chains of a multi-chain sequence.
    names : list[str] or None, optional
        Row labels for the aligned output, same length as `sequences`. Omit them
        and the aligner labels the rows itself.
    scheme : AbNumberScheme, optional
        Antibody numbering scheme. Default is IMGT.
    drop_minority_chains : bool, optional
        If True, drop sequences belonging to chain types that are in the
        minority (e.g. heavy vs light) so the resulting alignment contains
        only the dominant chain type. Default is False.
    unnumbered_policy : UnnumberedPolicy, optional
        What to do with residues outside the numbered variable domain.
        Default is ERROR, which fails the job.

    Returns
    -------
    Job
        Job details.
    """
    endpoint = "v1/align/abnumber"

    if isinstance(scheme, str):
        if scheme not in {value.value for value in AbNumberScheme}:
            raise InvalidParameterError(f"Antibody numbering {scheme} not recognized")
    if isinstance(unnumbered_policy, str):
        if unnumbered_policy not in {value.value for value in UnnumberedPolicy}:
            raise InvalidParameterError(
                f"Unnumbered residue policy {unnumbered_policy} not recognized"
            )

    params = {
        "scheme": scheme if isinstance(scheme, str) else scheme.value,
        "drop_minority_chains": drop_minority_chains,
        "unnumbered_policy": (
            unnumbered_policy
            if isinstance(unnumbered_policy, str)
            else unnumbered_policy.value
        ),
    }

    response = session.post(endpoint, json=_body(sequences, names), params=params)
    return Job.model_validate(response.json())


def antibody_schema_get(session: APISession, job_id: str) -> dict:
    """
    Retrieve the antibody numbering for an AbNumber job.

    Parameters
    ----------
    session : APISession
        The API session.
    job_id : str
        The job identifier.

    Returns
    -------
    dict
        The job's extras: a ``chains`` entry per chain position carrying the
        numbering and CDR regions, and a ``warnings`` list.
    """
    endpoint = "v1/align/antibody_schema"

    response = session.get(endpoint, params={"job_id": job_id})
    return response.json()


def _sample_params(
    msa_id: str,
    num_sequences: int | None,
    num_residues: int | None,
    method: MSASamplingMethod,
    homology_level: float,
    max_similarity: float,
    min_similarity: float,
    always_include_seed_sequence: bool,
    num_ensemble_prompts: int,
    random_seed: int | None,
) -> dict:
    """The query params of a prompt sampling request, checked."""
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
    return params


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

    .. deprecated:: 0.18.0
        Use `MSAFuture.sample_prompt` or `MSA.sample_prompt`.

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
    _deprecation.warn_deprecated(
        "openprotein.align.api.prompt_post",
        "`MSAFuture.sample_prompt` or `MSA.sample_prompt`",
        since="0.18.0",
        removed_in="0.19.0",
    )
    params = _sample_params(
        msa_id,
        num_sequences,
        num_residues,
        method,
        homology_level,
        max_similarity,
        min_similarity,
        always_include_seed_sequence,
        num_ensemble_prompts,
        random_seed,
    )
    with _silence_server_warnings():
        response = session.post("v1/align/prompt", params=params)
    return Job.model_validate(response.json())


def _sample_prompt(
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
) -> PromptMetadata:
    """Sample a prompt from an MSA; returns its metadata in PENDING status.

    Takes the parameters of `prompt_post`.
    """
    params = _sample_params(
        msa_id,
        num_sequences,
        num_residues,
        method,
        homology_level,
        max_similarity,
        min_similarity,
        always_include_seed_sequence,
        num_ensemble_prompts,
        random_seed,
    )
    response = session.post("v1/prompt/sample_msa", params=params)
    return PromptMetadata.model_validate(response.json())
