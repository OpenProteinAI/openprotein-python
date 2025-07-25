"""Design REST API for making HTTP calls to our design backend."""

from typing import Iterator

import numpy as np
from pydantic import TypeAdapter

from openprotein import csv
from openprotein.base import APISession

from .schemas import (
    Criteria,
    Criterion,
    Design,
    DesignConstraint,
    DesignJob,
    DesignResult,
    Job,
    Subcriterion,
)

PATH_PREFIX = "v1/designer/design"


def designs_list(session: APISession) -> list[Design]:
    """
    List designs.

    Parameters
    ----------
    session : APISession
        Session object for API communication.

    Returns
    -------
    list[Design]
        List of designs.
    """
    endpoint = PATH_PREFIX
    response = session.get(endpoint)
    return TypeAdapter(list[Design]).validate_python(response.json())


def design_get(session: APISession, design_id: str) -> Design:
    """
    Get design.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    design_id: str
        ID of design to get.

    Returns
    -------
    Design
        Design metadata.
    """
    endpoint = PATH_PREFIX + f"/{design_id}"
    response = session.get(endpoint)
    return TypeAdapter(Design).validate_python(response.json())


def designer_create_genetic_algorithm(
    session: APISession,
    assay_id: str,
    criteria: Criteria | Subcriterion | Criterion,
    num_steps: int = 25,
    pop_size: int = 1024,  # TODO - rename to library_size
    n_offsprings: int = 5120,
    crossover_prob: float = 1.0,
    crossover_prob_pointwise: float = 0.2,
    mutation_average_mutations_per_seq: int = 1,
    allowed_tokens: DesignConstraint | dict[int, list[str]] = {},
) -> Job:
    """
    Create design using genetic algorithm.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_id : str
        Assay ID to fit GP on.
    criteria: list[list[DesignCriterion]]
        List of list of design criteria, logically grouping by OR then AND.
    num_steps: int, optional
        The number of steps in the genetic algorithm. Default is 8.
    pop_size: int, optional
        The population size for the genetic algorithm. Default is 256.
    n_offsprings: int, optional
        The number of offspring for the genetic algorithm. Default is 5120.
    crossover_prob: float, optional
        The crossover probability for the genetic algorithm. Default is 1.
    crossover_prob_pointwise: float, optional
        The pointwise crossover probability for the genetic algorithm. Default is 0.2.
    mutation_average_mutations_per_seq: int, optional
        The average number of mutations per sequence. Default is 1.
    allowed_tokens: DesignConstraint | dict[int, list[str]]
        A dict of positions and allows tokens (e.g. *{1:['G','L']})* ) designating how mutations may occur. Defaults to empty dict.

    Returns
    -------
    DesignJob
    """
    if isinstance(criteria, Subcriterion):
        criteria = Criteria([Criterion([criteria])])
    elif isinstance(criteria, Criterion):
        criteria = Criteria([criteria])

    if isinstance(allowed_tokens, DesignConstraint):
        allowed_tokens = allowed_tokens.as_dict()

    endpoint = PATH_PREFIX + "/genetic-algorithm"

    body = {
        "assay_id": assay_id,
        "criteria": criteria.model_dump(),
        "num_steps": num_steps,
        "pop_size": pop_size,
        "n_offsprings": n_offsprings,
        "crossover_prob": crossover_prob,
        "crossover_prob_pointwise": crossover_prob_pointwise,
        "mutation_average_mutations_per_seq": mutation_average_mutations_per_seq,
        "allowed_tokens": allowed_tokens,
    }
    response = session.post(endpoint, json=body)
    return DesignJob.model_validate(response.json())


def design_delete(session: APISession, design_id: str):
    raise NotImplementedError()


def designer_get_design_results(
    session: APISession,
    design_id: str,
    step: int | None = None,
) -> Iterator[list[str]]:
    """
    Get csv encoded results for a design ID.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    design_id : str
        Design ID to retrieve results from.
    step: int | None, optional
        Step of the design whose results to fetch. Defaults to -1, which refers to the last step.

    Returns
    -------
    bytes
    """
    params = {}
    if step is not None:
        if step != -1:
            step -= 1
        params["step"] = step
    endpoint = PATH_PREFIX + f"/{design_id}/results"
    response = session.get(endpoint, params=params, stream=True)
    return csv.parse_stream(response.iter_lines())


def decode_design_result(
    row: list[str],
    score_start_index: int,
    subscore_start_index: int,
    pred_start_index: int,
) -> DesignResult:
    """
    Decode prediction scores.

    Args:
        data (bytes): raw bytes encoding the array received over the API
        batched (bool): whether or not the result was batched. affects the retrieved csv format whether they contain additional columns and header rows.

    Returns:
        mus (np.ndarray): decoded array of means
        vars (np.ndarray): decoded array of variances
    """
    scores = np.array(
        [float(score) for score in row[score_start_index:subscore_start_index]]
    )
    subscores = np.array(
        [float(subscore) for subscore in row[subscore_start_index:pred_start_index]]
    )
    preds = np.array([float(pred) for pred in row[pred_start_index:]])
    result = DesignResult(
        step=int(row[0]) + 1,
        sample_index=int(row[1]) + 1,
        sequence=row[2],
        scores=scores,
        subscores=subscores,
        means=preds[::2],
        vars=preds[1::2],
    )
    return result


def decode_design_results_stream(
    data: Iterator[list[str]], header: list[str] | None = None
) -> Iterator[DesignResult]:
    """
    Decode design results.

    Args:
        data: Iterator[list[str]]
            Data in the form of an iterator of list of string-encoded values
        header: list[str] | None, optional
            Headers describing the data. Should be same length as each row returned from the data iterator.
            Defaults to None, which means the first row in the iterator should be header.

    Returns:
        step: int
            Step index of the design.
        sample_index: int
            Index of the sample in the overall design.
        sequence: str
            Output designed sequence.
        scores: np.ndarray[float]
            M array of scores based on provided criteria (M groups of subcriteria).
        subscores: np.ndarray[float]
            N array of subscores based on provided criteria (flattened N subcriteria).
        means: np.ndarray[float]
            K array of means for each model subscriterion.
        vars: np.ndarray[float]
            K array of variances for each model subscriterion.
        vars (np.ndarray): decoded array of variances
    """
    if header is None:
        header = next(data)
        if header[0].isnumeric():
            raise ValueError(
                "Expected first row in data to be header of 'step','sample_index',..."
            )
    score_start_index = subscore_start_index = pred_start_index = len(header)
    # first start indices
    for i, col_name in enumerate(header):
        if col_name.startswith("score"):
            score_start_index = i
            break
    for i, col_name in enumerate(header[score_start_index:]):
        if col_name.endswith("score"):
            subscore_start_index = score_start_index + i
            break
    for i, col_name in enumerate(header[subscore_start_index:]):
        if col_name.endswith("y_mu"):
            pred_start_index = subscore_start_index + i
            break
    for row in data:
        yield decode_design_result(
            row=row,
            score_start_index=score_start_index,
            subscore_start_index=subscore_start_index,
            pred_start_index=pred_start_index,
        )
