"""Schemas for OpenProtein align system."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from openprotein.jobs import Job, JobType


class AlignType(str, Enum):
    """
    Enumeration of alignment types.

    Attributes
    ----------
    INPUT : str
        Raw input alignment.
    MSA : str
        Generated multiple sequence alignment.
    PROMPT : str
        Prompt-based alignment.
    """

    INPUT = "RAW"
    MSA = "GENERATED"
    PROMPT = "PROMPT"


class MSASamplingMethod(str, Enum):
    """
    Enumeration of MSA sampling methods.

    Attributes
    ----------
    RANDOM : str
        Random sampling.
    NEIGHBORS : str
        Sampling based on neighbors.
    NEIGHBORS_NO_LIMIT : str
        Neighbor sampling without limit.
    NEIGHBORS_NONGAP_NORM_NO_LIMIT : str
        Neighbor sampling without gap normalization and without limit.
    TOP : str
        Top scoring sampling.
    """

    RANDOM = "RANDOM"
    NEIGHBORS = "NEIGHBORS"
    NEIGHBORS_NO_LIMIT = "NEIGHBORS_NO_LIMIT"
    NEIGHBORS_NONGAP_NORM_NO_LIMIT = "NEIGHBORS_NONGAP_NORM_NO_LIMIT"
    TOP = "TOP"


class PromptPostParams(BaseModel):
    """
    Parameters for posting a prompt to generate an MSA.

    Attributes
    ----------
    msa_id : str
        Identifier for the MSA.
    num_sequences : int or None, optional
        Number of sequences to sample (default is None, must be >=0 and <100).
    num_residues : int or None, optional
        Number of residues to sample (default is None, must be >=0 and <24577).
    method : MSASamplingMethod, optional
        Sampling method to use (default is NEIGHBORS_NONGAP_NORM_NO_LIMIT).
    homology_level : float, optional
        Homology level threshold (default is 0.8, must be between 0 and 1).
    max_similarity : float, optional
        Maximum similarity threshold (default is 1.0, must be between 0 and 1).
    min_similarity : float, optional
        Minimum similarity threshold (default is 0.0, must be between 0 and 1).
    always_include_seed_sequence : bool, optional
        Whether to always include the seed sequence (default is False).
    num_ensemble_prompts : int, optional
        Number of ensemble prompts to generate (default is 1).
    random_seed : int or None, optional
        Random seed for reproducibility (default is None).
    """

    msa_id: str
    num_sequences: int | None = Field(None, ge=0, lt=100)
    num_residues: int | None = Field(None, ge=0, lt=24577)
    method: MSASamplingMethod = MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    homology_level: float = Field(0.8, ge=0, le=1)
    max_similarity: float = Field(1.0, ge=0, le=1)
    min_similarity: float = Field(0.0, ge=0, le=1)
    always_include_seed_sequence: bool = False
    num_ensemble_prompts: int = 1
    random_seed: int | None = None


class MSAJob(Job):
    """
    Base class for MSA-related jobs.

    Attributes
    ----------
    job_type : Literal[JobType.align_align]
        The type of job (must be JobType.align_align).
    """

    job_type: Literal[JobType.align_align]

    @property
    def msa_id(self):
        """
        Returns the MSA identifier for this job.

        Returns
        -------
        str
            The MSA identifier.
        """
        return self.msa_id


class MafftJob(MSAJob, Job):
    """
    Job for running MAFFT alignment.

    Attributes
    ----------
    job_type : Literal[JobType.mafft]
        The type of job (must be JobType.mafft).
    """

    job_type: Literal[JobType.mafft]


class ClustalOJob(MSAJob, Job):
    """
    Job for running Clustal Omega alignment.

    Attributes
    ----------
    job_type : Literal[JobType.clustalo]
        The type of job (must be JobType.clustalo).
    """

    job_type: Literal[JobType.clustalo]


class AbNumberJob(MSAJob, Job):
    """
    Job for running AbNumber alignment.

    Attributes
    ----------
    job_type : Literal[JobType.abnumber]
        The type of job (must be JobType.abnumber).
    """

    job_type: Literal[JobType.abnumber]


class AbNumberScheme(str, Enum):
    """Antibody numbering scheme."""

    IMGT = "imgt"
    CHOTHIA = "chothia"
    KABAT = "kabat"
    AHO = "aho"
