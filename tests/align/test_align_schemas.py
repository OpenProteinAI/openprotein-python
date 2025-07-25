import pytest
from pydantic import ValidationError

from openprotein.jobs import JobStatus, JobType
from openprotein.align.schemas import (
    PromptPostParams,
    MSASamplingMethod,
    MSAJob,
    MafftJob,
    ClustalOJob,
    AbNumberJob,
)

# ===================================
# PromptPostParams Schema Tests
# ===================================


def test_prompt_post_params_valid_defaults():
    """Test PromptPostParams with only required fields, relying on defaults."""
    params = PromptPostParams(
        msa_id="msa123",
        num_sequences=10,
        num_residues=100,
        homology_level=0.5,
        max_similarity=0.9,
        min_similarity=0.1,
    )
    assert params.msa_id == "msa123"
    assert params.method == MSASamplingMethod.NEIGHBORS_NONGAP_NORM_NO_LIMIT
    assert params.homology_level == 0.5
    assert params.num_sequences == 10
    assert params.num_residues == 100


def test_prompt_post_params_valid_all_fields():
    """Test PromptPostParams with all fields provided."""
    data = {
        "msa_id": "msa123",
        "num_sequences": 50,
        "method": MSASamplingMethod.RANDOM,
        "homology_level": 0.5,
        "max_similarity": 0.9,
        "min_similarity": 0.1,
        "always_include_seed_sequence": True,
        "num_ensemble_prompts": 2,
        "random_seed": 42,
    }
    params = PromptPostParams(num_residues=100, **data)
    for key, value in data.items():
        assert getattr(params, key) == value


def test_prompt_post_params_invalid_range():
    """Test PromptPostParams with out-of-range values."""
    with pytest.raises(ValidationError):
        PromptPostParams(
            msa_id="msa123",
            num_sequences=10,
            num_residues=100,
            homology_level=1.1,
            max_similarity=0.9,
            min_similarity=0.1,
        )

    with pytest.raises(ValidationError):
        PromptPostParams(
            msa_id="msa123",
            num_sequences=101,
            num_residues=100,
            homology_level=0.5,
            max_similarity=0.9,
            min_similarity=0.1,
        )

    with pytest.raises(ValidationError):
        PromptPostParams(
            msa_id="msa123",
            num_sequences=10,
            num_residues=99999,
            homology_level=0.5,
            max_similarity=0.9,
            min_similarity=0.1,
        )


# ===================================
# Job Schemas Tests
# ===================================


def create_base_job_dict(job_id: str, status: JobStatus) -> dict:
    """Helper to create a base dictionary for a Job."""
    return {
        "job_id": job_id,
        "status": status,
        "job_type": "align_align",  # Placeholder, will be overwritten by specific types
        "created_date": "2023-01-01T00:00:00",
        "last_update": "2023-01-01T00:00:00",
    }


def test_msa_job_schema():
    """Test the MSAJob schema."""
    job_dict = create_base_job_dict("job1", JobStatus.SUCCESS)
    job_dict["job_type"] = JobType.align_align

    msa_job = MSAJob.model_validate(job_dict)
    assert msa_job.job_id == "job1"
    assert msa_job.job_type == JobType.align_align


def test_mafft_job_schema():
    """Test the MafftJob schema."""
    job_dict = create_base_job_dict("job2", JobStatus.RUNNING)
    job_dict["job_type"] = JobType.mafft

    mafft_job = MafftJob.model_validate(job_dict)
    assert mafft_job.job_id == "job2"
    assert mafft_job.job_type == JobType.mafft


def test_clustalo_job_schema():
    """Test the ClustalOJob schema."""
    job_dict = create_base_job_dict("job3", JobStatus.FAILURE)
    job_dict["job_type"] = JobType.clustalo

    clustalo_job = ClustalOJob.model_validate(job_dict)
    assert clustalo_job.job_id == "job3"
    assert clustalo_job.job_type == JobType.clustalo


def test_abnumber_job_schema():
    """Test the AbNumberJob schema."""
    job_dict = create_base_job_dict("job4", JobStatus.PENDING)
    job_dict["job_type"] = JobType.abnumber

    abnumber_job = AbNumberJob.model_validate(job_dict)
    assert abnumber_job.job_id == "job4"
    assert abnumber_job.job_type == JobType.abnumber


def test_job_schema_invalid_type():
    """Test that validation fails for an incorrect job_type."""
    job_dict = create_base_job_dict("job1", JobStatus.SUCCESS)
    job_dict["job_type"] = "not_a_real_type"  # Invalid type

    with pytest.raises(ValidationError):
        MafftJob.model_validate(job_dict)
