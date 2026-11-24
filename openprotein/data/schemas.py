from datetime import datetime

from pydantic import BaseModel, field_validator

from openprotein.jobs.schemas import JobStatus

__all__ = ["AssayDataPage", "AssayDataRow", "AssayMetadata"]


class AssayMetadata(BaseModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: list[str]
    sequence_length: int | None = None
    has_name_column: bool = False
    metadata_names: list[str] | None = None
    status: JobStatus = JobStatus.SUCCESS
    align_job_id: str | None = None

    @field_validator("status", mode="before")
    @classmethod
    def _null_status_is_success(cls, value):
        return JobStatus.SUCCESS if value is None else value


class AssayDataRow(BaseModel):
    mut_sequence: str
    measurement_values: list[float | None]
    metadata_values: list[str | None] | None = None


class AssayDataPage(BaseModel):
    assaymetadata: AssayMetadata
    page_size: int
    page_offset: int
    assaydata: list[AssayDataRow]
