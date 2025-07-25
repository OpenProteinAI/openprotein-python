from datetime import datetime

from pydantic import BaseModel


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


class AssayDataRow(BaseModel):
    mut_sequence: str
    measurement_values: list[float | None]


class AssayDataPage(BaseModel):
    assaymetadata: AssayMetadata
    page_size: int
    page_offset: int
    assaydata: list[AssayDataRow]
