from openprotein.base import APISession
import openprotein.config as config

import pandas as pd

import pydantic
from datetime import datetime
from typing import List, Optional, Union
from io import BytesIO


class AssayMetadata(pydantic.BaseModel):
    assay_name: str
    assay_description: str
    assay_id: str
    original_filename: str
    created_date: datetime
    num_rows: int
    num_entries: int
    measurement_names: List[str]


def assaydata_post(session: APISession, assay_file, assay_name: str, assay_description: Optional[str] = ''):
    endpoint = 'v1/assaydata'

    files = {'assay_data': assay_file}
    data = {'assay_name': assay_name, 'assay_description': assay_description}

    response = session.post(endpoint, files=files, data=data)
    return pydantic.parse_obj_as(AssayMetadata, response.json())


def assaydata_list(session: APISession):
    endpoint = 'v1/assaydata'
    response = session.get(endpoint)
    return pydantic.parse_obj_as(List[AssayMetadata], response.json())


def assaydata_put(session: APISession, assay_id: str, assay_name: Optional[str] = None, assay_description: Optional[str] = None):
    endpoint = f'v1/assaydata/{assay_id}'
    data = {}
    if assay_name is not None:
        data['assay_name'] = assay_name
    if assay_description is not None:
        data['assay_description'] = assay_description
    
    response = session.put(endpoint, data=data)
    return pydantic.parse_obj_as(AssayMetadata, response.json())


class AssayDataRow(pydantic.BaseModel):
    mut_sequence: str
    measurement_values: List[Union[float, None]]


class AssayDataPage(pydantic.BaseModel):
    assaymetadata: AssayMetadata
    page_size: int
    page_offset: int
    assaydata: List[AssayDataRow]


def assaydata_page_get(
        session: APISession,
        assay_id: str,
        measurement_name: Optional[str] = None,
        page_offset: int = 0,
        page_size: int = 1000,
        format: str = 'wide',
    ):
    endpoint = f'v1/assaydata/{assay_id}'

    params = {'page_offset': page_offset, 'page_size': page_size, 'format': format}
    if measurement_name is not None:
        params['measurement_name'] = measurement_name
    
    response = session.get(endpoint, params=params)
    return pydantic.parse_obj_as(AssayDataPage, response.json())


class AssayDataset:
    def __init__(self, session: APISession, metadata: AssayMetadata):
        self.session = session
        self.metadata = metadata

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

    @property
    def id(self):
        return self.metadata.assay_id

    @property
    def name(self):
        return self.metadata.assay_name

    @property
    def description(self):
        return self.metadata.assay_description

    @property
    def measurement_names(self):
        return self.metadata.measurement_names

    def __len__(self):
        return self.metadata.num_rows

    @property
    def shape(self):
        return (len(self), len(self.measurement_names) + 1)

    def update(self, assay_name=None, assay_description=None):
        metadata = assaydata_put(self.session, self.id, assay_name=assay_name, assay_description=assay_description)
        self.metadata = metadata

    def get_all(self):
        rows = []
        for i in range(0, len(self), config.BASE_PAGE_SIZE):
            entries = assaydata_page_get(self.session, self.id, page_offset=i, page_size=config.BASE_PAGE_SIZE)
            for row in entries.assaydata:
                row = [row.mut_sequence] + row.measurement_values
                rows.append(row)
        table = pd.DataFrame(rows, columns=['sequence'] + self.measurement_names)
        return table

    def get_slice(self, start, end):
        rows = []
        for i in range(start, end, config.BASE_PAGE_SIZE):
            entries = assaydata_page_get(self.session, self.id, page_offset=i, page_size=config.BASE_PAGE_SIZE)
            for row in entries.assaydata:
                row = [row.mut_sequence] + row.measurement_values
                rows.append(row)
        table = pd.DataFrame(rows, columns=['sequence'] + self.measurement_names)
        return table


class DataAPI:
    def __init__(self, session: APISession):
        self.session = session

    def list(self) -> List[AssayDataset]:
        metadata = assaydata_list(self.session)
        return [AssayDataset(self.session, x) for x in metadata]

    def create(self, table: pd.DataFrame, name: str, description=None) -> AssayDataset:
        stream = BytesIO()
        table.to_csv(stream, index=False)
        stream.seek(0)
        metadata = assaydata_post(self.session, stream, name, assay_description=description)
        return AssayDataset(self.session, metadata)
    
    def get(self, assay_id) -> AssayDataset:
        datasets = self.list()
        for dataset in datasets:
            if dataset.id == assay_id:
                return dataset
        raise KeyError(f'No assay with id={assay_id} found.')

    def __len__(self):
        return len(self.list())