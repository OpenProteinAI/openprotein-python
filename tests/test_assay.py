import pytest
from unittest.mock import MagicMock
from openprotein.base import APISession
from datetime import datetime
from openprotein.api.data import *
import io
import pandas as pd
from urllib.parse import urljoin

from typing import List, Optional, Union
from io import BytesIO
import pydantic
from unittest.mock import ANY
import json
from openprotein.base import BearerAuth


class APISessionMock(APISession):
    """
    A mock class for APISession.
    """

    def __init__(self):
        username = "test_username"
        password = "test_password"
        super().__init__(username, password)

    def get_auth_token(self, username, password):
        return BearerAuth('AUTHORIZED')

    def post(self, endpoint, data=None, json=None, **kwargs):
        return ResponseMock()

    def get(self, endpoint, **kwargs):
        return ResponseMock()

    def request(self, method, url, *args, **kwargs):
        full_url = urljoin(self.backend, url)
        response = super().request(method, full_url, *args, **kwargs)
        response.raise_for_status()
        return response


class ResponseMock:
    def __init__(self):
        super().__init__()
        self._json = {}
        self.headers = {}
        self.iter_content = MagicMock()
        self._content = None
        self.status_code =200
        self.raw = io.BytesIO()  # Create an empty raw bytes stream
        self.text = "blank"

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    def json(self):
        return self._json

@pytest.fixture
def api_session_mock():
    return APISessionMock()

@pytest.fixture
def assay_metadata_mock():
    return AssayMetadata(
        assay_name="test_assay",
        assay_description="A test assay",
        assay_id="1234",
        original_filename="test.csv",
        created_date=datetime.now(),
        num_rows=10,
        num_entries=10,
        measurement_names=["test1", "test2"]
    )


# Test the assaydata_post function
def test_assaydata_post(api_session_mock, assay_metadata_mock):
    assay_file = BytesIO()
    assay_name = 'Test Assay'
    response_mock = ResponseMock()
    response_mock._json = assay_metadata_mock.dict()
    api_session_mock.post = MagicMock(return_value=response_mock)

    result = assaydata_post(api_session_mock, assay_file, assay_name)

    api_session_mock.post.assert_called_once_with(
        'v1/assaydata',
        files={'assay_data': assay_file},
        data={'assay_name': assay_name, 'assay_description': ''}
    )
    assert isinstance(result, AssayMetadata)
    assert result == assay_metadata_mock

# Test the assaydata_list function
def test_assaydata_list(api_session_mock, assay_metadata_mock):
    response_mock = ResponseMock()
    response_mock._json = [assay_metadata_mock.dict()]
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = assaydata_list(api_session_mock)

    api_session_mock.get.assert_called_once_with('v1/assaydata')
    assert isinstance(result, list)
    assert isinstance(result[0], AssayMetadata)
    assert result[0] == assay_metadata_mock

# Test the assaydata_put function
def test_assaydata_put(api_session_mock, assay_metadata_mock):
    new_assay_name = 'New Test Assay'
    response_mock = ResponseMock()
    response_mock._json = {**assay_metadata_mock.dict(), 'assay_name': new_assay_name}
    api_session_mock.put = MagicMock(return_value=response_mock)

    result = assaydata_put(api_session_mock, '1234', new_assay_name)

    api_session_mock.put.assert_called_once_with(
        'v1/assaydata/1234',
        data={'assay_name': new_assay_name}
    )
    assert isinstance(result, AssayMetadata)
    assert result.assay_name == new_assay_name


# Test the assaydata_page_get function
def test_assaydata_page_get(api_session_mock, assay_metadata_mock):
    response_mock = ResponseMock()
    assay_data_row = {"mut_sequence": "sequence", "measurement_values": [1.0, None]}
    response_mock._json = {"assaymetadata": assay_metadata_mock.dict(), "page_size": 1000, "page_offset": 0, "assaydata": [assay_data_row]}
    api_session_mock.get = MagicMock(return_value=response_mock)

    result = assaydata_page_get(api_session_mock, '1234')

    api_session_mock.get.assert_called_once_with(
        'v1/assaydata/1234',
        params={'page_offset': 0, 'page_size': 1000, 'format': 'wide'}
    )
    assert isinstance(result, AssayDataPage)
    assert isinstance(result.assaymetadata, AssayMetadata)
    assert result.assaymetadata == assay_metadata_mock


def test_assay_dataset_properties():
    metadata = AssayMetadata(
        assay_name='Test Assay',
        assay_description='Description',
        assay_id='1234',
        original_filename='file.csv',
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=['m1', 'm2']
    )
    session = MagicMock()
    dataset = AssayDataset(session, metadata)

    assert dataset.id == '1234'
    assert dataset.name == 'Test Assay'
    assert dataset.description == 'Description'
    assert dataset.measurement_names == ['m1', 'm2']
    assert len(dataset) == 1000
    assert dataset.shape == (1000, 3)


def test_assaydata_api_len(api_session_mock):
    metadata_mock1 = AssayMetadata(
        assay_name='Test Assay 1',
        assay_description='Description 1',
        assay_id='1234',
        original_filename='file1.csv',
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=['m1', 'm2']
    )
    metadata_mock2 = AssayMetadata(
        assay_name='Test Assay 2',
        assay_description='Description 2',
        assay_id='5678',
        original_filename='file2.csv',
        created_date=datetime.now(),
        num_rows=2000,
        num_entries=4000,
        measurement_names=['m3', 'm4']
    )
    response_mock = ResponseMock()
    response_mock._json = [metadata_mock1, metadata_mock2]
    api_session_mock.get = MagicMock(return_value=response_mock)

    data_api = DataAPI(api_session_mock)

    assert len(data_api) == 2


def test_assaydata_api_get_error(api_session_mock):
    metadata_mock = AssayMetadata(
        assay_name='Test Assay',
        assay_description='Description',
        assay_id='1234',
        original_filename='file.csv',
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=['m1', 'm2']
    )
    response_mock = ResponseMock()
    response_mock._json = [metadata_mock]
    api_session_mock.get = MagicMock(return_value=response_mock)

    data_api = DataAPI(api_session_mock)

    with pytest.raises(KeyError) as exc_info:
        data_api.get('5678')

    assert "No assay with id=5678 found".lower() in str(exc_info.value).lower()

def test_assaydata_api_get_400(api_session_mock):
    metadata_mock = AssayMetadata(
        assay_name='Test Assay',
        assay_description='Description',
        assay_id='1234',
        original_filename='file.csv',
        created_date=datetime.now(),
        num_rows=1000,
        num_entries=2000,
        measurement_names=['m1', 'm2']
    )
    response_mock = ResponseMock()
    response_mock.status_code = 400
    response_mock._json = [metadata_mock]
    api_session_mock.get = MagicMock(return_value=response_mock)

    data_api = DataAPI(api_session_mock)

    with pytest.raises(APIError) as exc_info:
        data_api.get('5678')

    assert "unable to list assay data".lower() in str(exc_info.value).lower()
