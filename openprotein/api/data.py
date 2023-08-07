import pandas as pd
import pydantic
from typing import Optional, List
from io import BytesIO

from openprotein.models import AssayMetadata, AssayDataPage
from openprotein.errors import APIError
from openprotein.base import APISession
import openprotein.config as config


def list_models(session: APISession, assay_id: str) -> List:
    """
    List models assoicated with assay.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_id : str
        assay ID

    Returns
    -------
    List
        List of models
    """
    endpoint = "v1/models"
    response = session.get(endpoint, params={"assay_id": assay_id})
    return response.json()


def assaydata_post(
    session: APISession,
    assay_file,
    assay_name: str,
    assay_description: Optional[str] = "",
) -> AssayMetadata:
    """
    Post assay data.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_file : str
        Path to the assay data file.
    assay_name : str
        Name of the assay.
    assay_description : str, optional
        Description of the assay, by default ''.

    Returns
    -------
    AssayMetadata
        Metadata of the posted assay data.
    """
    endpoint = "v1/assaydata"

    files = {"assay_data": assay_file}
    data = {"assay_name": assay_name, "assay_description": assay_description}

    response = session.post(endpoint, files=files, data=data)
    if response.status_code == 200:
        return pydantic.parse_obj_as(AssayMetadata, response.json())
    else:
        raise APIError(f"Unable to post assay data: {response.text}")


def assaydata_list(session: APISession) -> List[AssayMetadata]:
    """
    Get a list of all assay metadata.

    Parameters
    ----------
    session : APISession
        Session object for API communication.

    Returns
    -------
    List[AssayMetadata]
        List of all assay metadata.

    Raises
    ------
    APIError
        If an error occurs during the API request.
    """
    endpoint = "v1/assaydata"
    response = session.get(endpoint)
    if response.status_code == 200:
        return pydantic.parse_obj_as(List[AssayMetadata], response.json())
    else:
        raise APIError(f"Unable to list assay data: {response.text}")


def get_assay_metadata(session: APISession, assay_id: str) -> AssayMetadata:
    """
    Retrieve metadata for a specified assay.


    Parameters
    ----------
    session : APISession
        The current API session for communication with the server.
    assay_id : str
        The identifier of the assay for which metadata is to be retrieved.

    Returns
    -------
    AssayMetadata
        An AssayMetadata  that contains the metadata for the specified assay.

    Raises
    ------
    InvalidJob
        If no assay metadata with the specified assay_id is found.
    """

    endpoint = f"v1/assaydata/metadata"
    response = session.get(endpoint, params={"assay_id": assay_id})
    if response.status_code == 200:
        data = pydantic.parse_obj_as(AssayMetadata, response.json())
    else:
        raise APIError(f"Unable to list assay data: {response.text}")
    if data == []:
        raise APIError(f"No assay with id={assay_id} found")
    return data


def assaydata_put(
    session: APISession,
    assay_id: str,
    assay_name: Optional[str] = None,
    assay_description: Optional[str] = None,
) -> AssayMetadata:
    """
    Update assay metadata.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_id : str
        Id of the assay.
    assay_name : str, optional
        New name of the assay, by default None.
    assay_description : str, optional
        New description of the assay, by default None.

    Returns
    -------
    AssayMetadata
        Updated metadata of the assay.

    Raises
    ------
    APIError
        If an error occurs during the API request.
    """
    endpoint = f"v1/assaydata/{assay_id}"
    data = {}
    if assay_name is not None:
        data["assay_name"] = assay_name
    if assay_description is not None:
        data["assay_description"] = assay_description

    response = session.put(endpoint, data=data)
    if response.status_code == 200:
        return pydantic.parse_obj_as(AssayMetadata, response.json())
    else:
        raise APIError(f"Unable to update assay data: {response.text}")


def assaydata_page_get(
    session: APISession,
    assay_id: str,
    measurement_name: Optional[str] = None,
    page_offset: int = 0,
    page_size: int = 1000,
    data_format: str = "wide",
) -> AssayDataPage:
    """
    Get a page of assay data.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    assay_id : str
        Id of the assay.
    measurement_name : str, optional
        Name of the measurement, by default None.
    page_offset : int, optional
        Offset of the page, by default 0.
    page_size : int, optional
        Size of the page, by default 1000.
    data_format : str, optional
        data_format of the data, by default 'wide'.

    Returns
    -------
    AssayDataPage
        Page of assay data.

    Raises
    ------
    APIError
        If an error occurs during the API request.
    """
    endpoint = f"v1/assaydata/{assay_id}"

    params = {"page_offset": page_offset, "page_size": page_size, "format": data_format}
    if measurement_name is not None:
        params["measurement_name"] = measurement_name

    response = session.get(endpoint, params=params)
    if response.status_code == 200:
        return pydantic.parse_obj_as(AssayDataPage, response.json())
    else:
        raise APIError(f"Unable to get assay data page: {response.text}")


class AssayDataset:
    """Future Job for manipulating results"""
    def __init__(self, session: APISession, metadata: AssayMetadata):
        """
        init for AssayDataset.

        Parameters
        ----------
        session : APISession
            Session object for API communication.
        metadata : AssayMetadata
            Metadata object of the assay data.
        """
        self.session = session
        self.metadata = metadata
        self.page_size = config.BASE_PAGE_SIZE
        if self.page_size > 1000:
            self.page_size = 1000

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

    @property
    def sequence_length(self):
        return self.metadata.sequence_length

    def __len__(self):
        return self.metadata.num_rows

    @property
    def shape(self):
        return (len(self), len(self.measurement_names) + 1)

    def list_models(self):
        """
        List models assoicated with assay.

        Returns
        -------
        List
            List of models
        """
        return list_models(self.session, self.id)

    def update(
        self, assay_name: Optional[str] = None, assay_description: Optional[str] = None
    ) -> None:
        """
        Update the assay metadata.

        Parameters
        ----------
        assay_name : str, optional
            New name of the assay, by default None.
        assay_description : str, optional
            New description of the assay, by default None.

        Returns
        -------
        None
        """
        metadata = assaydata_put(
            self.session,
            self.id,
            assay_name=assay_name,
            assay_description=assay_description,
        )
        self.metadata = metadata

    def _get_all(self, verbose: bool = False) -> pd.DataFrame:
        """
        Get all assay data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing all assay data.
        """
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                result = self.get_slice(offset, offset + step)
                results.append(result)
                num_returned = len(result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return pd.concat(results)
        return pd.concat(results)

    def get_first(self) -> pd.DataFrame:
        """
        Get head slice of assay data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the slice of assay data.
        """
        rows = []
        entries = assaydata_page_get(self.session, self.id, page_offset=0, page_size=1)
        for row in entries.assaydata:
            row = [row.mut_sequence] + row.measurement_values
            rows.append(row)
        table = pd.DataFrame(rows, columns=["sequence"] + self.measurement_names)
        return table

    def get_slice(self, start: int, end: int) -> pd.DataFrame:
        """
        Get a slice of assay data.

        Parameters
        ----------
        start : int
            Start index of the slice.
        end : int
            End index of the slice.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the slice of assay data.
        """
        rows = []
        page_size = self.page_size
        # loop over the range
        for i in range(start, end, page_size):
            # the last page might be smaller than the page size
            current_page_size = min(page_size, end - i)

            entries = assaydata_page_get(
                self.session, self.id, page_offset=i, page_size=current_page_size
            )

            for row in entries.assaydata:
                row = [row.mut_sequence] + row.measurement_values
                rows.append(row)

        table = pd.DataFrame(rows, columns=["sequence"] + self.measurement_names)
        return table


class DataAPI:
    """API interface for calling AssayData endpoints"""

    def __init__(self, session: APISession):
        """
        init the DataAPI.

        Parameters
        ----------
        session : APISession
            Session object for API communication.
        """
        self.session = session

    def list(self) -> List[AssayDataset]:
        """
        List all assay datasets.

        Returns
        -------
        List[AssayDataset]
            List of all assay datasets.
        """
        metadata = assaydata_list(self.session)
        return [AssayDataset(self.session, x) for x in metadata]

    def create(
        self, table: pd.DataFrame, name: str, description: Optional[str] = None
    ) -> AssayDataset:
        """
        Create a new assay dataset.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame containing the assay data.
        name : str
            Name of the assay dataset.
        description : str, optional
            Description of the assay dataset, by default None.

        Returns
        -------
        AssayDataset
            Created assay dataset.
        """
        stream = BytesIO()
        table.to_csv(stream, index=False)
        stream.seek(0)
        metadata = assaydata_post(
            self.session, stream, name, assay_description=description
        )
        metadata.sequence_length = len(table["sequence"].values[0])
        return AssayDataset(self.session, metadata)

    def get(self, assay_id: str) -> AssayDataset:
        """
        Get an assay dataset by its ID.

        Parameters
        ----------
        assay_id : str
            ID of the assay dataset.

        Returns
        -------
        AssayDataset
            Assay dataset with the specified ID.

        Raises
        ------
        KeyError
            If no assay dataset with the given ID is found.
        """
        return get_assay_metadata(self.session, assay_id)

    def load_job(self, assay_id: str) -> AssayDataset:
        """
        Reload a Submitted job to resume from where you left off!


        Parameters
        ----------
        assay_id : str
            The identifier of the job whose details are to be loaded.

        Returns
        -------
        Job
            Job

        Raises
        ------
        HTTPError
            If the request to the server fails.
        InvalidJob
            If the Job is of the wrong type

        """
        metadata = self.get(assay_id)
        # if job_details.job_type != JobType.train:
        #    raise InvalidJob(f"Job {job_id} is not of type {JobType.train}")
        return AssayDataset(
            self.session,
            metadata,
        )

    def __len__(self) -> int:
        """
        Get the number of assay datasets.

        Returns
        -------
        int
            Number of assay datasets.
        """
        return len(self.list())
