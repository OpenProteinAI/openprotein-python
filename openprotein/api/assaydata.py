from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.schemas import AssayDataPage, AssayMetadata
from pydantic import TypeAdapter


def list_models(session: APISession, assay_id: str) -> list:
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
    assay_description: str | None = "",
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
        return TypeAdapter(AssayMetadata).validate_python(response.json())
    else:
        raise APIError(f"Unable to post assay data: {response.text}")


def assaydata_list(session: APISession) -> list[AssayMetadata]:
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
        return TypeAdapter(list[AssayMetadata]).validate_python(response.json())
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

    endpoint = "v1/assaydata/metadata"
    response = session.get(endpoint, params={"assay_id": assay_id})
    if response.status_code == 200:
        data = TypeAdapter(AssayMetadata).validate_python(response.json())
    else:
        raise APIError(f"Unable to list assay data: {response.text}")
    if data == []:
        raise APIError(f"No assay with id={assay_id} found")
    return data


def assaydata_put(
    session: APISession,
    assay_id: str,
    assay_name: str | None = None,
    assay_description: str | None = None,
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
        return TypeAdapter(AssayMetadata).validate_python(response.json())
    else:
        raise APIError(f"Unable to update assay data: {response.text}")


def assaydata_page_get(
    session: APISession,
    assay_id: str,
    measurement_name: str | None = None,
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
        return TypeAdapter(AssayDataPage).validate_python(response.json())
    else:
        raise APIError(f"Unable to get assay data page: {response.text}")
