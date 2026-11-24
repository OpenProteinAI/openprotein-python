from pydantic import BaseModel

from openprotein.base import APISession
from openprotein.data._reads import read

from .schemas import MSADownloadFormat, MSAFormat, MSAMetadata


class MSAPage(BaseModel):
    """One page of MSA metadata."""

    items: list[MSAMetadata]
    total: int
    limit: int
    offset: int


def msa_post(
    session: APISession,
    file: bytes,
    format: MSAFormat,
    name: str | None = None,
    description: str | None = None,
    filename: str | None = None,
) -> MSAMetadata:
    """
    Upload an MSA, declaring its ``format``.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    file : bytes
        The MSA contents to upload.
    format : MSAFormat
        The upload's format (one of "a2m", "a3m", "csv", "fasta", "sto"), naming its
        parser.
    name, description : str, optional
        Human-readable metadata to store alongside the MSA.
    filename : str, optional
        Original filename to record; defaults to a name derived from ``format``.

    Returns
    -------
    MSAMetadata
        Metadata of the stored MSA.
    """
    endpoint = "v1/data/msa"
    data: dict = {"format": format}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description
    files = {"file": (filename or f"msa.{format}", file)}
    response = session.post(endpoint, data=data, files=files)
    return MSAMetadata.model_validate(response.json())


def msa_list(
    session: APISession,
    limit: int | None = None,
    offset: int = 0,
) -> MSAPage:
    """
    List one page of the caller's stored MSAs (metadata only).

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    limit : int, optional
        Maximum number of records to return; the server pages when left unset.
    offset : int
        Offset into the result set.

    Returns
    -------
    MSAPage
        A page of MSA metadata and the total across pages.
    """
    endpoint = "v1/data/msa"
    params: dict = {"offset": offset}
    if limit is not None:
        params["limit"] = limit
    response = session.get(endpoint, params=params)
    return MSAPage.model_validate(response.json())


def msa_get_metadata(session: APISession, msa_id: str) -> MSAMetadata:
    """
    Retrieve metadata for one MSA.

    A deleted or unknown id is a 404.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa_id : str
        Id of the MSA.

    Returns
    -------
    MSAMetadata
        Metadata for the MSA.
    """
    response = read(session, f"v1/data/msa/{msa_id}/metadata")
    return MSAMetadata.model_validate(response.json())


def msa_download(
    session: APISession, msa_id: str, format: MSADownloadFormat | None = None
) -> bytes:
    """
    Download a live MSA's alignment bytes, optionally converted to ``format``.

    Raises a 404 unless the MSA is live.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa_id : str
        Id of the MSA.
    format : MSADownloadFormat, optional
        Render to this format (one of "a2m", "a3m", "csv", "fasta", "sto"), or
        "parquet" for the stored ``msa-parquet-v1`` bytes; defaults to the
        source format.

    Returns
    -------
    bytes
        The rendered alignment.
    """
    params = {"format": format} if format is not None else None
    response = read(session, f"v1/data/msa/{msa_id}", params=params)
    return response.content


def msa_put(
    session: APISession,
    msa_id: str,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> MSAMetadata:
    """
    Update the mutable metadata (name / description / tags) of an MSA.

    Only the fields provided are changed; the alignment itself is immutable.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa_id : str
        Id of the MSA.
    name, description : str, optional
        New name / description.
    tags : list[str], optional
        Replacement tag list.

    Returns
    -------
    MSAMetadata
        The updated metadata.
    """
    body: dict = {}
    if name is not None:
        body["name"] = name
    if description is not None:
        body["description"] = description
    if tags is not None:
        body["tags"] = tags
    if not body:
        # nothing to change - skip the no-op PUT and return the current metadata
        return msa_get_metadata(session, msa_id)
    response = session.put(f"v1/data/msa/{msa_id}", json=body)
    return MSAMetadata.model_validate(response.json())


def msa_delete(session: APISession, msa_id: str) -> None:
    """
    Delete an MSA.

    Parameters
    ----------
    session : APISession
        Session object for API communication.
    msa_id : str
        Id of the MSA.
    """
    session.delete(f"v1/data/msa/{msa_id}")
