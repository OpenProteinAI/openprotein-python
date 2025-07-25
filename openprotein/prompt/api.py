"""Prompt REST API interface for making HTTP calls to the prompt backend."""

import copy
import io
import zipfile
from typing import BinaryIO, Sequence, cast

from openprotein.base import APISession
from openprotein.errors import APIError, InvalidParameterError, RawAPIError
from openprotein.protein import Protein

from .schemas import Context, PromptMetadata, QueryMetadata


def create_prompt(
    session: APISession,
    context: Context | Sequence[Context],
    name: str | None = None,
    description: str | None = None,
) -> PromptMetadata:
    """
    Create a prompt.

    Parameters
    ----------
    session : APISession
        The API session.
    context : Context or Sequence[Context]
        Context or list of contexts, each of which is a list of sequences/structures.
    name : str or None, optional
        Name of the prompt.
    description : str or None, optional
        Description of the prompt.

    Returns
    -------
    PromptMetadata
        Metadata of the created prompt.

    Raises
    ------
    InvalidParameterError
        If the parameters are invalid.
    APIError
        If the API returns an error.
    """
    endpoint = "v1/prompt/create_prompt"
    data = {}
    if name is not None:
        data["name"] = name
    if description is not None:
        data["description"] = description

    context_zip_files = zip_prompt(context=context)

    files = [
        ("context", (f"context-{i}.zip", context_zip_file, "application/zip"))
        for i, context_zip_file in enumerate(context_zip_files)
    ]
    form: dict = {
        "files": files,
    }
    if len(data) > 0:
        form["data"] = data

    response = session.post(endpoint, **form)

    if response.status_code == 200:
        return PromptMetadata.model_validate(response.json())
    elif response.status_code == 400:
        error = RawAPIError.model_validate(response.json())
        raise InvalidParameterError(error.detail)
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def get_prompt_metadata(session: APISession, prompt_id: str) -> PromptMetadata:
    """
    Get metadata for a given prompt ID.

    Parameters
    ----------
    session : APISession
        The API session.
    prompt_id : str
        The prompt ID.

    Returns
    -------
    PromptMetadata
        Metadata of the prompt.

    Raises
    ------
    APIError
        If the API returns an error.
    """
    endpoint = f"v1/prompt/{prompt_id}"
    response = session.get(endpoint)

    if response.status_code == 200:
        return PromptMetadata.model_validate(response.json())
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code == 404:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def get_prompt(session: APISession, prompt_id: str) -> list[list[Protein]]:
    """
    Get the prompt content for a given prompt ID.

    Parameters
    ----------
    session : APISession
        The API session.
    prompt_id : str
        The prompt ID.

    Returns
    -------
    list of list of Protein
        The prompt data as a list of context protein lists.

    Raises
    ------
    APIError
        If the API returns an error.
    """
    endpoint = f"v1/prompt/{prompt_id}/content"
    response = session.get(endpoint, stream=True)

    if response.status_code == 200:
        return unzip_prompt(io.BytesIO(response.content))
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code == 404:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def list_prompts(session: APISession) -> list[PromptMetadata]:
    """
    List all prompts.

    Parameters
    ----------
    session : APISession
        The API session.

    Returns
    -------
    list of PromptMetadata
        List of prompt metadata.

    Raises
    ------
    APIError
        If the API returns an error.
    """
    endpoint = "v1/prompt"
    response = session.get(endpoint)

    if response.status_code == 200:
        return [PromptMetadata.model_validate(prompt) for prompt in response.json()]
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def zip_prompt(
    context: Context | Sequence[Context],
) -> list[io.BytesIO]:
    """
    Zip a prompt context to prepare for upload.

    Parameters
    ----------
    context : Context or Sequence[Context]
        A list of proteins, or a group of such proteins (for ensembles), representing the context for the prompt.

    Returns
    -------
    list of io.BytesIO
        A list of in-memory zip files for the contexts.
    """
    if len(context) == 0:
        context = [[]]
    if isinstance(context[0], (bytes, str, Protein)):
        context = [cast(Context, context)]
    context = cast(Sequence[Context], context)

    context_zip_files = []
    for this_context in context:
        this_context_as_proteins: list[Protein] = []
        for i, x in enumerate(this_context):
            if not isinstance(x, Protein):
                x = Protein(name=f"unnamed-{i:06}", sequence=x)
            else:
                x = copy.copy(x)
            if x.name is None:
                x.name = f"unnamed-{i:06}"
            this_context_as_proteins.append(x)
        context_files: list[tuple[str, io.BytesIO]] = []
        for protein in this_context_as_proteins:
            index = len(context_files)
            if protein.has_structure:
                context_files.append(
                    (
                        f"{index:06}.{protein.name}.cif",
                        io.BytesIO(protein.make_cif_string().encode()),
                    )
                )
            else:
                # write sequences with no structure as fasta, continuing existing fasta file
                # if previous protein was sequence only
                if len(context_files) == 0 or not context_files[-1][0].endswith(
                    ".fasta"
                ):
                    context_files.append((f"{index:06}.fasta", io.BytesIO()))
                _, current_file = context_files[-1]
                current_file.write(protein.make_fasta_bytes())
        # generate context zip file
        in_memory_zip = io.BytesIO()
        with zipfile.ZipFile(in_memory_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, contents in context_files:
                zf.writestr(filename, contents.getvalue())
        in_memory_zip.seek(0)
        context_zip_files.append(in_memory_zip)

    return context_zip_files


def unzip_prompt(prompt_zip: BinaryIO) -> list[list[Protein]]:
    """
    Unzip a prompt zip file retrieved from the prompt API.

    This function is the reverse of zip_prompt. It extracts the context proteins
    from a prompt zip file returned by get_prompt().

    Parameters
    ----------
    prompt_zip : BinaryIO
        The binary data of the prompt zip file returned by get_prompt().

    Returns
    -------
    list of list of Protein
        List of context protein lists, where each inner list represents a context group.
    """
    context_zip_files = []
    with zipfile.ZipFile(prompt_zip, "r") as zip_file:
        file_names = zip_file.namelist()

        for file_name in file_names:
            if file_name.startswith("context-"):
                context_zip_file = io.BytesIO(zip_file.read(file_name))
                context_zip_files.append(context_zip_file)
    context = __parse_prompt(context_files=context_zip_files)

    return context


def __parse_prompt(
    context_files: Sequence[BinaryIO],
) -> list[list[Protein]]:
    """
    Parse context and query files into proteins.

    Parameters
    ----------
    context_files : Sequence[BinaryIO]
        Sequence of binary zip files, each representing a context group.

    Returns
    -------
    list of list of Protein
        List of context protein lists, where each inner list represents a context group.
    """
    context: list[list[Protein]] = []

    # Process each context file (representing an ensemble)
    for context_file in context_files:
        # Reset the file pointer to the beginning
        context_file.seek(0)
        proteins_in_context: list[Protein] = []

        with zipfile.ZipFile(context_file, "r") as zf:
            # Sort filenames to process them in a consistent order
            filenames = zf.namelist()

            # Process each file in the zip
            for filename in filenames:
                with zf.open(filename) as f:
                    content = f.read()

                    if filename.endswith(".cif"):
                        # For CIF files, create a temporary file for gemmi to read
                        import tempfile

                        with tempfile.NamedTemporaryFile(
                            suffix=".cif", delete=True
                        ) as tmp:
                            tmp.write(content)
                            tmp.flush()
                            # extract chain ID (using 'A' as default)
                            chain_id = "A"
                            # extract name from filename (without extension)
                            name = filename[:-4]
                            protein = Protein.from_filepath(
                                path=tmp.name, chain_id=chain_id, verbose=False
                            )
                            # override the name with the filename
                            protein.name = name
                            proteins_in_context.append(protein)

                    elif filename.endswith(".fasta"):
                        # Process FASTA file
                        import io

                        from openprotein import fasta

                        fasta_stream = io.BytesIO(content)
                        for name, sequence in fasta.parse_stream(fasta_stream):
                            proteins_in_context.append(
                                Protein(name=name, sequence=sequence)
                            )

        # Add this group of proteins to the context
        context.append(proteins_in_context)

    return context


def create_query(
    session: APISession,
    query: bytes | str | Protein,
) -> QueryMetadata:
    """
    Create a query.

    Parameters
    ----------
    session : APISession
        The API session.
    query : bytes or str or Protein
        A query representing a protein to be used with a query.

    Returns
    -------
    QueryMetadata
        Metadata of the created query.

    Raises
    ------
    InvalidParameterError
        If the parameters are invalid.
    APIError
        If the API returns an error.
    """
    endpoint = "v1/prompt/query"

    if not isinstance(query, Protein):
        query = Protein(name="query", sequence=query)
    if query.has_structure:
        qf, filename, typ = (
            query.make_cif_string().encode(),
            "query.cif",
            "chemical/x-mmcif",
        )
    else:
        qf, filename, typ = query.make_fasta_bytes(), "query.fasta", "text/x-fasta"

    response = session.post(endpoint, files={"query": (filename, io.BytesIO(qf), typ)})

    if response.status_code == 200:
        return QueryMetadata.model_validate(response.json())
    elif response.status_code == 400:
        error = RawAPIError.model_validate(response.json())
        raise InvalidParameterError(error.detail)
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def get_query_metadata(session: APISession, query_id: str) -> QueryMetadata:
    """
    Get metadata for a given query ID.

    Parameters
    ----------
    session : APISession
        The API session.
    query_id : str
        The query ID.

    Returns
    -------
    QueryMetadata
        Metadata of the query.

    Raises
    ------
    APIError
        If the API returns an error.
    """
    endpoint = f"v1/prompt/query/{query_id}"
    response = session.get(endpoint)

    if response.status_code == 200:
        return QueryMetadata.model_validate(response.json())
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code == 404:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")


def get_query(session: APISession, query_id: str) -> Protein:
    """
    Get the query content for a given query ID.

    Parameters
    ----------
    session : APISession
        The API session.
    query_id : str
        The query ID.

    Returns
    -------
    Protein
        The query protein.

    Raises
    ------
    APIError
        If the API returns an error or the file format is unexpected.
    """
    endpoint = f"v1/prompt/query/{query_id}/content"
    response = session.get(endpoint, stream=True)
    filename = response.headers.get("Content-Disposition", "query")
    media_type = response.headers.get("Content-Type", "text/plain")
    is_mmcif = filename.endswith(".cif") or media_type == "chemical/x-mmcif"
    is_fasta = filename.endswith(".fasta") or media_type == "text/x-fasta"

    query_protein = None
    if is_mmcif:
        # for cif files, create a temporary file for gemmi to read
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cif", delete=True) as tmp:
            tmp.write(response.content)
            tmp.flush()
            # extract chain id (using 'A' as default)
            chain_id = "A"
            query_protein = Protein.from_filepath(
                path=tmp.name, chain_id=chain_id, verbose=False
            )

    elif is_fasta:
        # Process FASTA file - take only the first sequence
        import io

        from openprotein import fasta

        fasta_stream = io.BytesIO(response.content)
        for name, sequence in fasta.parse_stream(fasta_stream):
            query_protein = Protein(name=name, sequence=sequence)
            break  # Only take the first sequence
    else:
        raise APIError(
            f"Unexpected file returned with filename {filename} and type {media_type}"
        )

    if query_protein is None:
        raise APIError(f"Invalid query file returned from API {response.content[:10]}")

    if response.status_code == 200:
        return query_protein
    elif response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code == 404:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    else:
        raise APIError(f"Unexpected response status code: {response.status_code}")
