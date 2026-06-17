"""Prompt REST API interface for making HTTP calls to the prompt backend."""

import io
import zipfile
from typing import BinaryIO, Sequence, cast

from openprotein.base import APISession
from openprotein.errors import APIError, InvalidParameterError, RawAPIError
from openprotein.molecules import Complex, Protein

from .schemas import Context, PromptMetadata, QueryMetadata


def _coerce_sequence(name: str, sequence: bytes | str) -> Protein | Complex:
    """Parse a raw sequence into a Protein, or a Complex if it contains ':' chain breaks."""
    raw = sequence.encode() if isinstance(sequence, str) else sequence
    parts = raw.split(b":")
    if any(len(p) == 0 for p in parts):
        raise InvalidParameterError("Invalid chain break usage in sequence")
    if len(parts) == 1:
        return Protein(name=name, sequence=parts[0])
    complex_ = Complex()
    for p in parts:
        complex_ &= Protein(sequence=p)
    complex_.name = name
    return complex_


def _assert_protein_only(complex_: Complex) -> None:
    """Raise InvalidParameterError if the Complex has any non-protein chain."""
    if len(complex_.get_chains()) != len(complex_.get_proteins()):
        raise InvalidParameterError(
            "prompts and queries support only protein chains; "
            "found non-protein chains in the input Complex"
        )


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
        Entries may be raw sequences (``bytes``/``str``), :py:class:`Protein`, or
        :py:class:`Complex`. Raw sequences may include ``:`` chain breaks to denote
        a multichain protein (e.g. ``"ACDE:GHIK"`` becomes a two-chain Complex).
        Currently only protein chains are accepted; passing a Complex with DNA, RNA,
        or Ligand chains raises :py:class:`InvalidParameterError`. This restriction
        may be relaxed in the future.
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


def get_prompt(session: APISession, prompt_id: str) -> list[list[Protein | Complex]]:
    """
    Get the prompt content for a given prompt ID.

    Single-chain entries collapse to :py:class:`Protein`; multichain entries are
    returned as :py:class:`Complex`. For a uniform return type, see
    :py:meth:`Prompt.get_as_complexes` or :py:meth:`Prompt.get_as_proteins`.

    Parameters
    ----------
    session : APISession
        The API session.
    prompt_id : str
        The prompt ID.

    Returns
    -------
    list of list of Protein or Complex
        The prompt data as a list of context entry lists.

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
        A list of context entries, or a group of such entries (for ensembles).
        Each entry is a raw sequence (``bytes``/``str``, optionally with ``:``
        chain breaks for multichain), :py:class:`Protein`, or :py:class:`Complex`.

    Returns
    -------
    list of io.BytesIO
        A list of in-memory zip files for the contexts.
    """
    if len(context) == 0:
        context = [[]]
    if isinstance(context[0], (bytes, str, Protein, Complex)):
        context = [cast(Context, context)]
    context = cast(Sequence[Context], context)

    context_zip_files = []
    for this_context in context:
        context_files: list[tuple[str, io.BytesIO]] = []
        for i, x in enumerate(this_context):
            if isinstance(x, (bytes, str)):
                x = _coerce_sequence(name=f"unnamed-{i:06}", sequence=x)
            elif not isinstance(x, (Protein, Complex)):
                raise InvalidParameterError(
                    f"unexpected context entry type: {type(x).__name__}"
                )
            name = x.name if x.name is not None else f"unnamed-{i:06}"
            index = len(context_files)
            if isinstance(x, Protein):
                has_struct = x.has_structure
            else:
                _assert_protein_only(x)
                has_struct = any(p.has_structure for p in x.get_proteins().values())
            if has_struct:
                context_files.append(
                    (
                        f"{index:06}.{name}.cif",
                        io.BytesIO(x.to_string().encode()),
                    )
                )
            else:
                # write sequences with no structure as fasta, continuing existing
                # fasta file if previous entry was also sequence only
                if len(context_files) == 0 or not context_files[-1][0].endswith(
                    ".fasta"
                ):
                    context_files.append((f"{index:06}.fasta", io.BytesIO()))
                _, current_file = context_files[-1]
                if isinstance(x, Protein):
                    seq = x.sequence
                else:
                    seq = b":".join(p.sequence for p in x.get_proteins().values())
                current_file.write(b">" + name.encode() + b"\n" + seq + b"\n")
        # generate context zip file
        in_memory_zip = io.BytesIO()
        with zipfile.ZipFile(in_memory_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, contents in context_files:
                zf.writestr(filename, contents.getvalue())
        in_memory_zip.seek(0)
        context_zip_files.append(in_memory_zip)

    return context_zip_files


def unzip_prompt(prompt_zip: BinaryIO) -> list[list[Protein | Complex]]:
    """
    Unzip a prompt zip file retrieved from the prompt API.

    This function is the reverse of zip_prompt. It extracts the context entries
    from a prompt zip file returned by get_prompt(). Single-chain entries are
    returned as :py:class:`Protein`; multichain entries as :py:class:`Complex`.

    Parameters
    ----------
    prompt_zip : BinaryIO
        The binary data of the prompt zip file returned by get_prompt().

    Returns
    -------
    list of list of Protein or Complex
        List of context entry lists, where each inner list represents a context group.
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
) -> list[list[Protein | Complex]]:
    """Parse context files into Protein or Complex entries.

    Single-chain entries collapse to :py:class:`Protein`; multichain entries are
    returned as :py:class:`Complex`.
    """
    context: list[list[Protein | Complex]] = []

    for context_file in context_files:
        context_file.seek(0)
        entries_in_context: list[Protein | Complex] = []

        with zipfile.ZipFile(context_file, "r") as zf:
            filenames = zf.namelist()

            for filename in filenames:
                with zf.open(filename) as f:
                    content = f.read()

                    if filename.endswith(".cif") or filename.endswith(".pdb"):
                        # Structure files may carry one or many chains; load as
                        # Complex and collapse to Protein only when there's
                        # exactly one protein chain and no other chain types.
                        fmt = "cif" if filename.endswith(".cif") else "pdb"
                        name = filename[:-4]
                        complex_ = Complex.from_string(
                            filestring=content, format=fmt, verbose=False
                        )
                        complex_.name = name
                        proteins = complex_.get_proteins()
                        if len(proteins) == 1 and len(complex_.get_chains()) == 1:
                            protein = next(iter(proteins.values()))
                            protein.name = name
                            entries_in_context.append(protein)
                        else:
                            entries_in_context.append(complex_)

                    elif filename.endswith(".fasta"):
                        from openprotein import fasta

                        fasta_stream = io.BytesIO(content)
                        for name, sequence in fasta.parse_stream(fasta_stream):
                            entry_name = (
                                name.decode() if isinstance(name, bytes) else name
                            )
                            entries_in_context.append(
                                _coerce_sequence(name=entry_name, sequence=sequence)
                            )

                    else:
                        raise APIError(
                            f"Unrecognized prompt context file extension: "
                            f"{filename!r}; expected .cif, .pdb, or .fasta"
                        )

        context.append(entries_in_context)

    return context


def create_query(
    session: APISession,
    query: bytes | str | Protein | Complex,
    force_structure: bool = False,
) -> QueryMetadata:
    """
    Create a query.

    Parameters
    ----------
    session : APISession
        The API session.
    query : bytes or str or Protein or Complex
        A query protein or complex. Raw ``bytes``/``str`` inputs may include ``:``
        chain breaks to denote a multichain protein (e.g. ``"ACDE:GHIK"`` becomes
        a two-chain Complex). Currently only protein chains are accepted; passing a
        Complex with DNA, RNA, or Ligand chains raises :py:class:`InvalidParameterError`.
        This restriction may be relaxed in the future.
    force_structure : bool, optional
        Optionally force a query to be interpreted with a structure.
        Useful for creating structure prediction queries which can have
        no structure.

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

    if isinstance(query, (str, bytes)):
        query = _coerce_sequence("query", query)
    if isinstance(query, Protein):
        has_structure = query.has_structure
        if has_structure or force_structure:
            qf, filename, typ = (
                query.to_string(format="cif").encode(),
                "query.cif",
                "chemical/x-mmcif",
            )
        else:
            name = query.name or "query"
            qf, filename, typ = (
                b">" + name.encode() + b"\n" + query.sequence + b"\n",
                "query.fasta",
                "text/x-fasta",
            )
    elif isinstance(query, Complex):
        _assert_protein_only(query)
        has_structure = any(p.has_structure for p in query.get_proteins().values())
        if has_structure or force_structure:
            qf, filename, typ = (
                query.to_string("cif").encode(),
                "query.cif",
                "chemical/x-mmcif",
            )
        else:
            name = query.name or "query"
            seq = b":".join(p.sequence for p in query.get_proteins().values())
            qf, filename, typ = (
                b">" + name.encode() + b"\n" + seq + b"\n",
                "query.fasta",
                "text/x-fasta",
            )
    else:
        raise InvalidParameterError(
            f"unexpected query type: {type(query).__name__}"
        )

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


def get_query(session: APISession, query_id: str) -> Protein | Complex:
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
    Protein or Complex
        The query content. Single-chain entries collapse to :py:class:`Protein`;
        multichain entries are returned as :py:class:`Complex`.

    Raises
    ------
    APIError
        If the API returns an error or the file format is unexpected.
    """
    endpoint = f"v1/prompt/query/{query_id}/content"
    response = session.get(endpoint, stream=True)

    if response.status_code == 401:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code == 404:
        error = RawAPIError.model_validate(response.json())
        raise APIError(error.detail)
    elif response.status_code != 200:
        raise APIError(f"Unexpected response status code: {response.status_code}")

    filename = response.headers.get("Content-Disposition", "query")
    media_type = response.headers.get("Content-Type", "text/plain")
    is_mmcif = filename.endswith(".cif") or media_type == "chemical/x-mmcif"
    is_fasta = filename.endswith(".fasta") or media_type == "text/x-fasta"

    query = None
    if is_mmcif:
        complex_ = Complex.from_string(
            filestring=response.content, format="cif", verbose=False
        )
        proteins = complex_.get_proteins()
        if len(proteins) == 1 and len(complex_.get_chains()) == 1:
            query = next(iter(proteins.values()))
        else:
            query = complex_

    elif is_fasta:
        # Process FASTA file - take only the first sequence
        import io

        from openprotein import fasta

        fasta_stream = io.BytesIO(response.content)
        for name, sequence in fasta.parse_stream(fasta_stream):
            entry_name = name.decode() if isinstance(name, bytes) else name
            query = _coerce_sequence(name=entry_name, sequence=sequence)
            break  # Only take the first sequence
    else:
        raise APIError(
            f"Unexpected file returned with filename {filename} and type {media_type}"
        )

    if query is None:
        raise APIError(f"Invalid query file returned from API {response.content[:10]}")

    return query
