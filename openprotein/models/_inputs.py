"""SDK objects a verb takes, converted to the values its card's params name."""

from collections.abc import Callable
from typing import Any

from openprotein.base import APISession
from openprotein.data.assaydataset import AssayDataset
from openprotein.data.schemas import AssayMetadata
from openprotein.jobs.futures import Future
from openprotein.molecules import Complex, Protein
from openprotein.prompt import Prompt, PromptAPI, Query
from openprotein.prompt.api import _assert_protein_only, _coerce_sequence
from openprotein.utils.uuid import _as_id

from .structure_generation import StructureGenerationFuture

QueryInput = str | bytes | Protein | Complex | Query
DesignInput = Future


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_prompt(prompt: Any, name: str = "prompt") -> None:
    from openprotein.data.prompts.prompts import Prompt as StoredPrompt

    if not isinstance(prompt, (str, Prompt, StoredPrompt)):
        raise TypeError(
            f"{name} must be a Prompt or a prompt id, not {_type_name(prompt)}"
        )


def _check_query(query: Any, name: str = "query") -> None:
    if isinstance(query, list) and not query:
        raise ValueError(f"{name} has no entries; pass {name}=None to omit it")
    for entry in query if isinstance(query, list) else [query]:
        if not isinstance(entry, QueryInput):
            raise TypeError(
                f"{name} must be a str, bytes, Protein, Complex, Query or a list of "
                f"these, not {_type_name(entry)}"
            )
    _check_query_entries(query if isinstance(query, list) else [query], [])


def prompt_id(prompt: Any) -> str | None:
    """A prompt's id, from the prompt or the id itself."""
    if prompt is None:
        return None
    _check_prompt(prompt)
    return prompt if isinstance(prompt, str) else prompt.id


def query_id(
    session: APISession, query: Any, force_structure: bool = False
) -> str | list[str] | None:
    """The query's id, creating a query from a sequence, protein or complex.

    A list gives one id per entry.
    """
    if query is None:
        return None
    _check_query(query)
    return PromptAPI(session)._resolve_query(
        query=query, force_structure=force_structure
    )


def _check_design(design: Any, name: str = "design") -> None:
    from openprotein.jobs import _dispatch

    if isinstance(design, str) and _as_id(design) is not None:
        return
    if isinstance(design, StructureGenerationFuture):
        return
    if (
        isinstance(design, DesignInput)
        and design.job is not None
        and _dispatch.serves_outputs(design.job)
    ):
        _, path = _dispatch.identity(design.job)
        if path and path.strip("/") != "generate/structures":
            raise TypeError(
                f"{name} job {design.id} is {path.strip('/')}, not generate/structures"
            )
        return
    raise TypeError(
        f"{name} must be a StructureDesignFuture, a generate/structures "
        f"Future, a StructureGenerationFuture or a design job's id, not "
        f"{_type_name(design)}"
    )


def design_id(design: Any) -> str | None:
    """A design job's id, from the job or the id itself."""
    if design is None:
        return None
    _check_design(design)
    return _as_id(design) if isinstance(design, str) else design.id


def check_one_structure_source(params: dict[str, Any], query: Any, design: Any) -> None:
    """Refuse a query and a design together, before anything is uploaded."""
    has_query = query is not None or params.get("query_id") is not None
    has_design = design is not None or params.get("design_id") is not None
    if has_query and has_design:
        raise TypeError("pass query or design, not both")


def _check_query_entries(queries: list[Any], complexes: list[Complex]) -> None:
    """Run the checks query creation runs, on every entry, before any is uploaded."""
    for entry in [*queries, *complexes]:
        if isinstance(entry, (str, bytes)) and _as_id(entry) is None:
            entry = _coerce_sequence("query", entry)
        if isinstance(entry, Complex):
            _assert_protein_only(entry)


def query_ids(
    session: APISession,
    query: Any = None,
    force_structure: bool = False,
    many: bool = False,
) -> str | list[str] | None:
    """``query_id`` from a query or a list of them.

    One id is sent bare, several as a list. Unless ``many``, more than one is
    refused before any query is created. A list that holds nothing is refused.
    """
    if query is not None:
        _check_query(query)
    count = len(query) if isinstance(query, list) else int(query is not None)
    if count > 1 and not many:
        raise ValueError(f"this method takes one query, not {count}")
    resolved = query_id(session, query, force_structure=force_structure)
    if resolved is None:
        return None
    if isinstance(resolved, list):
        return resolved[0] if len(resolved) == 1 else resolved
    return resolved


def takes_list(schema: dict[str, Any], name: str) -> bool:
    """Whether the params ``schema`` lets ``name`` be a list.

    A param the schema doesn't declare is left for the models API to refuse.
    """
    properties = schema.get("properties", {})
    if name not in properties:
        return True
    param = properties[name]
    return any(arm.get("type") == "array" for arm in param.get("anyOf", [param]))


def _check_assay(assay: Any, name: str = "assay") -> None:
    if not isinstance(assay, (str, AssayMetadata, AssayDataset)):
        raise TypeError(
            f"{name} must be an assay dataset, its metadata or its id, "
            f"not {_type_name(assay)}"
        )


#: Each SDK object a verb takes: its type check, and the param it converts to.
_INPUTS: dict[str, tuple[Callable[[Any, str], None], str | None]] = {
    "prompt": (_check_prompt, "prompt_id"),
    "query": (_check_query, "query_id"),
    "design": (_check_design, "design_id"),
    "assay": (_check_assay, None),
}


def check_inputs(params: dict[str, Any], **objects: Any) -> None:
    """Refuse a wrong SDK-object type, or an object passed with its param.

    Verbs call this before converting anything, so nothing is uploaded or
    submitted for a call that fails.
    """
    for name, value in objects.items():
        if value is None:
            continue
        check, param = _INPUTS[name]
        check(value, name)
        if param is not None and params.get(param) is not None:
            raise TypeError(f"pass {name} or {param}, not both")


def check_fit_source(sequences: Any, assay: Any) -> None:
    """Refuse a fit given neither or both of ``sequences`` and ``assay``."""
    if (sequences is None) == (assay is None):
        raise ValueError("pass one of sequences or assay")


def assay_id(assay: Any) -> str:
    """An assay's id, given the assay or its id."""
    _check_assay(assay)
    if isinstance(assay, AssayMetadata):
        return assay.assay_id
    if isinstance(assay, AssayDataset):
        return assay.id
    return assay


def model_id(model: Any) -> str:
    """The models API id of a model id or a ``get_model`` handle, fitted SVDs included."""
    from .model import Model

    if isinstance(model, str):
        return model
    if isinstance(model, Model):
        return model.id
    raise TypeError(
        "model must be a model id, a model from get_model or a fitted svd, "
        f"not {_type_name(model)}"
    )


def as_dataset(session: APISession, assay: Any) -> AssayDataset:
    """The assay dataset, given the dataset, its metadata or its id."""
    _check_assay(assay)
    if isinstance(assay, str):
        return AssayDataset._for(session)(id=assay)
    if isinstance(assay, AssayMetadata):
        return AssayDataset._for(session)(session, assay)
    return assay


def assay_sequences(session: APISession, assay: Any) -> list[str]:
    """The sequences of an assay, given the assay or its id.

    A page that fails to load raises.
    """
    dataset = as_dataset(session, assay)
    sequences: list[str] = []
    while True:
        page = dataset.get_slice(len(sequences), len(sequences) + dataset.page_size)
        sequences += [str(sequence) for sequence in page["sequence"]]
        if len(page) < dataset.page_size:
            return sequences


def with_param(params: dict[str, Any], name: str, value: Any, given_as: str) -> None:
    """Set ``params[name]`` from a converted SDK object, refusing both forms."""
    if value is None:
        return
    if params.get(name) is not None:
        raise TypeError(f"pass {given_as} or {name}, not both")
    params[name] = value
