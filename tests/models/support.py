"""A session serving the models API's cards and params, and checking submissions.

``tests/data/models_params.json`` holds each method's params JSON Schema and
outputs, keyed ``<model>/<method path>`` (fitted families as ``gp/...``,
``svd/...``, ``umap/...``), and the fit bodies under ``fits``. Every body
posted through ``ModelsSession`` is checked against its schema.
"""

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from jsonschema import Draft202012Validator, FormatChecker

from openprotein import codecs
from openprotein.base import APISession
from openprotein.prompt import PromptAPI

PARAMS = json.loads(
    (Path(__file__).parents[1] / "data" / "models_params.json").read_text()
)
FITTED_FAMILIES = ("gp", "svd", "umap")

_ALL_CHAINS = ["protein", "dna", "rna", "ligand"]

#: the inputs each structure prediction card names
CARD_FACTS: dict[str, dict[str, Any]] = {
    "esmfold": {"input_types": ["protein"], "multichain": True},
    "minifold": {"input_types": ["protein"], "multichain": False},
    "alphafold2": {"input_types": ["protein"], "multichain": True},
    "rosettafold-3": {"input_types": ["protein", "ligand"], "multichain": True},
    **{
        model_id: {"input_types": _ALL_CHAINS, "multichain": True}
        for model_id in (
            "boltz-1",
            "boltz-1x",
            "boltz-2",
            "protenix",
            "protenix-v2",
            "esmfold2",
            "esmfold2-fast",
            "opendde",
            "opendde-abag",
        )
    },
}
CARD_FACTS.update(
    json.loads((Path(__file__).parents[1] / "data" / "ensemble_cards.json").read_text())
)

SUBMITTED_JOB: dict[str, Any] = {
    "job_id": "e0b1c3d4-0000-4000-8000-00000000000a",
    "job_type": "/models",
    "status": "SUCCESS",
    "created_date": "2026-10-21T08:00:00Z",
}


def created_query(n: int) -> str:
    """The id ``ModelsSession`` gives the ``n``-th query created, from 1."""
    return f"aaaaaaaa-0000-4000-8000-{n:012d}"


def _family(model_id: str) -> str:
    family = model_id.split("/", 1)[0]
    return family if family in FITTED_FAMILIES and "/" in model_id else model_id


def card(model_id: str) -> dict[str, Any]:
    """The model's card, listing the methods the params fixture has for it."""
    prefix = f"{_family(model_id)}/"
    methods = [
        {"path": key.removeprefix(prefix), "outputs": method["outputs"]}
        for key, method in PARAMS["methods"].items()
        if key.startswith(prefix)
    ]
    return {
        "model_id": model_id,
        "description": {"summary": model_id},
        "dimension": 1,
        "max_sequence_length": 1024,
        "methods": methods,
        **CARD_FACTS.get(model_id, {}),
    }


def _schema(path: str) -> dict[str, Any]:
    """The params schema a ``POST v1/models/...`` path submits against."""
    route = path.removeprefix("v1/models/")
    if route in PARAMS["fits"]:
        return PARAMS["fits"][route]
    if route.startswith("gp/") and route.endswith("/crossvalidate"):
        return PARAMS["fits"]["gp/crossvalidate"]
    # every method path in the fixture is two segments, so the model id is the rest
    model, first, last = route.rsplit("/", 2)
    key = f"{_family(model)}/{first}/{last}"
    assert key in PARAMS["methods"], f"the model listing has no method {key}"
    return PARAMS["methods"][key]["params"]


def check_body(path: str, body: dict[str, Any]) -> None:
    """Raise unless the params schema for ``path`` accepts ``body``.

    Every key must be a declared param, whether or not the schema forbids
    extras. A UMAP fit that lists sequences takes at least
    ``max(3, n_components + 2)`` of them.
    """
    schema = _schema(path)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(body)
    undeclared = set(body) - set(schema.get("properties", {}))
    assert not undeclared, f"{path} declares no {sorted(undeclared)}"
    if path == "v1/models/umap" and "sequences" in body:
        minimum = max(3, body.get("n_components", 2) + 2)
        assert len(body["sequences"]) >= minimum, f"UMAP fit needs {minimum}"


class Stored:
    """An output value stored as a ref, encoded by ``codec``; None stores bytes."""

    def __init__(self, codec: str | None, value: Any):
        if codec is None:
            self.data = value
        else:
            registered = codecs.get(codec)
            assert registered is not None, f"no codec {codec}"
            self.data = registered.encode(value)
        self.codec = codec


class ModelsSession:
    """Serves cards and outputs; ``post`` answers a query upload with a new id.

    ``outputs[(job_id, output)]`` lists one value per position: a ``Stored``
    ref, or an inline value.
    """

    def __init__(self):
        self.session = MagicMock(spec=APISession)
        self.session.timeout = 180
        self.session.headers = {}
        self.session.prompt = PromptAPI(self.session)
        self.queries: list[tuple[str, bytes]] = []
        self._query_ids = (created_query(n) for n in itertools.count(1))
        self.session.post.side_effect = self._post
        self.session.get.side_effect = self._get
        self.get_routes: dict[str, Any] = {}
        self.post_routes: dict[str, Any] = {}
        self.outputs: dict[tuple[str, str], list[Any]] = {}
        self.blobs: dict[str, bytes] = {}

    def _response(self, body: Any) -> MagicMock:
        response = MagicMock(status_code=200)
        response.json.return_value = body
        return response

    def _post(self, path: str, **kwargs: Any) -> MagicMock:
        if path == "v1/prompt/query":
            filename, content, _ = kwargs["files"]["query"]
            self.queries.append((filename, content.read()))
            return self._response(
                {"id": next(self._query_ids), "created_date": "2026-10-21T08:00:00Z"}
            )
        check_body(path, kwargs["json"])
        return self._response(self.post_routes.get(path, SUBMITTED_JOB))

    def _value(self, job_id: str, output: str, position: int) -> dict[str, Any]:
        value = self.outputs[(job_id, output)][position]
        if not isinstance(value, Stored):
            return {"inline": value}
        uri = f"s3://results/{job_id}/{output}/{position}"
        self.blobs[uri] = value.data
        return {
            "ref": {
                "uri": uri,
                "byte_length": len(value.data),
                "sha256": hashlib.sha256(value.data).hexdigest(),
                "codec": value.codec,
            }
        }

    def _results(self, job_id: str, output: str) -> dict[str, Any]:
        return {
            "output": output,
            "values": [
                {
                    "position": position,
                    "unit_id": f"u{position}",
                    "value": self._value(job_id, output, position),
                }
                for position in range(len(self.outputs[(job_id, output)]))
            ],
            "next_cursor": None,
        }

    def _get(self, path: str, **kwargs: Any) -> MagicMock:
        if path in self.get_routes:
            return self._response(self.get_routes[path])
        if path.startswith("v1/data/results/"):
            job_id, _, output = path.removeprefix("v1/data/results/").partition("/")
            if not output:
                output = next(o for j, o in self.outputs if j == job_id)
            return self._response(self._results(job_id, output))
        if path.startswith("v1/models/") and path.endswith("/params"):
            return self._response(_schema(path.removesuffix("/params")))
        return self._response(card(path.removeprefix("v1/models/")))

    @property
    def submissions(self) -> list[tuple[str, Any]]:
        """``(path, body)`` of every post other than a query upload."""
        return [
            (call.args[0], call.kwargs.get("json"))
            for call in self.session.post.call_args_list
            if call.args[0] != "v1/prompt/query"
        ]

    @property
    def submitted(self) -> tuple[str, Any]:
        (submission,) = self.submissions
        return submission
