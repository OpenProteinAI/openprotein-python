"""Schemas for the models API.

A model exposes methods addressed by a ``path`` under
``/api/v1/models/{model_id}/{path}``. Metadata carries the method list; each
method's request params are fetched separately as a JSON Schema.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MethodDescriptor",
    "ModelDescription",
    "ModelMetadata",
]


class ModelDescription(BaseModel):
    """Human-facing description of a model.

    Attributes
    ----------
    summary : str
        Description of the model's capabilities.
    citation_title : str or None
        Title of the model's reference publication.
    doi : str or None
        DOI of the model's reference publication.
    """

    summary: str
    citation_title: str | None = None
    doi: str | None = None


class MethodDescriptor(BaseModel):
    """A method supported by a model.

    Attributes
    ----------
    path : str
        The method's address under its model, such as ``batch/embeddings``.
    outputs : list of str
        Output names, with the method's default output first.
    description : str or None
        Description of the method's capabilities.
    name : str or None
        Display name of the method.
    """

    path: str = Field(
        description="The method's address under its model, e.g. ``batch/embeddings``."
    )
    outputs: list[str] = Field(
        min_length=1,
        description="Named outputs this method produces; the first is the default.",
    )
    description: str | None = None
    name: str | None = Field(
        default=None, description="Human-readable method name, when the model has one."
    )

    @model_validator(mode="before")
    @classmethod
    def _wire_shape(cls, data: Any) -> Any:
        """Accept an ``input``/``output`` pair as the path, and outputs keyed by name.

        Named outputs are ordered default first.
        """
        if not isinstance(data, dict):
            return data
        if "path" not in data and "input" in data and "output" in data:
            data = data | {"path": f"{data['input']}/{data['output']}"}
        if not isinstance(data.get("outputs"), dict):
            return data
        names = list(data["outputs"])
        default = data.get("default_output")
        if default in names:
            names.remove(default)
            names.insert(0, default)
        return data | {"outputs": names}

    @property
    def default_output(self) -> str:
        """The output to fetch when the caller doesn't choose one."""
        return self.outputs[0]


class ModelMetadata(BaseModel):
    """A model card and the methods it exposes.

    Attributes
    ----------
    model_id : str
        Identifier of the system model or fitted model.
    type : str or None
        Kind of fitted model, such as ``gp``, ``svd`` or ``umap``.
    family_id : str or None
        Model family identifier, such as ``esm``.
    model : str or None
        Label shared by the model's checkpoints.
    name : str or None
        Full display name of this model card.
    variant : str or None
        Label distinguishing this card within its model.
    is_system : bool
        Whether this is a system model.
    base : bool
        Whether this card represents its model in the model listing.
    tags : list of str or None
        Capability tags used to describe the model.
    input_types : list of str or None
        Input-category hints for model discovery; method schemas define inputs.
    description : ModelDescription
        Model summary and reference publication.
    dimension : int
        Representation dimension reported by the model.
    query : str or None
        Conditioning input type: ``sequence``, ``structure``, ``both`` or None.
    multichain : bool or None
        Whether the model supports multichain inputs, when specified.
    methods : list of MethodDescriptor
        Methods and named outputs available on this card.
    max_sequence_length : int
        Maximum sequence length reported by the model.
    owner : str or None
        Username of the fitted model's owner.
    created_at : datetime or None
        Creation time of a fitted model.
    spec : dict of str to Any or None
        Training configuration of a fitted model.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str = Field(
        description="The model's path under ``/models``: a system model's id, or "
        "``{type}/{id}`` for a fitted model."
    )
    type: str | None = Field(
        default=None, description="Fitted model kind, ``gp``, ``svd`` or ``umap``."
    )
    family_id: str | None = Field(
        default=None, description="Broad model family, e.g. ``esm``."
    )
    model: str | None = Field(
        default=None,
        description="Label shared by one model's checkpoints, e.g. ``ESM2``.",
    )
    name: str | None = Field(
        default=None, description="The model's full display name, e.g. ``ESM2 (650M)``."
    )
    variant: str | None = Field(
        default=None,
        description="What tells this card apart within its ``model``, e.g. ``650M``.",
    )
    is_system: bool = False
    base: bool = Field(
        default=False,
        description="The card that stands for its model; a model's only card is "
        "its base.",
    )
    tags: list[str] | None = None
    input_types: list[str] | None = None
    description: ModelDescription
    dimension: int
    query: str | None = None
    multichain: bool | None = None
    methods: list[MethodDescriptor]
    max_sequence_length: int
    owner: str | None = None
    created_at: datetime | None = None
    spec: dict[str, Any] | None = Field(
        default=None,
        description="Training-time configuration of a fitted model, by its ``kind``.",
    )
