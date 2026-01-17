"""Model metadata for OpenProtein models."""

from pydantic import BaseModel, Field


class ModelDescription(BaseModel):
    """Description of available protein embedding models."""

    citation_title: str | None = None
    doi: str | None = None
    summary: str = "Protein language model for embeddings"


class TokenInfo(BaseModel):
    """Information about the tokens used in the embedding model."""

    id: int
    token: str
    primary: bool
    description: str


class ModelMetadata(BaseModel):
    """Metadata about available protein embedding models."""

    id: str = Field(..., alias="model_id")
    description: ModelDescription
    max_sequence_length: int | None = None
    dimension: int | None = None
    output_types: list[str] | None = None
    input_tokens: list[str] | None = None
    output_tokens: list[str] | None = None
    token_descriptions: list[list[TokenInfo]] | None = None
