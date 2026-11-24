"""The fold request contract: the entities of a complex and their chain wrappers.

A fold request carries ``complexes``, a list of complexes, each a list of
chains. A chain names exactly one entity; the wrapper classes scope which
entity types a model accepts.
"""

import re
from collections.abc import Sequence
from typing import Annotated, Any
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

CHAIN_ID = re.compile(r"[A-Za-z0-9]{1,5}")


def _check_chain_id(value: str | list[str]) -> str | list[str]:
    ids = [value] if isinstance(value, str) else value
    if not ids:
        raise ValueError("an entity needs at least one chain id")
    for chain_id in ids:
        if not CHAIN_ID.fullmatch(chain_id):
            raise ValueError(
                f"invalid chain id {chain_id!r}: expected one to five letters or digits"
            )
    return value


ChainId = Annotated[
    str | list[str],
    AfterValidator(_check_chain_id),
    Field(description="Unique chain identifier(s); a list for identical chains."),
]


class _Entity(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Protein(_Entity):
    """Protein chain to fold."""

    id: ChainId
    sequence: str


class ProteinWithMSA(Protein):
    """Protein chain to fold, with an optional MSA."""

    msa_id: UUID | None = None


class DNA(_Entity):
    """DNA chain to fold."""

    id: ChainId
    sequence: str


class RNA(_Entity):
    """RNA chain to fold, with an optional MSA."""

    id: ChainId
    sequence: str
    msa_id: UUID | None = None


class Ligand(_Entity):
    """Small-molecule ligand, given as exactly one of SMILES or CCD code."""

    id: ChainId
    smiles: str | None = None
    ccd: str | None = None

    @field_validator("smiles", "ccd", mode="before")
    @classmethod
    def _empty_string_is_absent(cls, value: Any) -> Any:
        return None if value == "" else value

    @model_validator(mode="after")
    def _one_of_smiles_or_ccd(self) -> "Ligand":
        if (self.smiles is None) == (self.ccd is None):
            raise ValueError("a ligand needs exactly one of smiles or ccd")
        return self


Entity = Protein | DNA | RNA | Ligand


class _Chain(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _one_entity(self) -> "_Chain":
        named = [name for name in type(self).model_fields if getattr(self, name)]
        if len(named) != 1:
            raise ValueError("a chain names exactly one entity")
        return self

    @property
    def entity(self) -> Entity:
        """The one entity this chain names."""
        for name in type(self).model_fields:
            if (entity := getattr(self, name)) is not None:
                return entity
        raise AssertionError("validated chain names an entity")


class MsaChain(_Chain):
    """A chain: protein (with MSA), ligand, DNA, or RNA."""

    protein: ProteinWithMSA | None = None
    ligand: Ligand | None = None
    dna: DNA | None = None
    rna: RNA | None = None


class NoMsaChain(_Chain):
    """A chain for single-sequence models: protein (no MSA), ligand, DNA, or RNA."""

    protein: Protein | None = None
    ligand: Ligand | None = None
    dna: DNA | None = None
    rna: RNA | None = None


class ProteinChain(_Chain):
    """A protein-only chain (no MSA)."""

    protein: Protein | None = None


class MsaProteinChain(_Chain):
    """A protein-only chain, with an optional MSA."""

    protein: ProteinWithMSA | None = None


class MsaProteinLigandChain(_Chain):
    """A protein (with optional MSA) or ligand chain."""

    protein: ProteinWithMSA | None = None
    ligand: Ligand | None = None


def complex_chain_ids(chains: Sequence[_Chain]) -> list[str]:
    """Return one id per chain of a complex, in request order with copies expanded.

    A complex's fold outputs name their chains with it.

    Parameters
    ----------
    chains : sequence of chain
        The validated chains of one complex.

    Returns
    -------
    list of str
        The ``chain_ids`` of the complex's fold outputs.

    Raises
    ------
    ValueError
        If an id repeats within the complex.
    """
    ids: list[str] = []
    for chain in chains:
        entity_ids = chain.entity.id
        for chain_id in [entity_ids] if isinstance(entity_ids, str) else entity_ids:
            if chain_id in ids:
                raise ValueError(f"duplicate chain id {chain_id!r}")
            ids.append(chain_id)
    return ids
