"""Structure-prediction handle, shaped by the model's card and params."""

import copy
import warnings
from collections.abc import Sequence
from typing import Any

from openprotein.align import MSAFuture
from openprotein.fold.common import (
    _msa_id_of,
    msa_future_to_complex,
    normalize_inputs,
    normalize_templates,
    resolve_templates,
    serialize_input,
)
from openprotein.jobs import futures
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein

from ..model import Model

__all__ = ["StructurePredictionModel"]


_CHAIN_TYPES: dict[str, type] = {
    "protein": Protein,
    "dna": DNA,
    "rna": RNA,
    "ligand": Ligand,
}


def _is_serialized(complexes: Any) -> bool:
    return (
        isinstance(complexes, (list, tuple))
        and len(complexes) > 0
        and all(isinstance(c, (list, tuple)) for c in complexes)
    )


def _without(complex: Complex, drops: tuple[type, ...], model_id: str) -> Complex:
    kept = {k: c for k, c in complex.get_chains().items() if not isinstance(c, drops)}
    if len(kept) == len(complex.get_chains()):
        return complex
    warnings.warn(
        f"{model_id} does not support "
        f"{'/'.join(t.__name__ for t in drops)} input; those chains are left out."
    )
    trimmed = copy.copy(complex)
    trimmed._chains = kept
    return trimmed


def _declares(schema: Any, name: str) -> bool:
    """Whether any object in ``schema`` declares the property ``name``."""
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict) and name in properties:
            return True
        return any(_declares(value, name) for value in schema.values())
    if isinstance(schema, list):
        return any(_declares(value, name) for value in schema)
    return False


def _needs_msa(schema: Any) -> bool:
    """Whether a protein in ``schema`` requires a non-null ``msa_id``."""
    if isinstance(schema, dict):
        properties = schema.get("properties")
        msa_id = properties.get("msa_id") if isinstance(properties, dict) else None
        if isinstance(msa_id, dict) and "msa_id" in schema.get("required", []):
            arms = msa_id.get("anyOf", [msa_id])
            if not any(arm.get("type") == "null" for arm in arms):
                return True
        return any(_needs_msa(value) for value in schema.values())
    if isinstance(schema, list):
        return any(_needs_msa(value) for value in schema)
    return False


def _has_msa(protein: Protein) -> bool:
    return protein.msa is not None and _msa_id_of(protein.msa) is not None


class StructurePredictionModel(Model):
    """A structure-prediction model.

    ``fold`` takes a sequence, ``Protein``, ``Complex``, a list of these, or
    an ``MSAFuture`` as one multichain complex, and sends them as
    ``complexes``. Chains outside the card's ``input_types`` are left out,
    and a card that isn't ``multichain`` folds one protein per complex.
    MSAs and templates, given as ``templates`` or attached to the inputs,
    are sent where the model's params take them.
    """

    def fold(
        self,
        complexes: Any,
        templates: Sequence[Protein | Complex | Any] | None = None,
        **params: Any,
    ) -> futures.StructurePredictionFuture:
        """Predict structures of complexes (``batch/structures``)."""
        if not _is_serialized(complexes) and not isinstance(complexes, MSAFuture):
            entities = (
                complexes if isinstance(complexes, (list, tuple)) else [complexes]
            )
            for entity in entities:
                if not isinstance(entity, (Complex, Protein, str, bytes)):
                    raise TypeError(
                        "complexes must be a str, bytes, Protein, Complex, a list of "
                        f"these or an MSAFuture, not {type(entity).__name__}"
                    )
            if not entities:
                raise ValueError("Expected at least one complex to fold")
        schema = self._params_schema("batch/structures")
        takes_templates = "templates" in schema.get("properties", {})
        if templates is not None and not takes_templates:
            raise ValueError(f"model {self.id!r} takes no templates")

        if _is_serialized(complexes):
            if templates is not None:
                params["templates"] = templates
            return super().fold(list(complexes), **params)

        if isinstance(complexes, MSAFuture):
            normalized = [msa_future_to_complex(self.session, complexes)]
        else:
            normalized = normalize_inputs(complexes)
        if not normalized:
            raise ValueError("Expected at least one complex to fold")

        card = self.metadata
        if card.multichain is False:
            for complex in normalized:
                if len(complex.get_proteins()) > 1:
                    raise ValueError(f"{self.id} only supports monomers")
                if len(complex.get_chains()) != len(complex.get_proteins()):
                    raise ValueError(f"{self.id} only supports proteins")
        if card.input_types is not None:
            drops = tuple(
                chain_type
                for name, chain_type in _CHAIN_TYPES.items()
                if name not in card.input_types
            )
            if drops:
                normalized = [_without(c, drops, self.id) for c in normalized]
                for index, complex in enumerate(normalized):
                    if not complex.get_chains():
                        raise ValueError(
                            f"complex {index} has no chain {self.id} can fold"
                        )
        takes_msa = _declares(schema, "msa_id")
        if _needs_msa(schema):
            without_msa = [
                chain_id
                for complex in normalized
                for chain_id, protein in complex.get_proteins().items()
                if not _has_msa(protein)
            ]
            if without_msa:
                raise ValueError(
                    f"{self.id} needs an MSA for every protein; chain "
                    f"{without_msa[0]!r} has none. Attach one with "
                    "`protein.msa = msa`."
                )
        if not takes_msa:
            with_msa = [
                chain_id
                for complex in normalized
                for chain_id, protein in complex.get_proteins().items()
                if _has_msa(protein)
            ]
            if with_msa:
                raise ValueError(
                    f"{self.id} is a single-sequence model; chain "
                    f"{with_msa[0]!r} has an MSA attached. Set "
                    "`protein.msa = Protein.single_sequence_mode`."
                )

        normalized_templates = (
            normalize_templates(self.session, complexes, templates)
            if takes_templates
            else []
        )
        serialized = serialize_input(self.session, normalized, takes_msa)
        if normalized_templates:
            params["templates"] = resolve_templates(self.session, normalized_templates)
        return super().fold(serialized, **params)
