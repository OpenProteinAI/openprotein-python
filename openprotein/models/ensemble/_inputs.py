"""Ensemble inputs converted to request fields."""

from collections.abc import Sequence
from typing import Any, TypeVar

from openprotein.fold.common import _msa_id_of
from openprotein.molecules import Complex, Protein, Structure

_T = TypeVar("_T")


def input_list(inputs: Any, input_type: type[_T]) -> list[_T]:
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError(f"inputs must be a list of {input_type.__name__}")
    if not inputs:
        raise ValueError("inputs must contain at least one input")
    for index, value in enumerate(inputs):
        if not isinstance(value, input_type):
            raise TypeError(
                f"inputs[{index}] must be a {input_type.__name__}, "
                f"not {type(value).__name__}"
            )
    return list(inputs)


def sequence_text(value: Any, field: str) -> str:
    if isinstance(value, Protein):
        value = value.sequence
    if isinstance(value, bytes):
        value = value.decode()
    if not isinstance(value, str):
        raise TypeError(
            f"{field} must be a Protein or sequence text, not {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{field} is empty")
    if ":" in value:
        raise ValueError(f"{field} holds several chains; pass one protein chain")
    return value


def msa_id(value: Any, field: str, *, required: bool) -> str | None:
    resolved = None if value is None else _msa_id_of(value)
    if required and resolved is None:
        raise ValueError(
            f"{field} is required; pass an MSA whose query is the sequence"
        )
    return resolved


def protein_complex(sequence: str, msa: str | None, *, takes_msa: bool) -> list:
    protein: dict[str, Any] = {"id": "A", "sequence": sequence}
    if takes_msa:
        protein["msa_id"] = msa
    return [{"protein": protein}]


def _text(value: str | bytes, field: str) -> str:
    text = value.decode() if isinstance(value, bytes) else value
    if not text.strip():
        raise ValueError(f"{field} is empty")
    return text


def _parse_pdb(value: str | bytes, field: str) -> Structure:
    text = _text(value, field)
    if any(
        line.lstrip().startswith(("data_", "_atom_site.")) for line in text.splitlines()
    ):
        raise ValueError(
            f"{field} requires PDB text; parse mmCIF into a Protein, "
            "Complex or Structure before passing it"
        )
    try:
        return Structure.from_string(text, format="pdb", verbose=False)
    except ValueError as error:
        raise ValueError(
            f"{field} must contain a supported protein structure: {error}"
        ) from error


def _reference_complex(value: Any, field: str, *, single_chain: bool) -> Complex:
    if isinstance(value, Structure):
        if not len(value):
            raise ValueError(f"{field} contains no structure")
        value = value[0]
    if isinstance(value, Protein):
        value = Complex({"A": value})
    if not isinstance(value, Complex):
        raise TypeError(
            f"{field} must be structure text, Protein, Complex or Structure"
        )

    proteins = value.get_proteins()
    if not proteins:
        raise ValueError(f"{field} contains no protein chain")
    nonprotein = [
        chain_id
        for chain_id, chain in value.get_chains().items()
        if not isinstance(chain, Protein)
    ]
    if nonprotein:
        raise ValueError(
            f"{field} contains non-protein chains: {', '.join(nonprotein)}"
        )
    if single_chain and len(proteins) != 1:
        raise ValueError(f"{field} requires exactly one protein chain")
    for chain_id, protein in proteins.items():
        if protein.get_structure_mask().all():
            raise ValueError(f"{field} chain {chain_id!r} has no coordinates")
    return value


def _check_matches(reference: str, sequence: str, field: str) -> None:
    if len(reference) != len(sequence):
        raise ValueError(
            f"{field} has {len(reference)} residues and the sequence has "
            f"{len(sequence)}; they must be the same protein, residue for residue"
        )
    for position, (ours, theirs) in enumerate(zip(reference, sequence), start=1):
        if ours != theirs and "X" not in (ours, theirs):
            raise ValueError(
                f"{field} differs from the sequence at position {position}: "
                f"{ours} in the reference, {theirs} in the sequence"
            )


def flow_reference(value: Any, sequence: str, field: str) -> str:
    """``value`` as PDB text of one fully modelled chain matching ``sequence``."""
    if isinstance(value, (str, bytes)):
        value = _parse_pdb(value, field)
    complex = _reference_complex(value, field, single_chain=True)
    (protein,) = complex.get_proteins().values()
    if protein.get_structure_mask().any():
        raise ValueError(
            f"{field} contains unmodelled residues; trim them from both "
            "the reference and the sequence, for example "
            "protein[~protein.get_structure_mask()]"
        )
    _check_matches(protein.sequence.decode(), sequence, field)
    return complex.to_string(format="pdb")


def flow_references(
    references: list[Any], sequences: list[str]
) -> list[dict[str, str]] | None:
    """One reference record per input, or None when no input has a reference."""
    given = [reference is not None for reference in references]
    if not any(given):
        return None
    if not all(given):
        raise ValueError(
            "give every input a reference_structure, or none of them; "
            f"inputs[{given.index(False)}] has none"
        )
    return [
        {
            "structure_file": flow_reference(
                reference, sequence, f"inputs[{index}].reference_structure"
            )
        }
        for index, (reference, sequence) in enumerate(zip(references, sequences))
    ]
