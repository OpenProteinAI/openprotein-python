"""Additional chains that can be used with OpenProtein."""

from dataclasses import dataclass, replace
from typing import ClassVar, Protocol

import gemmi

import openprotein.utils.cif as _cif_utils

from .protein import (
    _extract_full_sequence_from_residues,
    _extract_one_letter_from_full_sequence,
)


class _BasicSerde(Protocol):
    _GEMMI_ENTITY_TYPE: ClassVar[gemmi.EntityType]
    _GEMMI_POLYMER_TYPE: ClassVar[gemmi.PolymerType]
    _structure_block: _cif_utils.StructureCIFBlock | None

    def _make_structure(
        self,
        structure: gemmi.Structure | None = None,
        model_idx: int = 0,
        chain_id: str = "A",
        entity_name: str = "1",
    ) -> gemmi.Structure:
        assert (
            self._structure_block is not None
        ), "only chains constructed directly from a structure file can be serialized for now"
        assert (
            self._structure_block.structure.input_format == gemmi.CoorFormat.Mmcif
        ), "only chains that were deserialized from cif can be serialized for now"
        # Create an empty structure and add a model with a default chain.
        if structure is None:
            structure = gemmi.Structure()
        # Get existing model or create new one
        if len(structure) > 0:
            model = structure[model_idx]
        else:
            model = structure.add_model(gemmi.Model(str(model_idx)))  # type: ignore - gemmi 0.6 needs str
        # Get existing chain
        subchain = self._structure_block.structure[model_idx].get_subchain(chain_id)
        assert len(subchain) > 0
        # Create entity
        if self._GEMMI_ENTITY_TYPE == gemmi.EntityType.Polymer:
            entity = gemmi.Entity(entity_name)
            entity.name = entity_name
            entity.subchains = [chain_id]
            entity.entity_type = self._GEMMI_ENTITY_TYPE
            entity.polymer_type = self._GEMMI_POLYMER_TYPE
            entity.full_sequence = self._structure_block.structure.get_entity_of(
                subchain
            ).full_sequence
            structure.entities.append(entity)
        else:
            matching_entities = [
                entity for entity in structure.entities if chain_id in entity.subchains
            ]
            if len(matching_entities) == 0:
                original_entity = self._structure_block.structure.get_entity_of(
                    subchain
                )
                entity = gemmi.Entity(entity_name)
                entity.name = entity_name
                entity.subchains = original_entity.subchains
                entity.entity_type = original_entity.entity_type
                entity.polymer_type = original_entity.polymer_type
                entity.full_sequence = original_entity.full_sequence
                structure.entities.append(entity)
            elif len(matching_entities) == 1:
                pass
            else:
                raise ValueError("more matching entities found than expected")
        # Create chain
        chain = model.add_chain(gemmi.Chain(chain_id))
        chain.append_residues(list(subchain.first_conformer()))
        return structure

    def _append_loop_data(
        self, chain_id: str, sequence_loop: gemmi.cif.Loop, atom_loop: gemmi.cif.Loop
    ):
        pass


@dataclass(frozen=True, eq=False)
class DNA(_BasicSerde):
    """
    Represents a DNA sequence.

    Attributes:
        sequence (str): The nucleotide sequence of the DNA.
    """

    sequence: str
    cyclic: bool = False

    _GEMMI_ENTITY_TYPE: ClassVar[gemmi.EntityType] = gemmi.EntityType.Polymer
    _GEMMI_POLYMER_TYPE: ClassVar[gemmi.PolymerType] = gemmi.PolymerType.Dna
    _structure_block: _cif_utils.StructureCIFBlock | None = None

    def __post_init__(self):
        if not all(nt in set("ACGT") for nt in self.sequence.upper()):
            raise ValueError("Sequence contains invalid DNA nucleotides.")

    def __len__(self):
        return len(self.sequence)

    def copy(self) -> "DNA":
        return replace(self)

    @staticmethod
    def _from_structure_block(
        structure_block: _cif_utils.StructureCIFBlock, chain_id: str, model_idx: int
    ) -> "DNA":
        assert structure_block.structure.input_format == gemmi.CoorFormat.Mmcif
        structure = structure_block.structure
        model = structure[model_idx]
        polymer = model.get_subchain(chain_id)
        assert len(polymer) > 0
        # extract sequence
        entity = structure.get_entity_of(polymer)
        residues = list(polymer.first_conformer())
        # TODO: consider utilizing polymer.make_one_letter_sequence() here or elsewhere
        del polymer
        if len(entity.full_sequence) > 0:
            chain_seq = entity.full_sequence
        elif entity.name in structure_block.full_sequences:
            chain_seq, _ = structure_block.full_sequences[entity.name]
        else:
            chain_seq, _ = _extract_full_sequence_from_residues(residues=residues)
        chain_seq = _extract_one_letter_from_full_sequence(full_sequence=chain_seq)
        return DNA(sequence="".join(chain_seq), _structure_block=structure_block)


@dataclass(frozen=True, eq=False)
class RNA(_BasicSerde):
    """
    Represents an RNA sequence.

    Attributes:
        sequence (str): The nucleotide sequence of the RNA.
    """

    sequence: str
    cyclic: bool = False

    _GEMMI_ENTITY_TYPE: ClassVar[gemmi.EntityType] = gemmi.EntityType.Polymer
    _GEMMI_POLYMER_TYPE: ClassVar[gemmi.PolymerType] = gemmi.PolymerType.Rna
    _structure_block: _cif_utils.StructureCIFBlock | None = None

    def __post_init__(self):
        if not all(nt in set("ACGU") for nt in self.sequence.upper()):
            raise ValueError("Sequence contains invalid RNA nucleotides.")

    def __len__(self):
        return len(self.sequence)

    def copy(self) -> "RNA":
        return replace(self)

    @staticmethod
    def _from_structure_block(
        structure_block: _cif_utils.StructureCIFBlock, chain_id: str, model_idx: int
    ) -> "RNA":
        assert structure_block.structure.input_format == gemmi.CoorFormat.Mmcif
        structure = structure_block.structure
        model = structure[model_idx]
        polymer = model.get_subchain(chain_id)
        assert len(polymer) > 0
        # extract sequence
        entity = structure.get_entity_of(polymer)
        residues = list(polymer.first_conformer())
        # TODO: consider utilizing polymer.make_one_letter_sequence() here or elsewhere
        del polymer
        if len(entity.full_sequence) > 0:
            chain_seq = entity.full_sequence
        elif entity.name in structure_block.full_sequences:
            chain_seq, _ = structure_block.full_sequences[entity.name]
        else:
            chain_seq, _ = _extract_full_sequence_from_residues(residues=residues)
        chain_seq = _extract_one_letter_from_full_sequence(full_sequence=chain_seq)
        return RNA(sequence="".join(chain_seq), _structure_block=structure_block)


@dataclass(frozen=True, eq=False)
class Ligand(_BasicSerde):
    """
    Represents a ligand with optional Chemical Component Dictionary (CCD) identifier and SMILES string.

    Requires either a CCD identifier or SMILES string.

    Attributes:
        ccd (str | None): The CCD identifier for the ligand.
        smiles (str | None): The SMILES representation of the ligand.
    """

    ccd: str | None = None
    smiles: str | None = None

    _GEMMI_ENTITY_TYPE: ClassVar[gemmi.EntityType] = gemmi.EntityType.NonPolymer
    _GEMMI_POLYMER_TYPE: ClassVar[gemmi.PolymerType] = gemmi.PolymerType.Unknown
    _structure_block: _cif_utils.StructureCIFBlock | None = None

    def __post_init__(self):
        if (self.ccd is None and self.smiles is None) or (
            self.ccd is not None and self.smiles is not None
        ):
            raise ValueError("Exactly one of 'ccd' or 'smiles' must be provided.")

    def copy(self) -> "Ligand":
        return replace(self)

    @staticmethod
    def _from_structure_block(
        structure_block: _cif_utils.StructureCIFBlock, chain_id: str, model_idx: int
    ) -> "Ligand":
        assert structure_block.structure.input_format == gemmi.CoorFormat.Mmcif
        subchain = structure_block.structure[model_idx].get_subchain(chain_id)
        residues = list(subchain.first_conformer())
        assert len(residues) == 1
        return Ligand(ccd=residues[0].name, _structure_block=structure_block)
