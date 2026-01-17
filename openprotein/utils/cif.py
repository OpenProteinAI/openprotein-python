from collections.abc import Sequence
from functools import cached_property
from types import MappingProxyType
from typing import Literal, Mapping, cast

import gemmi
import numpy as np


class StructureCIFBlock:
    """
    Represents a gemmi.Structure with additional data stored in a CIF block.
    Users should not mutate any outputs of this class.
    """

    def __init__(self, filestring: str | bytes, format: Literal["pdb", "cif"]):
        if format == "pdb":
            self.structure = gemmi.read_pdb_string(filestring)
            self.block = gemmi.cif.Block(self.structure.name)
        elif format == "cif":
            self.block = gemmi.cif.read_string(filestring).sole_block()
            self.structure = gemmi.make_structure_from_block(self.block)
        else:
            raise ValueError(f"Unknown {format=}")
        self.structure.setup_entities()
        self.structure.assign_label_seq_id()

    @cached_property
    def full_sequences(self) -> Mapping[str, tuple[Sequence[str], int]]:
        """
        Returns a mapping from entity name to a tuple of:
            (full_sequence, start_residue_number)
        where full_sequence is a list of residue names and start_residue_number
        is the 1-indexed residue number corresponding to full_sequence[0].

        Only polymer entities are considered for now. Entities for which a full sequence
        cannot be determined are omitted.

        Sequence determination follows this fallback order:
            1. If entity.full_sequence is non-empty, it is used directly, with a
               start_residue_number of 1.
            2. Otherwise, the sequence is reconstructed from the `_entity_poly_seq`
               loop, using the minimum and maximum residue numbers present for that
               entity. Missing residue numbers within this range are filled with "UNK",
               and the start_residue_number is set to the minimum residue number
               observed.
        When reconstructing from `_entity_poly_seq`, no validation is performed to
        ensure consistency with the entity's subchains (at least for now).
        """
        full_sequences: dict[str, tuple[Sequence[str], int]] = {}
        entities_to_impute: list[gemmi.Entity] = []
        for entity in self.structure.entities:
            if entity.entity_type != gemmi.EntityType.Polymer:
                continue
            if len(entity.full_sequence) > 0:
                full_sequences[entity.name] = (tuple(entity.full_sequence), 1)
            else:
                entities_to_impute.append(entity)
        if len(entities_to_impute) == 0:
            return MappingProxyType(full_sequences)
        full_sequences |= parse_full_sequences_from_entity_poly_seq(
            block=self.block, entities=entities_to_impute
        )
        return MappingProxyType(full_sequences)


def parse_full_sequences_from_entity_poly_seq(
    block: gemmi.cif.Block, entities: list[gemmi.Entity]
) -> Mapping[str, tuple[Sequence[str], int]]:
    full_sequences: Mapping[str, tuple[Sequence[str], int]] = {}
    table = block.find("_entity_poly_seq.", ["entity_id", "num", "mon_id"])
    if len(table) == 0:
        return MappingProxyType(full_sequences)
    entity_ids = np.array(table.find_column("entity_id"))
    nums = np.fromiter(table.find_column("num"), dtype=int, count=len(table))
    mon_ids = np.array(table.find_column("mon_id"))
    for entity in entities:
        mask = entity_ids == entity.name
        if not mask.any():
            continue
        entity_nums, entity_mon_ids = nums[mask], mon_ids[mask]
        min_num = cast(int, entity_nums.min())
        max_num = cast(int, entity_nums.max())
        full_sequence = [None] * (max_num - min_num + 1)
        for num, mon_id in zip(entity_nums, entity_mon_ids, strict=True):
            if (current_mon_id := full_sequence[num - min_num]) is None:
                full_sequence[num - min_num] = mon_id
            else:
                full_sequence[num - min_num] = current_mon_id + f",{mon_id}"
        full_sequences[entity.name] = (
            tuple(x if x is not None else "UNK" for x in full_sequence),
            min_num,
        )
    return MappingProxyType(full_sequences)


def init_loops(
    block: gemmi.cif.Block,
    version: str = "1",
) -> tuple[gemmi.cif.Loop, gemmi.cif.Loop]:
    """
    Initialize CIF loops for OpenProtein data.

    Args:
        block: CIF block to initialize
        version: OpenProtein format version (default "1")

    Returns:
        Tuple of (sequence_loop, atom_loop)
    """
    block.set_pair("_openprotein_version", version)
    sequence_loop = block.init_loop(
        "_openprotein_sequence.", ["label_asym_id", "cyclic", "msa_id"]
    )
    atom_loop = block.init_loop(
        "_openprotein_atom.",
        [
            "label_asym_id",
            "label_seq_id",
            "label_atom_id",
            "group",
            "binding",
        ],
    )
    return sequence_loop, atom_loop
