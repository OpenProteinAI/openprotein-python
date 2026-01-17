import gemmi

from openprotein.utils.cif import (
    StructureCIFBlock,
    parse_full_sequences_from_entity_poly_seq,
)


def test_parse_full_sequences_from_entity_poly_seq_includes_microheterogeneities():
    block = gemmi.cif.read_file("tests/data/1PFE.cif").sole_block()
    structure_block = StructureCIFBlock(filestring=block.as_string(), format="cif")
    entities = [
        e
        for e in structure_block.structure.entities
        if e.entity_type == gemmi.EntityType.Polymer
    ]
    assert all(len(e.full_sequence) > 0 for e in entities)
    full_sequences_from_entity_poly_seq = parse_full_sequences_from_entity_poly_seq(
        block=block, entities=entities
    )
    assert full_sequences_from_entity_poly_seq == structure_block.full_sequences
