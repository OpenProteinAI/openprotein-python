"""Test the BoltzGen design specification schema."""

import pytest
from pydantic import ValidationError

from openprotein.models.foundation.boltzgen_schema import (
    BoltzGenDesignSpec,
    BondConstraint,
    Constraint,
    DesignInsertion,
    Entity,
    FileEntity,
    LigandEntity,
    ProteinEntity,
    TotalLengthConstraint,
)


class TestProteinEntity:
    """Test ProteinEntity schema."""

    def test_protein_entity_basic(self):
        """Test basic protein entity creation."""
        protein = ProteinEntity(id="A", sequence="ACDEFGHIKLMNPQRSTVWY")
        assert protein.id == "A"
        assert protein.sequence == "ACDEFGHIKLMNPQRSTVWY"
        assert protein.cyclic is False
        assert protein.secondary_structure is None
        assert protein.binding_types is None

    def test_protein_entity_with_design_residues(self):
        """Test protein entity with design residue patterns."""
        protein = ProteinEntity(id="G", sequence="15..20AAAAAAVTTTT18PPP")
        assert protein.sequence == "15..20AAAAAAVTTTT18PPP"

    def test_protein_entity_with_cyclic(self):
        """Test cyclic protein entity."""
        protein = ProteinEntity(id="T", sequence="C10C6C3C", cyclic=True)
        assert protein.cyclic is True

    def test_protein_entity_with_binding_types_string(self):
        """Test protein entity with binding types as string."""
        protein = ProteinEntity(
            id="A", sequence="AAAAAAAAAAAAAAAAAAAAAAAA", binding_types="uuuuBBBuNNNuBuu"
        )
        assert protein.binding_types == "uuuuBBBuNNNuBuu"

    def test_protein_entity_with_binding_types_dict(self):
        """Test protein entity with binding types as dict."""
        protein = ProteinEntity(
            id="B",
            sequence="AAAAAAAAAAAAAAAAAAAAAAAA",
            binding_types={"binding": "5..7,13", "not_binding": "9..11"},
        )
        assert protein.binding_types == {"binding": "5..7,13", "not_binding": "9..11"}

    def test_protein_entity_with_list_ids(self):
        """Test protein entity with multiple chain IDs."""
        protein = ProteinEntity(id=["A", "B"], sequence="ACDEFG")
        assert protein.id == ["A", "B"]


class TestLigandEntity:
    """Test LigandEntity schema."""

    def test_ligand_entity_with_ccd(self):
        """Test ligand entity with CCD code."""
        ligand = LigandEntity(id="Q", ccd="WHL")
        assert ligand.id == "Q"
        assert ligand.ccd == "WHL"
        assert ligand.smiles is None

    def test_ligand_entity_with_smiles(self):
        """Test ligand entity with SMILES string."""
        ligand = LigandEntity(id=["E", "F"], smiles="N[C@@H](Cc1ccc(O)cc1)C(=O)O")
        assert ligand.id == ["E", "F"]
        assert ligand.smiles == "N[C@@H](Cc1ccc(O)cc1)C(=O)O"
        assert ligand.ccd is None

    def test_ligand_entity_with_binding_types(self):
        """Test ligand entity with binding types."""
        ligand = LigandEntity(
            id=["E", "F"], smiles="N[C@@H](Cc1ccc(O)cc1)C(=O)O", binding_types="B"
        )
        assert ligand.binding_types == "B"

    def test_ligand_entity_missing_ccd_and_smiles(self):
        """Test that ligand entity requires either ccd or smiles."""
        with pytest.raises(ValidationError, match="Either 'ccd' or 'smiles'"):
            LigandEntity(id="Q")


class TestFileEntity:
    """Test FileEntity schema."""

    def test_file_entity_basic(self):
        """Test basic file entity creation."""
        file_entity = FileEntity(path="7rpz.cif", include="all")
        assert file_entity.path == "7rpz.cif"
        assert file_entity.include == "all"

    def test_file_entity_with_chain_include(self):
        """Test file entity with chain inclusion."""
        file_entity = FileEntity(
            path="7rpz.cif", include=[{"chain": {"id": "A"}}, {"chain": {"id": "B"}}]
        )
        assert isinstance(file_entity.include, list)
        assert len(file_entity.include) == 2

    def test_file_entity_with_fuse(self):
        """Test file entity with fuse parameter."""
        file_entity = FileEntity(
            path="7rpz.cif",
            fuse="A",
            include=[{"chain": {"id": "A", "res_index": "..5"}}],
        )
        assert file_entity.fuse == "A"

    def test_file_entity_with_all_options(self):
        """Test file entity with multiple options."""
        file_entity = FileEntity(
            path="7rpz.cif",
            include=[{"chain": {"id": "A"}}],
            exclude=[{"chain": {"id": "A", "res_index": "..5"}}],
            binding_types=[{"chain": {"id": "A", "binding": "5..7,13"}}],
            structure_groups=[
                {"group": {"visibility": 1, "id": "A", "res_index": "10..16"}}
            ],
            design=[{"chain": {"id": "A", "res_index": "..4,20..27"}}],
            secondary_structure=[{"chain": {"id": "A", "loop": "1", "helix": "2..3"}}],
            design_insertions=[
                {
                    "insertion": {
                        "id": "A",
                        "res_index": 20,
                        "num_residues": "2..9",
                        "secondary_structure": "HELIX",
                    }
                }
            ],
        )
        assert file_entity.path == "7rpz.cif"
        assert file_entity.design is not None
        assert file_entity.secondary_structure is not None


class TestEntity:
    """Test Entity wrapper schema."""

    def test_entity_with_protein(self):
        """Test entity with protein."""
        entity = Entity(protein=ProteinEntity(id="A", sequence="ACDEFG"))
        assert entity.protein is not None
        assert entity.ligand is None
        assert entity.file is None

    def test_entity_with_ligand(self):
        """Test entity with ligand."""
        entity = Entity(ligand=LigandEntity(id="B", ccd="ATP"))
        assert entity.protein is None
        assert entity.ligand is not None
        assert entity.file is None

    def test_entity_with_file(self):
        """Test entity with file."""
        entity = Entity(file=FileEntity(path="test.cif", include="all"))
        assert entity.protein is None
        assert entity.ligand is None
        assert entity.file is not None

    def test_entity_requires_exactly_one(self):
        """Test that entity requires exactly one entity type."""
        with pytest.raises(ValidationError, match="Exactly one"):
            Entity()

        with pytest.raises(ValidationError, match="Exactly one"):
            Entity(
                protein=ProteinEntity(id="A", sequence="ACDEFG"),
                ligand=LigandEntity(id="B", ccd="ATP"),
            )


class TestConstraints:
    """Test constraint schemas."""

    def test_bond_constraint(self):
        """Test bond constraint creation."""
        bond = BondConstraint(atom1=["R", 4, "SG"], atom2=["Q", 1, "CK"])
        assert bond.atom1 == ["R", 4, "SG"]
        assert bond.atom2 == ["Q", 1, "CK"]

    def test_bond_constraint_invalid_length(self):
        """Test that bond constraint requires exactly 3 elements."""
        with pytest.raises(ValidationError):
            BondConstraint(atom1=["R", 4], atom2=["Q", 1, "CK"])

    def test_total_length_constraint(self):
        """Test total length constraint creation."""
        total_len = TotalLengthConstraint(min=10, max=20)
        assert total_len.min == 10
        assert total_len.max == 20

    def test_constraint_with_bond(self):
        """Test constraint wrapper with bond."""
        constraint = Constraint(
            bond=BondConstraint(atom1=["R", 4, "SG"], atom2=["Q", 1, "CK"])
        )
        assert constraint.bond is not None
        assert constraint.total_len is None

    def test_constraint_with_total_len(self):
        """Test constraint wrapper with total length."""
        constraint = Constraint(total_len=TotalLengthConstraint(min=10, max=20))
        assert constraint.bond is None
        assert constraint.total_len is not None

    def test_constraint_requires_at_least_one(self):
        """Test that constraint requires at least one constraint type."""
        with pytest.raises(ValidationError, match="At least one constraint"):
            Constraint()


class TestBoltzGenDesignSpec:
    """Test complete BoltzGen design specification."""

    def test_design_spec_minimal(self):
        """Test minimal design spec with one entity."""
        spec = BoltzGenDesignSpec(
            entities=[Entity(protein=ProteinEntity(id="A", sequence="ACDEFG"))]
        )
        assert len(spec.entities) == 1
        assert spec.constraints is None

    def test_design_spec_with_constraints(self):
        """Test design spec with entities and constraints."""
        spec = BoltzGenDesignSpec(
            entities=[
                Entity(protein=ProteinEntity(id="A", sequence="ACDEFG")),
                Entity(ligand=LigandEntity(id="B", ccd="ATP")),
            ],
            constraints=[
                Constraint(
                    bond=BondConstraint(atom1=["A", 10, "CA"], atom2=["B", 1, "O"])
                )
            ],
        )
        assert len(spec.entities) == 2
        assert isinstance(spec.constraints, list)
        assert len(spec.constraints) == 1

    def test_design_spec_empty_entities(self):
        """Test that design spec requires at least one entity."""
        with pytest.raises(ValidationError, match="At least one entity"):
            BoltzGenDesignSpec(entities=[])

    def test_design_spec_from_dict(self):
        """Test creating design spec from dict."""
        data = {
            "entities": [
                {"protein": {"id": "A", "sequence": "ACDEFG"}},
                {"ligand": {"id": "B", "ccd": "ATP"}},
            ],
            "constraints": [
                {"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 1, "O"]}}
            ],
        }
        spec = BoltzGenDesignSpec.model_validate(data)
        assert len(spec.entities) == 2
        assert isinstance(spec.constraints, list)
        assert len(spec.constraints) == 1

    def test_design_spec_round_trip(self):
        """Test round-trip conversion (dict -> model -> dict)."""
        original_data = {
            "entities": [
                {"protein": {"id": "A", "sequence": "ACDEFG"}},
                {"ligand": {"id": "B", "ccd": "ATP"}},
            ],
            "constraints": [
                {"bond": {"atom1": ["A", 10, "CA"], "atom2": ["B", 1, "O"]}}
            ],
        }
        spec = BoltzGenDesignSpec.model_validate(original_data)
        spec_dict = spec.model_dump(exclude_none=True)
        spec2 = BoltzGenDesignSpec.model_validate(spec_dict)
        assert len(spec2.entities) == len(spec.entities)
        assert isinstance(spec2.constraints, list)
        assert isinstance(spec.constraints, list)
        assert len(spec2.constraints) == len(spec.constraints)

    def test_design_spec_complex_example(self):
        """Test complex design spec with multiple entity types."""
        data = {
            "entities": [
                {"protein": {"id": "G", "sequence": "15..20AAAAAAVTTTT18PPP"}},
                {"protein": {"id": "R", "sequence": "3..5C6C3"}},
                {"ligand": {"id": "Q", "ccd": "WHL"}},
                {"protein": {"id": "S", "sequence": "10C6C3"}},
                {"protein": {"id": "T", "sequence": "C10C6C3C", "cyclic": True}},
            ],
            "constraints": [
                {"bond": {"atom1": ["R", 4, "SG"], "atom2": ["Q", 1, "CK"]}},
                {"bond": {"atom1": ["R", 11, "SG"], "atom2": ["Q", 1, "CH"]}},
                {"bond": {"atom1": ["S", 11, "SG"], "atom2": ["S", 18, "SG"]}},
                {"bond": {"atom1": ["T", 12, "SG"], "atom2": ["T", 19, "SG"]}},
                {"total_len": {"min": 10, "max": 20}},
            ],
        }
        spec = BoltzGenDesignSpec.model_validate(data)
        assert len(spec.entities) == 5
        assert isinstance(spec.constraints, list)
        assert len(spec.constraints) == 5
        assert spec.entities[4].protein is not None
        assert spec.entities[4].protein.cyclic is True
