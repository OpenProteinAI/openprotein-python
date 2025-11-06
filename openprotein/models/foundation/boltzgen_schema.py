"""Pydantic v2 schema for BoltzGen design specification."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# Entity Definitions
# ============================================================================


class ProteinEntity(BaseModel):
    """
    Protein entity specification.

    Attributes
    ----------
    id : str or list[str]
        Chain identifier(s) for the protein.
    sequence : str
        Protein sequence. Can include:
        - Amino acid letters (A-Z)
        - Design residues (numbers, e.g., "10" for 10 design residues)
        - Ranges (e.g., "15..20" for random number between 15-20)
        - Mixed patterns (e.g., "3..5C6C3" for variable design + fixed residues)
    secondary_structure : str | None
        Secondary structure specification. Defaults to None.
    binding_types : str | dict | None
        Binding type specification. Can be:
        - String with characters: 'u' (unspecified), 'B' (binding), 'N' (not binding)
        - Dict with 'binding' and/or 'not_binding' keys
    cyclic : bool
        Whether the protein is cyclic. Defaults to False.
    """

    id: str | list[str]
    sequence: str
    secondary_structure: str | None = None
    binding_types: str | dict | None = None
    cyclic: bool = False


class LigandEntity(BaseModel):
    """
    Ligand entity specification.

    Attributes
    ----------
    id : str or list[str]
        Chain identifier(s) for the ligand.
    ccd : str | None
        Chemical Component Dictionary identifier.
    smiles : str | None
        SMILES string representation of the ligand.
    binding_types : str | dict | None
        Binding type specification.
    """

    id: str | list[str]
    ccd: str | None = None
    smiles: str | None = None
    binding_types: str | dict | None = None

    @model_validator(mode="after")
    def check_ccd_or_smiles(self):
        """Ensure either ccd or smiles is provided."""
        if self.ccd is None and self.smiles is None:
            raise ValueError("Either 'ccd' or 'smiles' must be provided for ligand")
        return self


class ChainInclude(BaseModel):
    """
    Chain inclusion specification.

    Attributes
    ----------
    id : str
        Chain identifier.
    res_index : str | None
        Residue index range (e.g., "10..16", "..5", "20..").
    """

    id: str
    res_index: str | None = None


class ChainIncludeProximity(BaseModel):
    """
    Proximity-based chain inclusion.

    Attributes
    ----------
    id : str
        Chain identifier.
    res_index : str
        Residue index range.
    radius : float
        Radius in angstroms for proximity inclusion.
    """

    id: str
    res_index: str
    radius: float


class ChainBindingType(BaseModel):
    """
    Binding type specification for a chain.

    Attributes
    ----------
    id : str
        Chain identifier.
    binding : str | None
        Residue indices that are binding (e.g., "5..7,13").
    not_binding : str | None
        Residue indices that are not binding (e.g., "9..11" or "all").
    """

    id: str
    binding: str | None = None
    not_binding: str | None = None


class StructureGroup(BaseModel):
    """
    Structure group for visibility control.

    Attributes
    ----------
    visibility : int
        Visibility level (0, 1, 2, etc.).
    id : str
        Chain identifier or "all".
    res_index : str | None
        Residue index range.
    """

    visibility: int
    id: str
    res_index: str | None = None


class ChainDesign(BaseModel):
    """
    Design specification for a chain.

    Attributes
    ----------
    id : str
        Chain identifier.
    res_index : str
        Residue indices to design (e.g., "..4,20..27").
    """

    id: str
    res_index: str


class ChainSecondaryStructure(BaseModel):
    """
    Secondary structure specification for a chain.

    Attributes
    ----------
    id : str
        Chain identifier.
    loop : str | None
        Residue indices for loop regions.
    helix : str | None
        Residue indices for helix regions.
    sheet : str | None
        Residue indices for sheet regions.
    """

    id: str
    loop: str | None = None
    helix: str | None = None
    sheet: str | None = None


class DesignInsertion(BaseModel):
    """
    Design insertion specification.

    Attributes
    ----------
    id : str
        Chain identifier.
    res_index : int
        Residue index where insertion occurs (1-based).
    num_residues : str | int
        Number of residues to insert. Can be a range (e.g., "2..9") or fixed number.
    secondary_structure : Literal["UNSPECIFIED", "LOOP", "HELIX", "SHEET"]
        Secondary structure type for inserted residues.
    """

    id: str
    res_index: int
    num_residues: str | int
    secondary_structure: Literal["UNSPECIFIED", "LOOP", "HELIX", "SHEET"] = (
        "UNSPECIFIED"
    )


class FileEntity(BaseModel):
    """
    File-based entity specification (e.g., PDB/CIF files).

    Note
    ----
    When using the `generate()` method, the `path` field is overwritten by the
    `structure_file` argument. The OpenProtein platform backend currently only
    accepts structure files via the `structure_file` parameter, not as paths
    in the design spec. The `path` field is included here for compatibility with
    the BoltzGen YAML format, but will be replaced when submitting to the API.

    Attributes
    ----------
    path : str
        Path to the structure file. This is a placeholder that will be overwritten
        by the `structure_file` argument when calling `generate()`. The actual
        structure content must be provided via the `structure_file` parameter.
    fuse : str | None
        Chain ID to fuse with.
    include : str | list[dict]
        Chains or regions to include. Can be "all" or list of chain specifications.
    exclude : list[dict] | None
        Chains or regions to exclude.
    include_proximity : list[dict] | None
        Proximity-based inclusion specifications.
    binding_types : list[dict] | None
        Binding type specifications for chains.
    structure_groups : list[dict] | None
        Structure group specifications.
    design : list[dict] | None
        Design specifications for chains.
    secondary_structure : list[dict] | None
        Secondary structure specifications for chains.
    design_insertions : list[dict] | None
        Design insertion specifications.
    """

    path: str
    fuse: str | None = None
    include: str | list[dict] | None = None
    exclude: list[dict] | None = None
    include_proximity: list[dict] | None = None
    binding_types: list[dict] | None = None
    structure_groups: list[dict] | None = None
    design: list[dict] | None = None
    secondary_structure: list[dict] | None = None
    design_insertions: list[dict] | None = None


class Entity(BaseModel):
    """
    Entity wrapper for different entity types.

    Attributes
    ----------
    protein : ProteinEntity | None
        Protein entity specification.
    ligand : LigandEntity | None
        Ligand entity specification.
    file : FileEntity | None
        File-based entity specification.
    """

    protein: ProteinEntity | None = None
    ligand: LigandEntity | None = None
    file: FileEntity | None = None

    @model_validator(mode="after")
    def check_exactly_one_entity(self):
        """Ensure exactly one entity type is specified."""
        entities = [self.protein, self.ligand, self.file]
        if sum(x is not None for x in entities) != 1:
            raise ValueError(
                "Exactly one of 'protein', 'ligand', or 'file' must be specified"
            )
        return self


# ============================================================================
# Constraint Definitions
# ============================================================================


class BondConstraint(BaseModel):
    """
    Covalent bond constraint between two atoms.

    Attributes
    ----------
    atom1 : list[str | int]
        First atom specification: [CHAIN_ID, RES_IDX, ATOM_NAME].
    atom2 : list[str | int]
        Second atom specification: [CHAIN_ID, RES_IDX, ATOM_NAME].
    """

    atom1: list[str | int] = Field(..., min_length=3, max_length=3)
    atom2: list[str | int] = Field(..., min_length=3, max_length=3)


class TotalLengthConstraint(BaseModel):
    """
    Total length constraint for the design.

    Attributes
    ----------
    min : int | None
        Minimum total length.
    max : int | None
        Maximum total length.
    """

    min: int | None = None
    max: int | None = None


class Constraint(BaseModel):
    """
    Constraint wrapper for different constraint types.

    Attributes
    ----------
    bond : BondConstraint | None
        Bond constraint specification.
    total_len : TotalLengthConstraint | None
        Total length constraint specification.
    """

    bond: BondConstraint | None = None
    total_len: TotalLengthConstraint | None = None

    @model_validator(mode="after")
    def check_at_least_one_constraint(self):
        """Ensure at least one constraint type is specified."""
        constraints = [self.bond, self.total_len]
        if sum(x is not None for x in constraints) == 0:
            raise ValueError("At least one constraint type must be specified")
        return self


# ============================================================================
# Top-Level Design Spec
# ============================================================================


class BoltzGenDesignSpec(BaseModel):
    """
    Complete BoltzGen design specification.

    This schema represents the full design specification for BoltzGen,
    including entities (proteins, ligands, files) and constraints.

    Attributes
    ----------
    entities : list[Entity]
        List of entities in the design.
    constraints : list[Constraint] | None
        List of constraints for the design.

    Examples
    --------
    >>> spec = BoltzGenDesignSpec(
    ...     entities=[
    ...         Entity(protein=ProteinEntity(id="A", sequence="ACDEFGHIKLMNPQRSTVWY")),
    ...         Entity(ligand=LigandEntity(id="B", ccd="ATP"))
    ...     ],
    ...     constraints=[
    ...         Constraint(bond=BondConstraint(atom1=["A", 10, "CA"], atom2=["B", 1, "O"]))
    ...     ]
    ... )
    """

    entities: list[Entity]
    constraints: list[Constraint] | None = None

    @field_validator("entities")
    @classmethod
    def check_entities_not_empty(cls, v):
        """Ensure at least one entity is provided."""
        if not v:
            raise ValueError("At least one entity must be specified")
        return v
