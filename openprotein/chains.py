"""Additional chains that can be used with OpenProtein."""

from dataclasses import dataclass


@dataclass
class DNA:
    """
    Represents a DNA sequence.

    Attributes:
        sequence (str): The nucleotide sequence of the DNA.
    """

    sequence: str
    chain_id: str | list[str] | None = None
    cyclic: bool = False

    def __init__(
        self,
        sequence: str,
        chain_id: str | list[str] | None = None,
        cyclic: bool = False,
    ):
        # validate the sequence matches DNA
        if not all(nt in set("ACGT") for nt in sequence.upper()):
            raise ValueError("Sequence contains invalid DNA nucleotides.")
        self.sequence = sequence
        self.chain_id = chain_id
        self.cyclic = cyclic


@dataclass
class RNA:
    """
    Represents an RNA sequence.

    Attributes:
        sequence (str): The nucleotide sequence of the RNA.
    """

    sequence: str
    chain_id: str | list[str] | None = None
    cyclic: bool = False

    def __init__(
        self,
        sequence: str,
        chain_id: str | list[str] | None = None,
        cyclic: bool = False,
    ):
        # validate the sequence matches RNA
        if not all(nt in set("ACGU") for nt in sequence.upper()):
            raise ValueError("Sequence contains invalid RNA nucleotides.")
        self.sequence = sequence
        self.chain_id = chain_id
        self.cyclic = cyclic


@dataclass
class Ligand:
    """
    Represents a ligand with optional Chemical Component Dictionary (CCD) identifier and SMILES string.

    Requires either a CCD identifier or SMILES string.

    Attributes:
        ccd (str | None): The CCD identifier for the ligand.
        smiles (str | None): The SMILES representation of the ligand.
    """

    chain_id: str | list[str] | None = None
    ccd: str | None = None
    smiles: str | None = None

    def __init__(
        self,
        *,
        chain_id: str | list[str] | None = None,
        ccd: str | None = None,
        smiles: str | None = None,
    ):
        self.chain_id = chain_id
        if (ccd is None and smiles is None) or (ccd is not None and smiles is not None):
            raise ValueError("Exactly one of 'ccd' or 'smiles' must be provided.")
        # TODO add validation
        self.ccd = ccd
        self.smiles = smiles
