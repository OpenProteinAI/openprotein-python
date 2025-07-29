import io
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import gemmi
import numpy as np
import numpy.typing as npt

from . import fasta

if TYPE_CHECKING:
    from openprotein.align import MSAFuture


# fmt: off
_ATOM_TYPES = (
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
)
# fmt: on
_N_ATOM = len(_ATOM_TYPES)
_ATOM_TYPE_TO_IDX = {atom_type: i for i, atom_type in enumerate(_ATOM_TYPES)}

_BACKBONE_ATOM_TYPES = ("N", "CA", "C")

_NAN_BFACTOR_VALUE = 9999.75  # can't/hard to use 9999.99 due to precision issues


def calc_rmsd(
    xyz1: npt.NDArray[np.floating], xyz2: npt.NDArray[np.floating], eps: float = 1e-6
) -> tuple[float, npt.NDArray[np.floating]]:
    """
    Calculates RMSD between two sets of atoms (L, 3)
    Adapted from https://github.com/RosettaCommons/RFdiffusion/blob/b44206a2a79f219bb1a649ea50603a284c225050/rfdiffusion/util.py#L719
    """
    # center to CA centroid
    xyz1 = xyz1 - xyz1.mean(0)
    xyz2 = xyz2 - xyz2.mean(0)

    # Computation of the covariance matrix
    C = xyz2.T @ xyz1

    # Compute otimal rotation matrix using SVD
    V, S, W = np.linalg.svd(C)

    # get sign to ensure right-handedness
    d = np.ones([3, 3])
    d[:, -1] = np.sign(np.linalg.det(V) * np.linalg.det(W))

    # Rotation matrix U
    U = (d * V) @ W

    # Rotate xyz2
    xyz2_ = xyz2 @ U
    L = xyz2_.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_ - xyz1) * (xyz2_ - xyz1), axis=(0, 1)) / L + eps)

    return rmsd, U


class Protein:
    """
    Represents a protein with optional sequence, atomic coordinates, per-residue
    confidence scores (pLDDT), and name.

    This class supports partial or complete information: users may initialize a Protein
    with only a sequence, only a structure, or both. The class ensures that all
    provided fields have consistent residue-level lengths and provides convenient
    methods for indexing, masking, and structural comparisons.

    Attributes:
        sequence: Amino acid sequence as bytes. Unknown or masked residues are
            represented as b"X".
        coordinates: an array containing the 3D coordinates of the heavy atoms of the
            protein in atom37 format. It has shape `(L, 37, 3)`, where `L` is the
            length of the protein, `37` is the number of heavy atoms, and `3` is the
            number of coordinates (x, y, and z).
        plddt: an array of shape `(L,)`. For predicted structures, this contains the
            pLDDT of each residue, which is a measure of prediction confidence. For
            experimental structures, this should be set to `100` if the coordinates of
            the alpha carbon are known, and `NaN` otherwise.
        name: Optional identifier for the protein as a string.

    Conventions:
        - Missing or unknown residues in the sequence are denoted by b"X".
        - Missing structural data (coordinates or pLDDT) are represented by NaN.
        - Residue indices are 1-based for user-facing methods (e.g., `mask_sequence_at`),
          but internally stored as 0-based arrays.

    Examples:
        Create a Protein from sequence only:
            Protein(sequence="ACDEFGHIK")

        Create a Protein from sequence and name:
            Protein(sequence="ACDEFGHIK", name="my_protein")

        Create a Protein with sequence and structure:
            Protein(sequence="ACD", coordinates=coords_array, plddt=plddt_array)

    Raises:
        ValueError: If sequence, coordinates, or pLDDT are specified with inconsistent lengths.
        ValueError: If none of sequence, coordinates, or pLDDT are provided.
    """

    def __init__(
        self,
        sequence: bytes | str | None = None,
        coordinates: npt.NDArray[np.float32] | None = None,
        plddt: npt.NDArray[np.float32] | None = None,
        name: bytes | str | None = None,
    ):
        lengths = {len(x) for x in (sequence, coordinates, plddt) if x is not None}
        if len(lengths) == 0:
            raise ValueError(
                "At least one of sequence, coordinates, or plddt must be specified."
            )
        elif len(lengths) > 1:
            raise ValueError(
                "Specified sequence, coordinates, and plddt must all have the same length."
            )
        length = next(iter(lengths))
        if sequence is not None:
            self._sequence = (
                sequence.encode() if isinstance(sequence, str) else sequence
            )
        else:
            self._sequence = b"X" * length
        if coordinates is not None:
            self._coordinates = coordinates
        else:
            self._coordinates = np.full((length, _N_ATOM, 3), np.nan, dtype=np.float32)
        if plddt is not None:
            self._plddt = plddt
        else:
            self._plddt = np.full((length,), np.nan, dtype=np.float32)
        if name is not None:
            self._name = name if isinstance(name, str) else name.decode()
        else:
            self._name = name
        self._tags = {}

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, x: bytes | str) -> None:
        self._name = x if isinstance(x, str) else x.decode()

    @property
    def sequence(self) -> bytes:
        return self._sequence

    @sequence.setter
    def sequence(self, x: bytes | str) -> None:
        assert len(x) == len(self)
        self._sequence = x.encode() if isinstance(x, str) else x

    @property
    def coordinates(self) -> npt.NDArray[np.float32]:
        return self._coordinates

    @coordinates.setter
    def coordinates(self, x: npt.NDArray[np.float32]) -> None:
        assert len(x) == len(self)
        self._coordinates = x

    @property
    def plddt(self) -> npt.NDArray[np.float32]:
        return self._plddt

    @plddt.setter
    def plddt(self, x: npt.NDArray[np.float32]) -> None:
        assert len(x) == len(self)
        self._plddt = x

    @property
    def chain_id(self) -> str | list[str] | None:
        return self._tags.get("chain_id")

    @chain_id.setter
    def chain_id(self, chain_id: str | list[str]) -> None:
        self._tags["chain_id"] = chain_id

    @property
    def cyclic(self) -> bool:
        return self._tags.get("cyclic") or False

    @cyclic.setter
    def cyclic(self, cyclic: bool) -> None:
        self._tags["cyclic"] = cyclic

    class NullMSA: ...

    single_sequence_mode = NullMSA

    @property
    def msa(self) -> "str | MSAFuture | None | NullMSA":
        return self._tags.get("msa")

    @msa.setter
    def msa(self, msa: "str | MSAFuture | None | NullMSA") -> None:
        self._tags["msa"] = msa

    def __len__(self):
        lengths = {
            len(x)
            for x in (self.sequence, self.coordinates, self.plddt)
            if x is not None
        }
        assert len(lengths) == 1
        return next(iter(lengths))

    def __getitem__(
        self, idx: int | list[int] | slice | npt.NDArray[np.integer]
    ) -> "Protein":
        """Return a new Protein object indexing into residues by `idx`."""
        if isinstance(idx, int):
            idx = np.array([idx], dtype=int)
        return Protein(
            sequence=np.frombuffer(self.sequence, dtype=np.uint8)[idx].tobytes(),
            coordinates=self.coordinates[idx].copy(),
            plddt=self.plddt[idx].copy(),
            name=self.name,
        )

    def __add__(self, tgt: "Protein") -> "Protein":
        """Return a new Protein object by concatenating with another Protein."""
        assert isinstance(tgt, Protein)
        return Protein(
            sequence=self.sequence + tgt.sequence,
            coordinates=np.concatenate((self.coordinates, tgt.coordinates)),
            plddt=np.concatenate((self.plddt, tgt.plddt)),
            name=self.name if self.name == tgt.name else None,
        )

    def at(self, positions: Sequence[int] | npt.NDArray[np.integer]) -> "Protein":
        """
        Return a new Protein object containing residues at given 1-indexed positions.
        """
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=int)
        return self[positions - 1]

    def mask_sequence_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask sequence at given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=int)
        idxs = positions - 1
        sequence = np.frombuffer(self.sequence, dtype=np.uint8).copy()
        sequence[idxs] = ord(b"X")
        return Protein(
            sequence=sequence.tobytes(),
            coordinates=self.coordinates.copy(),
            plddt=self.plddt.copy(),
            name=self.name,
        )

    def mask_sequence_except_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask sequence at all positions except the given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=int)
        idxs = positions - 1
        sequence = np.frombuffer(self.sequence, dtype=np.uint8).copy()
        mask = np.ones_like(sequence, dtype=bool)
        mask[idxs] = False
        sequence[mask] = ord(b"X")
        return Protein(
            sequence=sequence.tobytes(),
            coordinates=self.coordinates.copy(),
            plddt=self.plddt.copy(),
            name=self.name,
        )

    def mask_structure_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask structure at given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=int)
        idxs = positions - 1
        coordinates, plddt = self.coordinates.copy(), self.plddt.copy()
        coordinates[idxs], plddt[idxs] = np.nan, np.nan
        return Protein(
            sequence=self.sequence, coordinates=coordinates, plddt=plddt, name=self.name
        )

    def mask_structure_except_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask structure at all positions except the given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions, dtype=int)
        idxs = positions - 1
        mask = np.ones(len(self), dtype=bool)
        mask[idxs] = False
        coordinates, plddt = self.coordinates.copy(), self.plddt.copy()
        coordinates[mask], plddt[mask] = np.nan, np.nan
        return Protein(
            sequence=self.sequence, coordinates=coordinates, plddt=plddt, name=self.name
        )

    @property
    def has_structure(self) -> bool:
        """Whether or not the structure is known at any position in the protein."""
        return (not np.isnan(self.coordinates).all()) or (
            not np.isnan(self.plddt).all()
        )

    def rmsd(
        self, tgt: "Protein", backbone_only: bool | str | Sequence[str] = False
    ) -> float:
        """
        Compute the root-mean-square deviation (RMSD) between this Protein and a target
        Protein.

        Only atoms that are present (i.e., not NaN) in both structures are included in
        the calculation.

        Args:
            tgt: The target Protein to compare against.
            backbone_only: Specifies which atoms to include in the RMSD calculation.
                - If False (default), all atom types are included.
                - If True, only backbone atoms ("N", "CA", "C") are included.
                - If a string, it must be a single atom type (e.g., "CA").
                - If a sequence of strings, it must be a non-empty list of atom types
                  (e.g., ["CA", "CB", "O"]). All specified atom types must be valid.

        Returns:
            The RMSD value between the aligned structures.

        Notes:
            This method assumes that residues in `self` and `tgt` are already aligned.
        """
        if backbone_only is False:
            atom_idxs = np.arange(len(_ATOM_TYPES))
        elif backbone_only is True:
            atom_idxs = np.arange(3)
        elif isinstance(backbone_only, str):
            atom_idxs = [_ATOM_TYPE_TO_IDX[backbone_only]]
        elif isinstance(backbone_only, Sequence):
            assert len(backbone_only) > 0 and isinstance(next(iter(backbone_only)), str)
            atom_idxs = [_ATOM_TYPE_TO_IDX[x] for x in backbone_only]
        else:
            raise ValueError(backbone_only)
        src_coords = self.coordinates[:, atom_idxs]
        tgt_coords = tgt.coordinates[:, atom_idxs]
        src_known_atoms = ~np.isnan(src_coords).any(axis=2)
        tgt_known_atoms = ~np.isnan(tgt_coords).any(axis=2)
        overlapping_known_atoms = src_known_atoms & tgt_known_atoms
        src_coords = src_coords[overlapping_known_atoms]
        tgt_coords = tgt_coords[overlapping_known_atoms]
        rmsd, _ = calc_rmsd(src_coords, tgt_coords)
        return rmsd

    def make_cif_string(self) -> str:
        # TODO: add note about _NAN_BFACTOR_VALUE
        assert (
            self.has_structure
        ), "cannot make cif string for protein with no structure data"
        # Create an empty structure and add a model with a default chain.
        structure = gemmi.Structure()
        if self.name is not None:
            structure.name = self.name
        model = structure.add_model(gemmi.Model(1))

        # Process the sequence.
        resnames = gemmi.expand_one_letter_sequence(
            self.sequence.decode(), gemmi.ResidueKind.AA
        )
        entity = gemmi.Entity("1")
        entity.full_sequence = resnames
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = ["A"]
        structure.entities.append(entity)

        # Process the coordinates.
        n_nan_coords = np.isnan(self.coordinates).sum(axis=2)
        assert (
            (n_nan_coords == 0) | (n_nan_coords == 3)
        ).all(), "either all coords of an atom must be nan, or none are"
        # Process the plddt.
        assert (
            np.isnan(self.plddt) | (~np.isnan(self.plddt) & (n_nan_coords[:, 1] == 0))
        ).all(), "if plddt is known, coord of CA must be known"

        # Write the chain
        chain = model.add_chain(gemmi.Chain("A"))
        for i in range(len(self)):
            # Add a residue to the chain; note that residue numbering starts at 1.
            residue = gemmi.Residue()
            residue.entity_id = "1"
            residue.entity_type = gemmi.EntityType.Polymer
            residue.subchain = "A"
            residue.name = resnames[i]
            residue.label_seq = i + 1
            residue = chain.add_residue(residue, i + 1)
            # For each residue, add the atoms.
            for j, atom_name in enumerate(_ATOM_TYPES):
                if np.isnan(self.coordinates[i, j]).any():
                    continue
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*self.coordinates[i, j])
                if not np.isnan(self.plddt[i]):
                    atom.b_iso = self.plddt[i]
                else:
                    atom.b_iso = _NAN_BFACTOR_VALUE
                atom = residue.add_atom(atom)
        block = structure.make_mmcif_block()
        # NB: gemmi doesn't seem to write the _chem_comp category properly... it says
        #     the type is `.`, but is should be something like `L-PEPTIDE LINKING`...
        block.find_mmcif_category("_chem_comp").erase()  # ...so we remove it
        return block.as_string()

    def make_fasta_bytes(self) -> bytes:
        assert self.name is not None
        data = io.BytesIO()
        data.write(b">")
        data.write(self.name.encode())
        data.write(b"\n")
        data.write(
            self.sequence.encode()
            if not isinstance(self.sequence, bytes)
            else self.sequence
        )
        data.write(b"\n")
        return data.getvalue()

    @staticmethod
    def from_filepath(
        path: str | Path,
        chain_id: str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Protein":
        """
        Create a Protein from a structure file.

        If the structure file has multiple conformers, the first conformer is always
        used.

        Args:
            path: path to structure file (e.g. pdb or cif file)
            chain_id: id of the chain in the structure file to use
            use_bfactor_as_plddt: whether or not to use the bfactor of the CA atom as
                the plddt of structure of its residue. If None, this will be set to
                true only if the resolution of the structure is unspecified or zero.
            model_idx: index of the model in the structure file to use
            verbose: whether or not to print debugging information such as oddities in
                the structure e.g. missing atoms
        """
        structure = gemmi.read_structure(str(path))
        structure.name = Path(path).stem
        return Protein.from_structure(
            structure=structure,
            chain_id=chain_id,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        )

    @staticmethod
    def from_string(
        filestring: bytes | str,
        format: Literal["pdb", "cif"],
        chain_id: str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Protein":
        filestring = filestring if isinstance(filestring, str) else filestring.decode()
        if format == "pdb":
            structure = gemmi.read_pdb_string(filestring)
        elif format == "cif":
            structure = gemmi.make_structure_from_block(
                gemmi.cif.read_string(filestring).sole_block()
            )
        else:
            raise ValueError(f"Unknown {format=}")
        return Protein.from_structure(
            structure=structure,
            chain_id=chain_id,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        )

    @staticmethod
    def from_structure(
        structure: gemmi.Structure,
        chain_id: str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Protein":
        structure.setup_entities()
        structure.assign_label_seq_id()
        if use_bfactor_as_plddt is None:
            use_bfactor_as_plddt = structure.resolution == 0.0
        model = structure[model_idx]
        chain = model.find_chain(chain_id)
        assert chain is not None
        polymer = chain.get_polymer()

        # extract sequence
        entity = structure.get_entity_of(polymer)
        if len(entity.full_sequence) > 0:
            chain_seq = entity.full_sequence
        else:
            chain_seq = [residue.name for residue in polymer]
        chain_seq = [
            gemmi.find_tabulated_residue(
                # gemmi.Entity.first_mon extracts the first conformer
                gemmi.Entity.first_mon(residue_name)
            ).one_letter_code
            for residue_name in chain_seq
        ]
        # for find_tabulated_residue: lowercase means nonstandard, " " means unknown
        chain_seq = [c.upper() if c != " " else "X" for c in chain_seq]
        # extract coordinates and plddt
        coordinates = np.full((len(chain_seq), _N_ATOM, 3), np.nan, dtype=np.float32)
        plddt = np.full(len(chain_seq), np.nan, dtype=np.float32)
        for residue_idx, residue in enumerate(polymer):
            i = residue.label_seq - 1 if residue.label_seq is not None else residue_idx
            code = gemmi.find_tabulated_residue(residue.name).one_letter_code
            code = code.upper() if code != " " else "X"
            if code != chain_seq[i]:
                if verbose:
                    # TODO: can this ever happen...? probably want to have this regardless i guess
                    # TODO: improve this message?
                    print(
                        f"Amino acid mismatch at position {i + 1}: SEQRES {chain_seq[i]} Structure {code}"
                    )
                chain_seq[i] = code
            if verbose and code == "X" and residue.name != "UNK":
                print(f"Unknown amino acid at position {i + 1}: {residue.name}")
            if verbose:
                for j, atom_name in enumerate(_BACKBONE_ATOM_TYPES):
                    if atom_name not in residue:
                        print(
                            f"Residue at position {i + 1} missing backbone atom={atom_name}"
                        )
            for atom in residue.first_conformer():
                atom_name = atom.name
                if residue.name == "MSE" and atom_name == "SE":
                    atom_name = "SD"
                if (j := _ATOM_TYPE_TO_IDX.get(atom.name)) is None:
                    continue
                coordinates[i, j] = atom.pos.tolist()
                if use_bfactor_as_plddt and atom_name == "CA":
                    plddt[i] = (
                        atom.b_iso if atom.b_iso != _NAN_BFACTOR_VALUE else np.nan
                    )
            # TODO: we should experiment and see if this is the behavior we want
            if (
                not use_bfactor_as_plddt
                and np.isfinite(coordinates[i, _ATOM_TYPE_TO_IDX["CA"]]).all()
            ):
                plddt[i] = 100.0
        assert np.isnan(plddt).all() or (
            (np.nanmin(plddt) >= 0) and (np.nanmax(plddt) <= 100)
        )
        return Protein(
            sequence="".join(chain_seq),
            coordinates=coordinates,
            plddt=plddt,
            name=structure.name if structure.name != "" else None,
        )


def parse_fasta_as_proteins(path: str | Path) -> list[Protein]:
    proteins = []
    with open(path, "rb") as fp:
        for name, sequence in fasta.parse_stream(fp):
            proteins.append(Protein(name=name, sequence=sequence))
    return proteins
