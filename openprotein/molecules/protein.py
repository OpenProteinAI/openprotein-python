import enum
import gzip
import io
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeVar, cast, overload

import gemmi
import numpy as np
import numpy.typing as npt

import openprotein.utils.chain_id as _chain_id_utils
import openprotein.utils.cif as _cif_utils
import openprotein.utils.numpy as _numpy_utils
import openprotein.utils.sequence as _sequence_utils

from .. import fasta

if TYPE_CHECKING:
    from ..align.msa import MSAFuture
    from .complex import Complex

V = TypeVar("V")


class StrEnum(str, enum.Enum): ...


@enum.unique
class Binding(StrEnum):
    # TODO: should we use any/X/?/* or something else instead of unknown?
    UNKNOWN = "U"
    BINDING = "B"
    NOT_BINDING = "N"


# TODO: deserialization note about plddt parsed per residue
class Protein:
    """
    Represents a protein with an optional name.

    This class supports partial or complete information: users may create a Protein
    with only a sequence, only a structure, or both. The class ensures that all
    provided fields have consistent residue-level lengths and provides convenient
    methods for indexing, masking, and structural comparisons.

    Conventions:
        - Missing or unknown residues in the sequence are denoted by b"X".
        - Missing structural data (coordinates or pLDDT) are represented by NaN.
        - Residue indices are 1-indexed for user-facing methods suffixed with `at` E.g.
          `.at`, `mask_sequence_at`

    Examples:
        Create a Protein from sequence only:
            Protein(sequence="ACDEFGHIK")

        Create a Protein from sequence and name:
            Protein(sequence="ACDEFGHIK", name="my_protein")
    """

    def __init__(self, sequence: bytes | str, name: bytes | str | None = None):
        assert set(
            sequence if isinstance(sequence, str) else sequence.decode()
        ).issubset(
            set(_sequence_utils.AMINO_ACIDS + _sequence_utils.EXTRA_TOKENS)
        ), "Expected only amino acids or the mask token 'X' or the variable length token '?'\nHint: Use Protein.from_expr if using a sequence expression"
        self._sequence = sequence.encode() if isinstance(sequence, str) else sequence
        self.name = name
        # sequence-level properties
        self._cyclic: bool = False
        self._msa: "str | MSAFuture | None | Type[Protein.NullMSA]" = None
        # per-residue arrays
        self._data: dict[str, npt.NDArray] = {}

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, x: bytes | str | None) -> None:
        self._name = x.decode() if isinstance(x, bytes) else x

    def set_name(self, x: bytes | str | None) -> "Protein":
        self.name = x
        return self

    @property
    def sequence(self) -> bytes:
        return self._sequence

    @sequence.setter
    def sequence(self, x: bytes | str) -> None:
        assert set(x if isinstance(x, str) else x.decode()).issubset(
            set(_sequence_utils.AMINO_ACIDS + _sequence_utils.EXTRA_TOKENS)
        ), "Expected only amino acids or the mask token 'X' or the variable length token '?'\nHint: Use Protein.from_expr if using a sequence expression"
        assert len(x) == len(self)
        self._sequence = x.encode() if isinstance(x, str) else x

    def set_sequence(self, x: bytes | str) -> "Protein":
        self.sequence = x
        return self

    @property
    def coordinates(self) -> npt.NDArray[np.float32]:
        return _numpy_utils.readonly_view(self._coordinates)

    @property
    def plddt(self) -> npt.NDArray[np.float32]:
        return _numpy_utils.readonly_view(self._plddt)

    @property
    def cyclic(self) -> bool:
        return self._cyclic

    @cyclic.setter
    def cyclic(self, cyclic: bool) -> None:
        self._cyclic = cyclic

    def get_cyclic(self) -> bool:
        return self._cyclic

    def set_cyclic(self, x: bool) -> "Protein":
        self._cyclic = x
        return self

    class NullMSA: ...

    single_sequence_mode = NullMSA

    @property
    def msa(self) -> "str | MSAFuture | None | Type[NullMSA]":
        """A reference identifier to the MSA associated to this protein."""
        return self._msa

    @msa.setter
    def msa(self, msa: "str | MSAFuture | None | Type[NullMSA]") -> None:
        # NB: no defensive copy of msa b/c we don't want to copy things like session
        #     objects, but msa should really be immutable anyways...
        self._msa = msa

    def get_msa(self) -> "str | MSAFuture | None | Type[NullMSA]":
        return self._msa

    def set_msa(self, x: "str | MSAFuture | None | Type[NullMSA]") -> "Protein":
        self._msa = x
        return self

    def __len__(self):
        return len(self.sequence)

    def __getitem__(
        self, idx: int | slice | Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Return a new Protein object indexing into residues by `idx`."""
        if isinstance(idx, int):
            idx = np.array([idx], dtype=int)
        elif isinstance(idx, slice):
            idx = np.arange(idx.start or 0, idx.stop or len(self), idx.step or 1)
        elif not isinstance(idx, np.ndarray):
            idx = np.fromiter(idx, dtype=int)
        new = Protein(
            sequence=np.frombuffer(self.sequence, dtype=np.uint8)[idx].tobytes(),
            name=self.name,
        )
        # TODO: check msa compatible?
        new = new.set_msa(self._msa).set_cyclic(self._cyclic)
        new._data = {k: v[idx].copy() for k, v in self._data.items()}
        return new

    def __radd__(self, left: "Protein | str") -> "Protein":
        assert isinstance(left, Protein) or isinstance(left, str)
        if isinstance(left, str):
            left = self.from_expr(expr=left)
        return left + self

    def __add__(self, right: "Protein | str") -> "Protein":
        """Return a new Protein object by concatenating with another Protein."""
        assert isinstance(right, Protein) or isinstance(right, str)
        if isinstance(right, str):
            right = self.from_expr(right)
        # TODO: if either cyclic, should we actually disable adding?
        assert right._msa == self._msa and right._cyclic == self._cyclic
        new = Protein(
            sequence=self.sequence + right.sequence,
            name=(  # set name if equal, or if only one of the two have a name
                self.name
                if self.name == right.name or right.name is None
                else right.name if self.name is None else None
            ),
        )
        new = new.set_msa(self._msa).set_cyclic(self._cyclic)
        new._data = {
            k: np.concatenate((getattr(self, f"_{k}"), getattr(right, f"_{k}")))
            for k in self._data.keys() | right._data.keys()
        }
        return new

    def __rand__(self, left: "Complex | Protein | str") -> "Complex":
        if isinstance(left, str):
            left = self.from_expr(expr=left)
        return left & self

    def __and__(self, right: "Complex | Protein | str") -> "Complex":
        """Combine multiple objects into a single Complex."""
        from .complex import Complex

        assert (
            isinstance(right, Complex)
            or isinstance(right, Protein)
            or isinstance(right, str)
        )
        if isinstance(right, str):
            right = self.from_expr(right)
        if isinstance(right, Protein):
            id_gen = _chain_id_utils.id_generator()
            return Complex({next(id_gen): self, next(id_gen): right})
        return right & self

    def at(self, positions: Sequence[int] | npt.NDArray[np.integer]) -> "Protein":
        """
        Return a new Protein object containing residues at given 1-indexed positions.
        """
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        return self[positions - 1]

    def mask_sequence(self) -> "Protein":
        """Mask entire sequence."""
        return self.mask_sequence_except_at([])

    def mask_sequence_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask sequence at given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        sequence = np.frombuffer(self.sequence, dtype=np.uint8).copy()
        sequence[positions - 1] = ord(b"X")
        return self.set_sequence(sequence.tobytes())

    def mask_sequence_except_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> "Protein":
        """Mask sequence at all positions except the given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        sequence = np.frombuffer(self.sequence, dtype=np.uint8).copy()
        mask = np.ones_like(sequence, dtype=bool)
        mask[positions - 1] = False
        sequence[mask] = ord(b"X")
        return self.set_sequence(sequence.tobytes())

    def mask_structure(self, side_chain_only: bool = False) -> "Protein":
        """Mask entire structure."""
        return self.mask_structure_except_at([], side_chain_only=side_chain_only)

    def mask_structure_at(
        self,
        positions: Sequence[int] | npt.NDArray[np.integer],
        side_chain_only: bool = False,
    ) -> "Protein":
        """Mask structure at given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        idxs = positions - 1
        atom_idxs = (
            np.arange(len(_ATOM_TYPES))
            if not side_chain_only
            else _SIDE_CHAIN_ATOM_IDXS
        )
        self._coordinates[np.ix_(idxs, atom_idxs)] = np.nan
        if not side_chain_only:
            self._plddt[idxs] = np.nan
        return self

    def mask_structure_except_at(
        self,
        positions: Sequence[int] | npt.NDArray[np.integer],
        side_chain_only: bool = False,
    ) -> "Protein":
        """Mask structure at all positions except the given 1-indexed positions."""
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        mask = np.ones(len(self), dtype=bool)
        mask[positions - 1] = False
        return self.mask_structure_at(
            positions=np.where(mask)[0] + 1, side_chain_only=side_chain_only
        )

    def get_structure_mask(self) -> npt.NDArray[np.bool_]:
        """
        Computes the structure mask of the protein. The structure mask is a boolean
        array indicating, at each position, whether the structure is undefined at that
        position.
        """
        return np.all(np.all(np.isnan(self._coordinates), axis=2), axis=1)

    @property
    def has_structure(self) -> bool:
        """Whether or not the structure is known at any position in the protein."""
        return (not np.isnan(self._coordinates).all()) or (
            not np.isnan(self._plddt).all()
        )

    def get_group_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray[np.int_]:
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        return _numpy_utils.readonly_view(self._group[positions - 1])

    def set_group_at(
        self,
        positions: Sequence[int] | npt.NDArray[np.integer],
        value: int | Sequence[int],
    ) -> "Protein":
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        self._group[positions - 1] = value
        return self

    @property
    def group(self) -> npt.NDArray[np.integer]:
        return _numpy_utils.readonly_view(self._group)

    def get_group(self) -> npt.NDArray[np.integer]:
        return _numpy_utils.readonly_view(self._group)

    def set_group(self, value: int) -> "Protein":
        self._group[:] = value
        return self

    def get_binding_at(
        self, positions: Sequence[int] | npt.NDArray[np.integer]
    ) -> npt.NDArray[np.str_]:
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        return _numpy_utils.readonly_view(self._binding[positions - 1])

    def set_binding_at(
        self,
        positions: Sequence[int] | npt.NDArray[np.integer],
        value: Binding | str | Sequence[Binding | str],
    ) -> "Protein":
        if not isinstance(positions, np.ndarray):
            positions = np.fromiter(positions, dtype=int)
        self._binding[positions - 1] = _enum_to_str(value=value, enum_type=Binding)
        return self

    @property
    def binding(self) -> npt.NDArray[np.str_]:
        return _numpy_utils.readonly_view(self._binding)

    def get_binding(self) -> npt.NDArray[np.str_]:
        return _numpy_utils.readonly_view(self._binding)

    @overload
    def rmsd(
        self,
        tgt: "Protein",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: Literal[False] = False,
    ) -> float: ...

    @overload
    def rmsd(
        self,
        tgt: "Protein",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: Literal[True] = True,
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    def rmsd(
        self,
        tgt: "Protein",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: bool = False,
    ) -> float | tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
            return_transform: If True, returns both the rmsd and the transformation that
                should be applied to `tgt` to superimpose it onto this Protein. If False
                (default), returns only the rmsd value.

        Returns:
            If `return_transform` is False (default):
                The RMSD value (float).
            If `return_transform` is True:
                A tuple `(float, np.ndarray, np.ndarray)` containing the RMSD value,
                the rotation matrix, and the translation vector.

        Notes:
            This method assumes that sequences of `self` and `tgt` are already aligned.
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
        src_coords = self._coordinates[:, atom_idxs]
        tgt_coords = tgt._coordinates[:, atom_idxs]
        src_known_atoms = ~np.isnan(src_coords).any(axis=-1)
        tgt_known_atoms = ~np.isnan(tgt_coords).any(axis=-1)
        overlapping_known_atoms = src_known_atoms & tgt_known_atoms
        src_coords = src_coords[overlapping_known_atoms]
        tgt_coords = tgt_coords[overlapping_known_atoms]
        rmsd, R, t = _calc_rmsd_and_transform(src_coords, tgt_coords)
        if return_transform:
            return rmsd, R, t
        return rmsd

    def transform(
        self,
        R: npt.NDArray[np.floating] | None = None,
        t: npt.NDArray[np.floating] | None = None,
    ) -> "Protein":
        if R is None:
            R = np.eye(3, dtype=np.float32)
        if t is None:
            t = np.zeros(3, dtype=np.float32)
        return self._set_coordinates(
            self._coordinates @ R.T.astype(np.float32) + t.astype(np.float32)
        )

    def superimpose_onto(
        self, tgt: "Protein", backbone_only: bool | str | Sequence[str] = False
    ) -> "Protein":
        _, R, t = tgt.rmsd(self, backbone_only=backbone_only, return_transform=True)
        return self.transform(R=R, t=t)

    def to_string(self, format: Literal["cif", "pdb"] = "cif") -> str:
        """
        Serialize this Protein to a string. Note that format="pdb" may not serialize all
        aspects of this object, so format="cif", the default, is preferred.
        """
        if format == "cif":
            return self._make_cif_string()
        elif format == "pdb":
            return self._make_pdb_string()
        else:
            raise ValueError(format)

    def make_cif_string(self) -> str:
        warnings.warn(
            "`make_cif_string()` is deprecated and will be removed in v0.11. "
            "Use `to_string()` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self._make_cif_string()

    def make_pdb_string(self) -> str:
        warnings.warn(
            "`make_pdb_string()` is deprecated and will be removed in v0.11. "
            'Use `to_string(format="pdb")` instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self._make_pdb_string()

    def make_fasta_bytes(self) -> bytes:
        assert self.name is not None
        data = io.BytesIO()
        data.write(b">")
        data.write(self.name.encode())
        data.write(b"\n")
        data.write(self.sequence)
        data.write(b"\n")
        return data.getvalue()

    @staticmethod
    def from_expr(expr: str | int, name: str | None = None) -> "Protein":
        """
        Create a Protein from a sequence expression.

        A sequence expression allows you to define protein sequences using a concise
        notation that mixes fixed sequences, design regions, and length ranges.

        Useful for creating a design :py:class:`~openprotein.prompt.Query`.

        Args:
            expr: Sequence expression string or integer
                - Fixed sequences: "ACGT" (literal amino acids)
                - Design regions: "6" or 6 (any 6 amino acids)
                - Length ranges: "3..5" (between 3-5 amino acids)
                - Combined: "AAAA6C3..5" (AAAA + 6 design + C + 3-5 design)
            name: Optional name for the protein

        Returns:
            Protein object with the parsed sequence

        Examples:
            >>> # Fixed sequence with 6 flexible positions and fixed end
            >>> Protein.from_expr("MKLL6VVAA").sequence
            >>> b'MKLLXXXXXXVVAA'

            >>> # Design region of any 15 amino acids
            >>> Protein.from_expr(15).sequence
            >>> b'XXXXXXXXXXXXXXX'

            >>> # Variable length region between 10-20 residues
            >>> Protein.from_expr("10..20").sequence
            >>> b'XXXXXXXXXX??????????'
        """
        if isinstance(expr, int):
            expr = str(expr)
        sequence = _sequence_utils.SequenceExpr.parse(expr).to_protein_sequence()
        return Protein(sequence=sequence, name=name)

    @staticmethod
    def from_filepath(
        path: Path | str,
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
            use_bfactor_as_plddt: whether or not to use bfactors as pLDDTs. If None,
                this parameter will be determined based on heuristics. These heuristics
                may change over time.
            model_idx: index of the model in the structure file to use
            verbose: whether or not to print debugging information such as oddities in
                the structure e.g. missing atoms
        """
        path = Path(path)
        if path.suffix == ".gz":
            if path.name.endswith(".cif.gz"):
                ext, format = ".cif.gz", "cif"
            elif path.name.endswith(".pdb.gz"):
                ext, format = ".pdb.gz", "pdb"
            else:
                raise ValueError(f"unsupported format: {path}")
            with gzip.open(path, "rb") as f:
                data = f.read()
        else:
            ext = path.suffix
            format = ext.removeprefix(".")
            assert format == "cif" or format == "pdb"
            data = path.read_bytes()
        return Protein.from_string(
            filestring=data,
            format=format,
            chain_id=chain_id,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        ).set_name(path.name.removesuffix(ext))

    @staticmethod
    def from_string(
        filestring: bytes | str,
        format: Literal["pdb", "cif"],
        chain_id: str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Protein":
        structure_block = _cif_utils.StructureCIFBlock(
            filestring=filestring, format=format
        )
        return Protein._from_structure_block(
            structure_block=structure_block,
            chain_id=chain_id,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        )

    def formatted(
        self,
        include: Sequence[Literal["sequence", "structure_mask", "binding", "group"]] = (
            "sequence",
        ),
        width: int = 60,
        value_maps: dict[str, dict[Any, str]] | None = None,
    ) -> str:
        """
        Format the sequence and/or additional feature tracks aligned and wrapped to a
        specific width.
        """
        value_maps = value_maps or {
            "structure_mask": {True: "^", False: " "},
            "binding": {Binding.UNKNOWN.value: " "},
        }
        tracks: dict[str, Sequence | npt.NDArray] = {}
        if "sequence" in include:
            tracks["sequence"] = self.sequence.decode()
        if "structure_mask" in include:
            tracks["structure_mask"] = self.get_structure_mask()
        if "binding" in include:
            tracks["binding"] = self.get_binding()
        if "group" in include:
            tracks["group"] = self.get_group()
        tracks = {x: tracks[x] for x in include}
        label_width = max(len(name) for name in tracks.keys())
        lines: list[str] = []
        for i in range(0, len(self), width):
            # Blank line between blocks (but not before the first one)
            if i > 0:
                lines.append("")
            for name, sequence in tracks.items():
                chunk = sequence[i : i + width]
                mapper = value_maps.get(name, {})
                chunk_str = "".join(mapper.get(x, str(x)) for x in chunk)
                # Line Format: Index Label Chunk
                lines.append(f"{i:<5} {name.upper():<{label_width}} {chunk_str}")
        return "\n".join(lines)

    def __str__(self):
        return self.formatted(include=("sequence",))

    def copy(self) -> "Protein":
        return self[:]

    @staticmethod
    def _from_structure_block(
        structure_block: _cif_utils.StructureCIFBlock,
        chain_id: str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Protein":
        structure = structure_block.structure
        maybe_use_bfactor_as_plddt = use_bfactor_as_plddt is None
        if use_bfactor_as_plddt is None:
            use_bfactor_as_plddt = _use_bfactor_as_plddt(structure=structure)
        model = structure[model_idx] if len(structure) > 0 else None
        subchain_id = chain_id
        if (
            structure.input_format == gemmi.CoorFormat.Pdb
            and not subchain_id.endswith("xp")
            and model is not None
        ):
            subchain_id = model.find_chain(chain_id).get_polymer().subchain_id()
        if model is None or len(model.get_subchain(subchain_id)) == 0:
            matching_entities = [
                e for e in structure.entities if subchain_id in e.subchains
            ]
            assert len(matching_entities) == 1, (
                f"expected only one entity to match {chain_id=}, but found "
                f"{len(matching_entities)}: {matching_entities}"
            )
            entity = matching_entities[0]
            assert (
                entity.entity_type == gemmi.EntityType.Polymer
            ), f"expected entity type polymer, got {entity.entity_type}"
            assert (
                entity.polymer_type == gemmi.PolymerType.PeptideL
            ), f"expected polymer type PeptideL, got {entity.polymer_type}"
            if len(entity.full_sequence) > 0:
                chain_seq = entity.full_sequence
            else:
                chain_seq, _ = structure_block.full_sequences[entity.name]
            chain_seq = _extract_one_letter_from_full_sequence(chain_seq)
            protein = Protein(
                sequence="".join(chain_seq),
                name=structure.name if structure.name != "" else None,
            )
            protein._set_loop_data(
                structure_block=structure_block, chain_id=subchain_id
            )
            return protein
        model = structure[model_idx]
        polymer = model.get_subchain(subchain_id)
        assert len(polymer) > 0
        # extract sequence
        entity = structure.get_entity_of(polymer)
        residues = list(polymer.first_conformer())
        # TODO: consider utilizing polymer.make_one_letter_sequence() here or elsewhere
        del polymer
        if len(entity.full_sequence) > 0:
            chain_seq, label_seq_offset = entity.full_sequence, 1
        elif entity.name in structure_block.full_sequences:
            chain_seq, label_seq_offset = structure_block.full_sequences[entity.name]
        else:
            chain_seq, label_seq_offset = _extract_full_sequence_from_residues(
                residues=residues
            )
        chain_seq = _extract_one_letter_from_full_sequence(full_sequence=chain_seq)
        # extract coordinates and plddt
        coordinates = np.full((len(chain_seq), _N_ATOM, 3), np.nan, dtype=np.float32)
        plddt = np.full(len(chain_seq), np.nan, dtype=np.float32)
        for residue_idx, residue in enumerate(residues):
            i = (
                residue.label_seq - label_seq_offset
                if residue.label_seq is not None
                else residue_idx
            )
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
        if (
            maybe_use_bfactor_as_plddt
            and not np.isnan(plddt).all()
            and np.nanmax(plddt) <= 10
        ):
            plddt[~np.isnan(plddt)] = 100.0  # these were almost surely not plddts
        assert np.isnan(plddt).all() or (
            (np.nanmin(plddt) >= 0) and (np.nanmax(plddt) <= 100)
        )
        protein = Protein(
            sequence="".join(chain_seq),
            name=structure.name if structure.name != "" else None,
        )
        protein._coordinates, protein._plddt = coordinates, plddt
        protein._set_loop_data(structure_block=structure_block, chain_id=subchain_id)
        return protein

    def _set_loop_data(
        self, structure_block: _cif_utils.StructureCIFBlock, chain_id: str
    ):
        # TODO: chain id overload, extract this into its own method
        # TODO: for all tables, support optional columns being missing
        columns = ["label_asym_id", "?cyclic", "?msa_id"]
        table = structure_block.block.find("_openprotein_sequence.", columns)
        if len(table) > 0:
            assert all(table.has_column(i) for i in range(len(columns)))
            for _chain_id, cyclic, msa_id in table:
                if _chain_id != chain_id:
                    continue
                self._cyclic = cyclic == "1"
                if msa_id == ".":
                    self._msa = Protein.single_sequence_mode
                elif msa_id != "?":
                    self._msa = msa_id
        columns = [
            "label_asym_id",
            "label_seq_id",
            "label_atom_id",
            "?group",
            "?binding",
        ]
        table = structure_block.block.find("_openprotein_atom.", columns)
        if len(table) > 0:
            assert all(table.has_column(i) for i in range(len(columns)))
            for _chain_id, seq_id, atom_id, group, binding in table:
                if _chain_id != chain_id:
                    continue
                idx = int(seq_id) - 1
                assert idx < len(self)
                assert atom_id == ".", "atom level not supported yet"
                self._group[idx] = int(group)
                self._binding[idx] = binding

    @property
    def _coordinates(self) -> npt.NDArray[np.float32]:
        if "coordinates" not in self._data:
            self._data["coordinates"] = np.full(
                (len(self), _N_ATOM, 3), np.nan, dtype=np.float32
            )
        return self._data["coordinates"]

    @_coordinates.setter
    def _coordinates(self, x: npt.NDArray[np.float32]) -> None:
        assert x.dtype == np.float32 and x.shape == (len(self), _N_ATOM, 3)
        self._data["coordinates"] = x

    @property
    def _plddt(self) -> npt.NDArray[np.float32]:
        if "plddt" not in self._data:
            self._data["plddt"] = np.full(len(self), np.nan, dtype=np.float32)
        return self._data["plddt"]

    @_plddt.setter
    def _plddt(self, x: npt.NDArray[np.float32]) -> None:
        assert x.dtype == np.float32 and x.shape == (len(self),)
        self._data["plddt"] = x

    @property
    def _group(self) -> npt.NDArray[np.int_]:
        if "group" not in self._data:
            self._data["group"] = np.zeros(len(self), dtype=int)
        return self._data["group"]

    @_group.setter
    def _group(self, x: npt.NDArray[np.int_]) -> None:
        assert np.issubdtype(x.dtype, np.integer) and x.shape == (len(self),)
        self._data["group"] = x.astype(int, copy=False)

    @property
    def _binding(self) -> npt.NDArray[np.str_]:
        if "binding" not in self._data:
            self._data["binding"] = np.full(
                len(self), Binding.UNKNOWN.value, dtype="<U1"
            )
        return self._data["binding"]

    @_binding.setter
    def _binding(self, x: npt.NDArray[np.str_]) -> None:
        assert x.dtype == "<U1" and x.shape == (len(self),)
        assert set(x).issubset({e.value for e in Binding})
        self._data["binding"] = x

    def _make_structure(
        self,
        structure: gemmi.Structure | None = None,
        model_idx: int = 1,
        chain_id: str = "A",
        entity_name: str = "1",
    ) -> gemmi.Structure:
        # TODO: add note about _NAN_BFACTOR_VALUE
        # Create an empty structure and add a model with a default chain.
        if structure is None:
            structure = gemmi.Structure()
            if self.name is not None:
                structure.name = self.name
        # Get existing model or create new one
        if len(structure) > 0:
            model = structure[model_idx]
        else:
            model = structure.add_model(gemmi.Model(str(model_idx)))  # type: ignore - gemmi 0.6 needs str
        # Process the sequence.
        # TODO: handle optional token...?
        resnames = gemmi.expand_one_letter_sequence(
            self.sequence.decode(), gemmi.ResidueKind.AA
        )
        entity = gemmi.Entity(entity_name)
        entity.full_sequence = resnames
        entity.entity_type = gemmi.EntityType.Polymer
        entity.polymer_type = gemmi.PolymerType.PeptideL
        entity.subchains = [chain_id]
        structure.entities.append(entity)
        # Process the coordinates.
        n_nan_coords = np.isnan(self._coordinates).sum(axis=2)
        assert (
            (n_nan_coords == 0) | (n_nan_coords == 3)
        ).all(), "either all coords of an atom must be nan, or none are"
        # Process the plddt.
        assert (
            np.isnan(self._plddt) | (~np.isnan(self._plddt) & (n_nan_coords[:, 1] == 0))
        ).all(), "if plddt is known, coord of CA must be known"
        # Write the chain
        chain = model.add_chain(gemmi.Chain(chain_id))
        for i in range(len(self)):
            # Add a residue to the chain; note that residue numbering starts at 1.
            residue = gemmi.Residue()
            residue.entity_id = entity_name
            residue.entity_type = gemmi.EntityType.Polymer
            residue.subchain = chain_id
            residue.name = resnames[i]
            residue.label_seq = i + 1
            residue.seqid = gemmi.SeqId(str(i + 1))
            residue = chain.add_residue(residue, i + 1)
            # For each residue, add the atoms.
            for j, atom_name in enumerate(_ATOM_TYPES):
                if np.isnan(self._coordinates[i, j]).any():
                    continue
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.element = gemmi.Element(atom_name[0])
                atom.pos = gemmi.Position(*self._coordinates[i, j])
                if not np.isnan(self._plddt[i]):
                    atom.b_iso = self._plddt[i]
                else:
                    atom.b_iso = _NAN_BFACTOR_VALUE
                atom = residue.add_atom(atom)
        return structure

    def _append_loop_data(
        self, chain_id: str, sequence_loop: gemmi.cif.Loop, atom_loop: gemmi.cif.Loop
    ):
        if self._cyclic or self._msa is not None:
            if self._msa is None:
                msa_id = "?"  # cif convention for unknown
            elif isinstance(self._msa, type):
                msa_id = "."  # cif convention for not applicable
            elif isinstance(self._msa, str):
                msa_id = self._msa
            else:
                msa_id = self._msa.id
            sequence_loop.add_row([chain_id, "1" if self._cyclic else "0", msa_id])
        for idx, (binding, group) in enumerate(zip(self._binding, self._group)):
            if binding == Binding.UNKNOWN and group == 0:
                continue  # don't write default
            atom_loop.add_row(
                # "."" for atom id indicates residue level annotation
                [chain_id, str(idx + 1), ".", str(group), binding]
            )

    def _make_cif_string(self) -> str:
        # TODO: make gemmi take into account chain_id
        structure = self._make_structure()
        # NB: gemmi doesn't seem to write the _chem_comp category properly... it says
        #     the type is `.`, but is should be something like `L-PEPTIDE LINKING`...
        #     see also: https://github.com/project-gemmi/gemmi/discussions/362
        block = structure.make_mmcif_block(
            groups=gemmi.MmcifOutputGroups(True, chem_comp=False)
        )
        sequence_loop, atom_loop = _cif_utils.init_loops(block=block)
        self._append_loop_data(
            chain_id="A", sequence_loop=sequence_loop, atom_loop=atom_loop
        )
        return block.as_string()

    def _make_pdb_string(self) -> str:
        # TODO: make gemmi take into account chain_id
        structure = self._make_structure()
        return structure.make_pdb_string(gemmi.PdbWriteOptions(minimal=True))

    def _set_coordinates(self, x: npt.NDArray[np.float32]) -> "Protein":
        self._coordinates = x.copy()
        return self


def parse_fasta_as_proteins(path: str | Path) -> list[Protein]:
    proteins = []
    with open(path, "rb") as fp:
        for name, sequence in fasta.parse_stream(fp):
            proteins.append(Protein(name=name, sequence=sequence))
    return proteins


T = TypeVar("T", bound=StrEnum)


def _enum_to_str(
    value: T | str | Sequence[T | str], enum_type: Type[T]
) -> str | Sequence[str]:
    if isinstance(value, enum_type):
        value = value.value
    elif isinstance(value, str):
        value = enum_type(value)
        value = value.value
    else:
        value = [enum_type(v) if isinstance(v, str) else v for v in value]
        value = [v.value for v in value]
    return value


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
_SIDE_CHAIN_ATOM_IDXS = np.array(
    [
        i
        for i, atom_type in enumerate(_ATOM_TYPES)
        if atom_type not in _BACKBONE_ATOM_TYPES + ("O",)
    ]
)

_EXPERIMENTAL_METHODS = {
    "X-RAY DIFFRACTION",
    "ELECTRON MICROSCOPY",
    "SOLUTION NMR",
    "SOLID-STATE NMR",
    "NEUTRON DIFFRACTION",
    "ELECTRON CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
    "POWDER DIFFRACTION",
    "INFRARED SPECTROSCOPY",
    "FLUORESCENCE TRANSFER",
    "EPR",
    "SOLUTION SCATTERING",
}

_NAN_BFACTOR_VALUE = 9999.75  # can't/hard to use 9999.99 due to precision issues


def _calc_rmsd_and_transform(
    xyz1: npt.NDArray[np.floating], xyz2: npt.NDArray[np.floating], eps: float = 1e-6
) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Calculates RMSD and the rigid transformation (R, t) to superimpose xyz2 onto xyz1.
    Adapted from https://github.com/RosettaCommons/RFdiffusion/blob/b44206a2a79f219bb1a649ea50603a284c225050/rfdiffusion/util.py#L719

    Returns:
        rmsd: Root Mean Square Deviation.
        R: Rotation matrix (3, 3) such that xyz2 @ R.T + t aligns with xyz1.
        t: Translation vector (3,) such that xyz2 @ R.T + t aligns with xyz1.
    """
    # 1. Compute means to center the coordinates
    mu1 = xyz1.mean(axis=0)
    mu2 = xyz2.mean(axis=0)

    xyz1_c = xyz1 - mu1
    xyz2_c = xyz2 - mu2

    # 2. Computation of the covariance matrix
    C = xyz2_c.T @ xyz1_c

    # 3. Compute optimal rotation matrix using SVD
    # Note: numpy.linalg.svd returns U, S, Vh (where Vh is V.T)
    # The variable names V, W below follow the original code's notation logic
    V, S, W = np.linalg.svd(C)

    # 4. Get sign to ensure right-handedness (correct for reflections)
    d = np.ones((3, 3))
    d[:, -1] = np.sign(np.linalg.det(V) * np.linalg.det(W))

    # 5. Rotation matrix U (applied on the right: xyz_new = xyz_old @ U)
    # This U corresponds to R.T in the formula: x_new = R @ x_old
    U = (d * V) @ W

    # 6. Rotate xyz2 (centered) to calculate RMSD
    xyz2_aligned_c = xyz2_c @ U
    L = xyz2_aligned_c.shape[0]
    rmsd = np.sqrt(np.sum((xyz2_aligned_c - xyz1_c) ** 2, axis=(0, 1)) / L + eps)

    # 7. Compute R and t
    # We want: xyz2 @ R.T + t
    # We have: xyz2_aligned = (xyz2 - mu2) @ U + mu1
    # Expand:  xyz2_aligned = xyz2 @ U - mu2 @ U + mu1
    # Therefore: R.T = U  => R = U.T
    #            t = mu1 - mu2 @ U

    R = U.T
    t = mu1 - mu2 @ U

    return rmsd, R.astype(xyz2.dtype), t.astype(xyz2.dtype)


def _is_experimental_structure(structure: gemmi.Structure) -> bool:
    """
    This heuristic decides whether the structure is an experimental structure.
    This heuristic may be changed in the future.
    """
    if structure.resolution > 0:
        return True
    else:
        return ("_exptl.method" in structure.info) and (
            structure.info["_exptl.method"] in _EXPERIMENTAL_METHODS
        )


def _use_bfactor_as_plddt(structure: gemmi.Structure) -> bool:
    """
    This heuristic decides whether to use B-factor as pLDDT.
    This heuristic may be changed in the future.
    """
    return not _is_experimental_structure(structure=structure)


def _extract_full_sequence_from_residues(
    residues: list[gemmi.Residue],
) -> tuple[list[str], int]:
    if all(residue.label_seq is not None for residue in residues):
        label_seqs = [cast(int, residue.label_seq) for residue in residues]
        first_label_seq, last_label_seq = min(label_seqs), max(label_seqs)
        chain_seq = ["UNK"] * (last_label_seq - first_label_seq + 1)
        for residue in residues:
            chain_seq[cast(int, residue.label_seq) - first_label_seq] = residue.name
    else:
        assert all(residue.label_seq is None for residue in residues), (
            "if entity.full_sequence is blank, then either all residues must "
            "have label_seq or all residues must not have label_seq"
        )
        chain_seq, first_label_seq = [residue.name for residue in residues], 0
    return chain_seq, first_label_seq


def _extract_one_letter_from_full_sequence(full_sequence: Sequence[str]) -> list[str]:
    chain_seq = [
        gemmi.find_tabulated_residue(
            # gemmi.Entity.first_mon extracts the first conformer
            gemmi.Entity.first_mon(residue_name)
        ).one_letter_code
        for residue_name in full_sequence
    ]
    # for find_tabulated_residue: lowercase means nonstandard, " " means unknown
    return [c.upper() if c != " " else "X" for c in chain_seq]
