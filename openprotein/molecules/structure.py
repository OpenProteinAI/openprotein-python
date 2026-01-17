import gzip
import requests
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import gemmi

import openprotein.utils.cif as _cif_utils

from .complex import Complex


class Structure:
    """Represents a collection of :class:`Complex` instances."""

    def __init__(self, complexes: Sequence[Complex], name: bytes | str | None = None):
        self._complexes = list(complexes)
        self.name = name

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, x: bytes | str | None) -> None:
        self._name = x.decode() if isinstance(x, bytes) else x

    def get_name(self) -> str | None:
        return self._name

    def set_name(self, x: bytes | str | None) -> "Structure":
        self.name = x
        return self

    def __len__(self) -> int:
        return len(self._complexes)

    def __getitem__(self, key: int) -> Complex:
        assert isinstance(key, int)
        return self._complexes[key]

    def __setitem__(self, key: int, value: Complex) -> None:
        assert isinstance(key, int) and isinstance(value, Complex)
        self._complexes[key] = value

    def add_complex(self, complex: Complex) -> "Structure":
        self._complexes.append(complex)
        return self

    def to_string(self, format: Literal["cif", "pdb"] = "cif") -> str:
        """
        Serialize this Structure to a string. Note that format="pdb" may not serialize
        all aspects of this object, so format="cif", the default, is preferred.
        """
        if format == "cif":
            return self._make_cif_string()
        elif format == "pdb":
            return self._make_pdb_string()
        else:
            raise ValueError(format)

    @staticmethod
    def from_filepath(
        path: Path | str, use_bfactor_as_plddt: bool | None = None, verbose: bool = True
    ) -> "Structure":
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
        return Structure.from_string(
            filestring=data,
            format=format,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            verbose=verbose,
        ).set_name(path.name.removesuffix(ext))

    @staticmethod
    def from_string(
        filestring: bytes | str,
        format: Literal["pdb", "cif"],
        use_bfactor_as_plddt: bool | None = None,
        verbose: bool = True,
    ) -> "Structure":
        structure_block = _cif_utils.StructureCIFBlock(
            filestring=filestring, format=format
        )
        return Structure(
            [
                Complex._from_structure_block(
                    structure_block=structure_block,
                    use_bfactor_as_plddt=use_bfactor_as_plddt,
                    model_idx=model_idx,
                    verbose=verbose,
                )
                # NB: if no models in structure, try to "read" first model which will
                #     try to construct a model based solely on entities
                for model_idx in range(max(len(structure_block.structure), 1))
            ],
            name=structure_block.structure.name,
        )

    @staticmethod
    def from_pdb_id(pdb_id: str, verbose: bool = True) -> "Structure":
        """
        Creates a Structure instance by downloading data from the RCSB PDB.

        This method performs an HTTP GET request to the RCSB web server to fetch
        the structure file (in CIF format) associated with the given PDB ID.

        Args:
            pdb_id (str): The 4-character PDB identifier (e.g. "1XYZ").
            verbose (bool, optional): Whether to print warnings to stdout. Defaults to
                True.

        Returns:
            Structure: A new instance containing the parsed structure data.

        Raises:
            requests.HTTPError: If the PDB ID is invalid, the server is unreachable,
                or the request returns a 404/500 status code.
        """
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        response = requests.get(url)
        response.raise_for_status()
        return Structure.from_string(
            filestring=response.content,
            format="cif",
            use_bfactor_as_plddt=False,
            verbose=verbose,
        ).set_name(pdb_id)

    def copy(self) -> "Structure":
        return Structure(
            [complex.copy() for complex in self._complexes], name=self._name
        )

    def _make_cif_string(self) -> str:
        structure = self._make_structure()
        block = structure.make_mmcif_block(
            groups=gemmi.MmcifOutputGroups(True, chem_comp=False)
        )
        if len(self._complexes) == 0:
            return block.as_string()
        # add additional loops for first complex
        sequence_loop, atom_loop = _cif_utils.init_loops(block=block)
        for chain_id, chain in self._complexes[0].get_chains().items():
            chain._append_loop_data(
                chain_id=chain_id, sequence_loop=sequence_loop, atom_loop=atom_loop
            )
        sequence_loop_length = sequence_loop.length()
        atom_loop_length = atom_loop.length()
        # we don't support complexes other than the first complex having additional loop
        # data, so we assert that here
        for complex in self._complexes[1:]:
            for chain_id, chain in complex.get_chains().items():
                chain._append_loop_data(
                    chain_id=chain_id, sequence_loop=sequence_loop, atom_loop=atom_loop
                )
        if (
            sequence_loop.length() != sequence_loop_length
            or atom_loop.length() != atom_loop_length
        ):
            raise NotImplementedError(
                "cannot serialize multiple models with additional loop data yet"
            )
        return block.as_string()

    def _make_pdb_string(self) -> str:
        structure = self._make_structure()
        return structure.make_pdb_string(gemmi.PdbWriteOptions(minimal=True))

    def _make_structure(self) -> gemmi.Structure:
        if len(self._complexes) == 0:
            return gemmi.Structure()
        structures: list[gemmi.Structure] = []
        for complex in self._complexes:
            structures.append(complex._make_structure())
        first_entities = structures[0].entities
        for structure in structures[1:]:
            for first_entity, this_entity in zip(
                first_entities, structure.entities, strict=True
            ):
                assert this_entity.name == first_entity.name
                assert this_entity.subchains == first_entity.subchains
                assert this_entity.entity_type == first_entity.entity_type
                assert this_entity.polymer_type == first_entity.polymer_type
                assert this_entity.sifts_unp_acc == first_entity.sifts_unp_acc
                assert this_entity.full_sequence == first_entity.full_sequence
        if sum(_structure_has_no_atoms(structure) for structure in structures) > 0:
            assert (
                len(structures) == 1
            ), "can only serialize Structures containing Complexes with no structure data if there is only one complex in the structure"
        structure = structures[0]
        for this_structure in structures[1:]:
            structure.add_model(this_structure[0])
        structure.renumber_models()
        if self._name is not None:
            structure.name = self._name
        return structure


def _structure_has_no_atoms(structure: gemmi.Structure) -> bool:
    if len(structure) == 0:
        return True
    assert len(structure) == 1
    model = structure[0]
    try:
        next(iter(model.all()))
        return False
    except StopIteration:
        return True
