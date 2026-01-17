import gzip
import operator
from collections.abc import Mapping, Sequence
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

import gemmi

import openprotein.utils.chain_id as _chain_id_utils
import openprotein.utils.cif as _cif_utils

from .chains import DNA, RNA, Ligand
from .protein import Protein


# TODO: deserialization note about plddt parsed per residue
class Complex:
    def __init__(
        self,
        chains: Mapping[str, Protein | DNA | RNA | Ligand] | None = None,
        name: bytes | str | None = None,
    ):
        self._chains = dict(sorted(chains.items())) if chains is not None else {}
        self.name = name

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, x: bytes | str | None) -> None:
        self._name = x.decode() if isinstance(x, bytes) else x

    def get_name(self) -> str | None:
        return self._name

    def set_name(self, x: bytes | str | None) -> "Complex":
        self.name = x
        return self

    def get_chains(self) -> Mapping[str, Protein | DNA | RNA | Ligand]:
        return MappingProxyType(self._chains)

    def get_proteins(self) -> Mapping[str, Protein]:
        return MappingProxyType(
            {k: v for k, v in self._chains.items() if isinstance(v, Protein)}
        )

    def get_protein(self, chain_id: str) -> Protein:
        chain = self._chains[chain_id]
        assert isinstance(chain, Protein)
        return chain

    def get_dnas(self) -> Mapping[str, DNA]:
        return MappingProxyType(
            {k: v for k, v in self._chains.items() if isinstance(v, DNA)}
        )

    def get_dna(self, chain_id: str) -> DNA:
        chain = self._chains[chain_id]
        assert isinstance(chain, DNA)
        return chain

    def get_rnas(self) -> Mapping[str, RNA]:
        return MappingProxyType(
            {k: v for k, v in self._chains.items() if isinstance(v, RNA)}
        )

    def get_rna(self, chain_id: str) -> RNA:
        chain = self._chains[chain_id]
        assert isinstance(chain, RNA)
        return chain

    def get_ligands(self) -> Mapping[str, Ligand]:
        return MappingProxyType(
            {k: v for k, v in self._chains.items() if isinstance(v, Ligand)}
        )

    def get_ligand(self, chain_id: str) -> Ligand:
        chain = self._chains[chain_id]
        assert isinstance(chain, Ligand)
        return chain

    def set_chain(
        self, chain_id: str, value: Protein | DNA | RNA | Ligand
    ) -> "Complex":
        self._chains[chain_id] = value
        self._chains = dict(sorted(self._chains.items()))
        return self

    def __rand__(self, left: "Complex | Protein | str") -> "Complex":
        if isinstance(left, str):
            left = Protein.from_expr(expr=left)
        return left & self

    def __and__(self, right: "Complex | Protein | str") -> "Complex":
        """Combine multiple objects into a single Complex."""

        assert (
            isinstance(right, Complex)
            or isinstance(right, Protein)
            or isinstance(right, str)
        )
        id_gen = _chain_id_utils.id_generator(list(self._chains.keys()))
        if isinstance(right, str):
            right = Protein.from_expr(right)
        if isinstance(right, Protein):
            self.set_chain(chain_id=next(id_gen), value=right)
        else:
            if (
                len(overlapping_chain_ids := self._chains.keys() & right._chains.keys())
                > 0
            ):
                raise ValueError(
                    f"Trying to combine two sets of chains with overlapping chain ids: {overlapping_chain_ids}"
                )
            self._chains = dict(sorted((self._chains | right._chains).items()))
        return self

    @overload
    def rmsd(
        self,
        tgt: "Complex",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: Literal[False] = False,
    ) -> float: ...

    @overload
    def rmsd(
        self,
        tgt: "Complex",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: Literal[True] = True,
    ) -> tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...

    def rmsd(
        self,
        tgt: "Complex",
        backbone_only: bool | str | Sequence[str] = False,
        return_transform: bool = False,
    ) -> float | tuple[float, npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        assert all(
            isinstance(v, Protein) for v in self._chains.values()
        ), "rmsd supported only for Protein chains, not supported for non-protein chains"
        assert all(
            isinstance(v, Protein) for v in tgt._chains.values()
        ), "rmsd supported only for Protein chains, not supported for non-protein chains"
        src_proteins, tgt_proteins = self.get_proteins(), tgt.get_proteins()
        assert tgt_proteins.keys() == src_proteins.keys()
        assert [len(x) for x in src_proteins.values()] == [
            len(x) for x in tgt_proteins.values()
        ]
        src_protein: Protein = reduce(operator.add, src_proteins.values())
        tgt_protein: Protein = reduce(operator.add, tgt_proteins.values())
        return src_protein.rmsd(
            tgt_protein,
            backbone_only=backbone_only,
            return_transform=return_transform,
        )

    def transform(
        self,
        R: npt.NDArray[np.floating] | None = None,
        t: npt.NDArray[np.floating] | None = None,
    ) -> "Complex":
        assert all(
            isinstance(v, Protein) for v in self._chains.values()
        ), "transform supported only for Protein chains, not supported for non-protein chains"
        for protein in self.get_proteins().values():
            protein.transform(R=R, t=t)
        return self

    def superimpose_onto(
        self, tgt: "Complex", backbone_only: bool | str | Sequence[str] = False
    ) -> "Complex":
        _, R, t = tgt.rmsd(self, backbone_only=backbone_only, return_transform=True)
        return self.transform(R=R, t=t)

    def to_string(self, format: Literal["cif", "pdb"] = "cif") -> str:
        """
        Serialize this Complex to a string. Note that format="pdb" may not serialize all
        aspects of this object, so format="cif", the default, is preferred.
        """
        if format == "cif":
            return self._make_cif_string()
        elif format == "pdb":
            return self._make_pdb_string()
        else:
            raise ValueError(format)

    @staticmethod
    def from_filepath(
        path: Path | str,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Complex":
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
        return Complex.from_string(
            filestring=data,
            format=format,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        ).set_name(path.name.removesuffix(ext))

    @staticmethod
    def from_string(
        filestring: bytes | str,
        format: Literal["pdb", "cif"],
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Complex":
        structure_block = _cif_utils.StructureCIFBlock(
            filestring=filestring, format=format
        )
        return Complex._from_structure_block(
            structure_block=structure_block,
            use_bfactor_as_plddt=use_bfactor_as_plddt,
            model_idx=model_idx,
            verbose=verbose,
        )

    def copy(self) -> "Complex":
        return Complex(
            chains={k: v.copy() for k, v in self._chains.items()}, name=self._name
        )

    @staticmethod
    def _from_structure_block(
        structure_block: _cif_utils.StructureCIFBlock,
        use_bfactor_as_plddt: bool | None = None,
        model_idx: int = 0,
        verbose: bool = True,
    ) -> "Complex":
        block, structure = structure_block.block, structure_block.structure
        model = structure[model_idx] if len(structure) > 0 else None
        # Use block info directly so that we can get chains with empty struct info
        subchain_ids = [x for x in block.find_loop("_struct_asym.id")]
        if len(subchain_ids) == 0 and model is not None:
            # Try to get actual chain IDs from the structure
            subchain_ids = [subchain.subchain_id() for subchain in model.subchains()]
        # collect chains
        chains = {}
        for subchain_id in sorted(subchain_ids):
            subchain = model.get_subchain(subchain_id) if model is not None else None
            # Get the entity for this chain to determine its type
            if subchain is not None and len(subchain) > 0:
                entity = structure.get_entity_of(subchain)
                if entity is None:
                    raise ValueError(f"Could not find entity for chain {subchain_id}")
            else:
                matching_entities = [
                    e for e in structure.entities if subchain_id in e.subchains
                ]
                assert len(matching_entities) == 1, (
                    f"expected only one entity to match {chain_id=}, but found "
                    f"{len(matching_entities)}: {matching_entities}"
                )
                entity = matching_entities[0]
                del matching_entities
            # Determine chain type based on entity type and polymer type
            if (entity_type := entity.entity_type) == gemmi.EntityType.Polymer:
                if structure.input_format == gemmi.CoorFormat.Pdb:
                    assert subchain_id.endswith("xp")
                    chain_id = subchain_id.removesuffix("xp")
                    assert chain_id not in chains
                else:
                    chain_id = subchain_id
                if (polymer_type := entity.polymer_type) in (
                    gemmi.PolymerType.PeptideL,
                    gemmi.PolymerType.PeptideD,
                ):
                    chains[chain_id] = Protein._from_structure_block(
                        structure_block=structure_block,
                        chain_id=subchain_id,
                        use_bfactor_as_plddt=use_bfactor_as_plddt,
                        model_idx=model_idx,
                        verbose=verbose,
                    )
                elif polymer_type == gemmi.PolymerType.Dna:
                    chains[chain_id] = DNA._from_structure_block(
                        structure_block=structure_block,
                        chain_id=subchain_id,
                        model_idx=model_idx,
                    )
                elif polymer_type == gemmi.PolymerType.Rna:
                    chains[chain_id] = RNA._from_structure_block(
                        structure_block=structure_block,
                        chain_id=subchain_id,
                        model_idx=model_idx,
                    )
                else:
                    # if verbose:
                    #     print(
                    #         f"Warning: Skipping unsupported polymer type {polymer_type} for chain {subchain_id}"
                    #     )
                    continue
            elif entity_type == gemmi.EntityType.NonPolymer:
                if structure.input_format == gemmi.CoorFormat.Pdb:
                    raise ValueError("ligands from pdb files not supported yet")
                chain_id = subchain_id
                assert (
                    structure.input_format != gemmi.CoorFormat.Pdb
                ), "ligands from pdb files not supported yet"
                chains[chain_id] = Ligand._from_structure_block(
                    structure_block=structure_block,
                    chain_id=subchain_id,
                    model_idx=model_idx,
                )
            elif entity_type == gemmi.EntityType.Water:
                continue
            else:
                # if verbose:
                #     print(
                #         f"Warning: Skipping unsupported entity type {entity_type} for chain {subchain_id}"
                #     )
                continue
        return Complex(chains=chains, name=structure.name)

    def _make_cif_string(self) -> str:
        structure = self._make_structure()
        block = structure.make_mmcif_block(
            groups=gemmi.MmcifOutputGroups(True, chem_comp=False)
        )
        sequence_loop, atom_loop = _cif_utils.init_loops(block=block)
        for chain_id, chain in self._chains.items():
            chain._append_loop_data(
                chain_id=chain_id, sequence_loop=sequence_loop, atom_loop=atom_loop
            )
        return block.as_string()

    def _make_pdb_string(self) -> str:
        structure = self._make_structure()
        return structure.make_pdb_string(gemmi.PdbWriteOptions(minimal=True))

    def _make_structure(self) -> gemmi.Structure:
        assert (
            len(set(x._structure_block for x in self.get_ligands().values())) <= 1
        ), "can only serialize ligands if they all originate from the same structure file"
        structure = gemmi.Structure()
        for chain_id, chain in self._chains.items():
            structure = chain._make_structure(
                structure=structure,
                model_idx=0,
                chain_id=chain_id,
                entity_name=str(len(structure.entities) + 1),
            )
        structure.setup_entities()  # this should deduplicate polymer entities
        for entity_idx, entity in enumerate(structure.entities):
            entity.name = str(entity_idx + 1)
        if self._name is not None:
            structure.name = self._name
        return structure
