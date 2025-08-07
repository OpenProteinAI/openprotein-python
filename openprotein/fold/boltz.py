"""Community-based Boltz models for complex structure prediction with ligands/dna/rna."""

import re
import string
from typing import Any

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.chains import DNA, RNA, Ligand
from openprotein.common import ModelMetadata
from openprotein.protein import Protein

from . import api
from .future import FoldComplexResultFuture
from .models import FoldModel

valid_id_pattern = re.compile(r"^[A-Z]{1,5}$|^\d{1,5}$")


def is_valid_id(id_str: str) -> bool:
    """
    Check if the id_str matches the valid pattern for IDs (1-5 uppercase or 1-5 digits).
    """
    if not id_str or len(id_str) > 5:
        return False
    return bool(valid_id_pattern.fullmatch(id_str))


def id_generator(used_ids: list[str] | None = None, max_alpha_len=5, max_numeric=99999):
    """
    Yields new chain IDs, skipping any in 'used_ids'.
    First A..Z, AA..ZZ, … up to max_alpha_len, then '1','2',… up to max_numeric.
    """
    used = set(tuple(used_ids or []))
    letters = list(string.ascii_uppercase)

    # --- Alphabetic IDs ---
    curr_len = 1
    curr_indices = [0] * curr_len  # start at 'A'

    def bump_indices():
        # lexicographically increment curr_indices; return False on overflow
        for i in reversed(range(len(curr_indices))):
            if curr_indices[i] < len(letters) - 1:
                curr_indices[i] += 1
                for j in range(i + 1, len(curr_indices)):
                    curr_indices[j] = 0
                return True
        return False

    while curr_len <= max_alpha_len:
        candidate = "".join(letters[i] for i in curr_indices)
        if candidate not in used:
            used.add(candidate)
            yield candidate
        # bump
        if not bump_indices():
            curr_len += 1
            if curr_len > max_alpha_len:
                break
            curr_indices = [0] * curr_len

    # --- Numeric IDs ---
    num = 1
    while num <= max_numeric:
        candidate = str(num)
        num += 1
        if candidate not in used:
            used.add(candidate)
            yield candidate

    # exhausted
    raise RuntimeError("exhausted all possible IDs")


class BoltzModel(FoldModel):
    """
    Class providing inference endpoints for Boltz structure prediction models.
    """

    model_id: str = "boltz"

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session, model_id, metadata)

    def fold(
        self,
        proteins: list[Protein] | MSAFuture | None = None,
        dnas: list[DNA] | None = None,
        rnas: list[RNA] | None = None,
        ligands: list[Ligand] | None = None,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
        **kwargs,
    ) -> FoldComplexResultFuture:
        """
        Request structure prediction with boltz model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture | None
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        dna : List[DNA] | None
            List of DNA sequences to include in folded output.
        rna : List[RNA] | None
            List of RNA sequences to include in folded output.
        ligands : List[Ligand] | None
            List of ligands to include in folded output.
        diffusion_samples: int
            Number of diffusion samples to use
        recycling_steps : int
            Number of recycling steps to use
        sampling_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldComplexResultFuture
            Future for the folding complex result.
        """
        # validate constraints
        if constraints is not None:
            TypeAdapter(list[BoltzConstraint]).validate_python(constraints)
        # collate the id's used
        used_ids = []
        if isinstance(proteins, list):
            for protein in proteins:
                if isinstance(protein, Protein) and protein.chain_id is not None:
                    if isinstance(protein.chain_id, str):
                        used_ids.append(protein.chain_id)
                    elif isinstance(protein.chain_id, list):
                        used_ids.extend(protein.chain_id)
        for dna in dnas or []:
            if isinstance(dna.chain_id, str):
                used_ids.append(dna.chain_id)
            elif isinstance(dna.chain_id, list):
                used_ids.extend(dna.chain_id)
        for rna in rnas or []:
            if isinstance(rna.chain_id, str):
                used_ids.append(rna.chain_id)
            elif isinstance(rna.chain_id, list):
                used_ids.extend(rna.chain_id)
        for ligand in ligands or []:
            if isinstance(ligand.chain_id, str):
                used_ids.append(ligand.chain_id)
            elif isinstance(ligand.chain_id, list):
                used_ids.extend(ligand.chain_id)
        id_gen = id_generator(used_ids)
        # build the proteins from msa
        if isinstance(proteins, MSAFuture):
            align_api = getattr(self.session, "align", None)
            assert isinstance(align_api, AlignAPI)
            msa = proteins  # rename
            proteins = []  # convert back to list of proteins
            seed = align_api.get_seed(job_id=msa.job.job_id)
            query_seqs_cardinality: dict[str, int] = dict()
            for seq in seed.split(":"):
                query_seqs_cardinality[seq] = query_seqs_cardinality.get(seq, 0) + 1
            for seq, card in query_seqs_cardinality.items():
                protein = Protein(sequence=seq)
                if card == 1:
                    id = next(id_gen)
                else:
                    id = [next(id_gen) for _ in range(card)]
                protein.chain_id = id
                protein.msa = msa
                proteins.append(protein)

        # build the sequences input
        sequences: list[dict[str, Any]] = []
        for protein in proteins or []:
            # check the msa
            msa = protein.msa
            if msa is None:
                raise ValueError(
                    "Expected all protein sequences to have `.msa` set with an `MSAFuture` or `Protein.single_sequence_mode` for single sequence mode."
                )
            # convert to msa id or null for single sequence mode
            msa_id = (
                msa
                if isinstance(msa, str)
                else msa.id if isinstance(msa, MSAFuture) else None
            )
            # add the protein in the expected boltz format
            p = {
                "id": protein.chain_id or next(id_gen),
                "msa_id": msa_id,
                "sequence": protein.sequence.decode(),
            }
            if protein.cyclic:
                p["cyclic"] = protein.cyclic
            sequences.append({"protein": p})
        for dna in dnas or []:
            d = {
                "id": dna.chain_id or next(id_gen),
                "sequence": dna.sequence,
            }
            if dna.cyclic:
                d["cyclic"] = dna.cyclic
            sequences.append(
                {
                    "dna": d,
                }
            )
        for rna in rnas or []:
            r = {
                "id": rna.chain_id or next(id_gen),
                "sequence": rna.sequence,
            }
            if rna.cyclic:
                r["cyclic"] = rna.cyclic
            sequences.append(
                {
                    "rna": r,
                }
            )
        for ligand in ligands or []:
            ligand_: dict = {"id": ligand.chain_id or next(id_gen)}
            if ligand.ccd:
                ligand_["ccd"] = ligand.ccd
            if ligand.smiles:
                ligand_["smiles"] = ligand.smiles
            sequences.append({"ligand": ligand_})

        if len(sequences) == 0:
            raise ValueError("Expected proteins, dna, rna or ligands")

        return FoldComplexResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=sequences,
                diffusion_samples=diffusion_samples,
                recycling_steps=recycling_steps,
                sampling_steps=sampling_steps,
                step_scale=step_scale,
                constraints=constraints,
                use_potentials=use_potentials,
                **kwargs,
            ),
            model_id=self.model_id,
            proteins=proteins,
            dnas=dnas,
            rnas=rnas,
            ligands=ligands,
        )


class Boltz2Model(BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-2 structure prediction model which jointly models complex structures and binding affinities.
    """

    model_id = "boltz-2"

    def fold(
        self,
        proteins: list[Protein] | MSAFuture | None = None,
        dnas: list[DNA] | None = None,
        rnas: list[RNA] | None = None,
        ligands: list[Ligand] | None = None,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
        templates: list[dict] | None = None,
        properties: list[dict] | None = None,
        method: str | None = None,
    ) -> FoldComplexResultFuture:
        """
        Request structure prediction with Boltz-2 model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture | None
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        dna : List[DNA] | None
            List of DNA sequences to include in folded output.
        rna : List[RNA] | None
            List of RNA sequences to include in folded output.
        ligands : List[Ligand] | None
            List of ligands to include in folded output.
        diffusion_samples: int
            Number of diffusion samples to use
        recycling_steps : int
            Number of recycling steps to use
        sampling_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        use_potentials: bool = False.
            Whether or not to use potentials.
        constraints : list[dict] | None = None
            List of constraints.
        templates: list[dict] | None = None
            List of templates to use for structure prediction.
        properties: list[dict] | None = None
            List of additional properties to predict. Should match the `BoltzProperties`
        method: str | None
            The experimental method or supervision source used for the prediction. Defults to None.
            Supported values (case-insensitive) include:
            'MD', 'X-RAY DIFFRACTION', 'ELECTRON MICROSCOPY', 'SOLUTION NMR',
            'SOLID-STATE NMR', 'NEUTRON DIFFRACTION', 'ELECTRON CRYSTALLOGRAPHY',
            'FIBER DIFFRACTION', 'POWDER DIFFRACTION', 'INFRARED SPECTROSCOPY',
            'FLUORESCENCE TRANSFER', 'EPR', 'THEORETICAL MODEL',
            'SOLUTION SCATTERING', 'OTHER', 'AFDB', 'BOLTZ-1'.
            View the documentation on Boltz for upstream details.

        Returns
        -------
        FoldComplexResultFuture
            Future for the folding result.
        """

        if templates is not None:
            raise ValueError("`templates` not yet supported!")

        # validate properties
        if properties is not None:
            props = TypeAdapter(list[BoltzProperty]).validate_python(properties)
            # Only allow affinity for ligands, and check binder refers to a ligand chain_id (str, not list)
            ligand_chain_ids = set()
            if ligands:
                for ligand in ligands:
                    if isinstance(ligand.chain_id, str):
                        ligand_chain_ids.add(ligand.chain_id)
                    elif isinstance(ligand.chain_id, list):
                        raise ValueError(
                            f"Ligand {ligand} has multiple chain_ids ({ligand.chain_id}); only single (str) chain_id allowed for affinity."
                        )
            for prop in props:
                if hasattr(prop, "affinity") and prop.affinity is not None:
                    binder_id = prop.affinity.binder
                    if binder_id not in ligand_chain_ids:
                        raise ValueError(
                            f"Affinity property binder '{binder_id}' does not match any ligand chain_id (must be a ligand with a single chain_id)."
                        )

        return super().fold(
            proteins=proteins,
            dnas=dnas,
            rnas=rnas,
            ligands=ligands,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            constraints=constraints,
            templates=templates,
            properties=properties,
            method=method,
        )


class Boltz1xModel(BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-1x open-source structure prediction model, which adds the use of inference potentials to improve performance.
    """

    model_id = "boltz-1x"

    def fold(
        self,
        proteins: list[Protein] | MSAFuture | None = None,
        dnas: list[DNA] | None = None,
        rnas: list[RNA] | None = None,
        ligands: list[Ligand] | None = None,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        step_scale: float = 1.638,
        constraints: list[dict] | None = None,
    ) -> FoldComplexResultFuture:
        """
        Request structure prediction with Boltz-1x model. Uses potentials with Boltz-1 model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture | None
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        dna : List[DNA] | None
            List of DNA sequences to include in folded output.
        rna : List[RNA] | None
            List of RNA sequences to include in folded output.
        ligands : List[Ligand] | None
            List of ligands to include in folded output.
        diffusion_samples: int
            Number of diffusion samples to use
        recycling_steps : int
            Number of recycling steps to use
        sampling_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldComplexResultFuture
            Future for the folding complex result.
        """

        return super().fold(
            proteins=proteins,
            dnas=dnas,
            rnas=rnas,
            ligands=ligands,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            step_scale=step_scale,
            use_potentials=True,
            constraints=constraints,
        )


class Boltz1Model(BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-1 open-source structure prediction model.
    """

    model_id = "boltz-1"

    def fold(
        self,
        proteins: list[Protein] | MSAFuture | None = None,
        dnas: list[DNA] | None = None,
        rnas: list[RNA] | None = None,
        ligands: list[Ligand] | None = None,
        diffusion_samples: int = 1,
        recycling_steps: int = 3,
        sampling_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
    ) -> FoldComplexResultFuture:
        """
        Request structure prediction with Boltz-1 model.

        Parameters
        ----------
        proteins : List[Protein] | MSAFuture | None
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        dna : List[DNA] | None
            List of DNA sequences to include in folded output.
        rna : List[RNA] | None
            List of RNA sequences to include in folded output.
        ligands : List[Ligand] | None
            List of ligands to include in folded output.
        diffusion_samples: int
            Number of diffusion samples to use
        recycling_steps : int
            Number of recycling steps to use
        sampling_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        use_potentials: bool = False.
            Whether or not to use potentials.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldComplexResultFuture
            Future for the folding complex result.
        """

        return super().fold(
            proteins=proteins,
            dnas=dnas,
            rnas=rnas,
            ligands=ligands,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            constraints=constraints,
        )


class BondConstraint(BaseModel):
    """
    Constraint specifying a covalent bond between two atoms.

    Attributes
    ----------
    atom1 : list of (str or int)
        The first atom, specified as [CHAIN_ID, RES_IDX, ATOM_NAME].
    atom2 : list of (str or int)
        The second atom, specified as [CHAIN_ID, RES_IDX, ATOM_NAME].
    """

    atom1: list[str | int]
    atom2: list[str | int]


class PocketConstraint(BaseModel):
    """
    Constraint specifying a ligand pocket.

    Attributes
    ----------
    binder : str
        The chain ID of the binder.
    contacts : list of list of (str or int)
        List of contacts, each specified as [CHAIN_ID, RES_IDX/ATOM_NAME].
    max_distance : float
        Maximum distance in angstroms for the pocket constraint.
    """

    binder: str
    contacts: list[list[str | int]]
    max_distance: float


class ContactConstraint(BaseModel):
    """
    Constraint specifying a contact between two tokens.

    Attributes
    ----------
    token1 : list of (str or int)
        The first token, specified as [CHAIN_ID, RES_IDX/ATOM_NAME].
    token2 : list of (str or int)
        The second token, specified as [CHAIN_ID, RES_IDX/ATOM_NAME].
    max_distance : float
        Maximum distance in angstroms for the contact constraint.
    """

    token1: list[str | int]
    token2: list[str | int]
    max_distance: float


class BoltzConstraint(BaseModel):
    """
    Possible constraints for Boltz.

    Attributes
    ----------
    bond : BondConstraint or None, optional
        Covalent bond constraint.
    pocket : PocketConstraint or None, optional
        Pocket constraint.
    contact : ContactConstraint or None, optional
        Contact constraint.
    """

    bond: BondConstraint | None = None
    pocket: PocketConstraint | None = None
    contact: ContactConstraint | None = None

    @model_validator(mode="after")
    def check_exactly_one(cls, self):
        fields = [self.bond, self.pocket, self.contact]
        if sum(x is not None for x in fields) != 1:
            raise ValueError(
                "Exactly one of 'bond', 'pocket', or 'contact' must be set."
            )
        return self


class AffinityProperty(BaseModel):
    """
    Property specifying affinity computation.

    Attributes
    ----------
    binder : str
        The chain ID of the ligand for which to compute affinity.
    """

    binder: str


class BoltzProperty(BaseModel):
    """
    Properties (additionally) requested for computation.

    Attributes
    ----------
    affinity : AffinityProperty
        Affinity property specification.
    """

    # TODO handle more than more property
    affinity: AffinityProperty


class BoltzConfidence(BaseModel):
    """
    Model representing the aggregated confidence scores for a prediction sample.

    Attributes
    ----------
    confidence_score : float
        Aggregated score used to sort the predictions, corresponds to
        0.8 * complex_plddt + 0.2 * iptm (ptm for single chains).
    ptm : float
        Predicted TM score for the complex.
    iptm : float
        Predicted TM score when aggregating at the interfaces.
    ligand_iptm : float
        ipTM but only aggregating at protein-ligand interfaces.
    protein_iptm : float
        ipTM but only aggregating at protein-protein interfaces.
    complex_plddt : float
        Average pLDDT score for the complex.
    complex_iplddt : float
        Average pLDDT score when upweighting interface tokens.
    complex_pde : float
        Average PDE score for the complex.
    complex_ipde : float
        Average PDE score when aggregating at interfaces.
    chains_ptm : dict[str, float]
        Predicted TM score within each chain, keyed by chain index as a string.
    pair_chains_iptm : dict[str, dict[str, float]]
        Predicted (interface) TM score between each pair of chains,
        keyed by chain indices as strings.
    """

    confidence_score: float
    ptm: float
    iptm: float
    ligand_iptm: float
    protein_iptm: float
    complex_plddt: float
    complex_iplddt: float
    complex_pde: float
    complex_ipde: float
    chains_ptm: dict[str, float]
    pair_chains_iptm: dict[str, dict[str, float]]


class BoltzAffinity(BaseModel):
    """
    Output schema for Boltz affinity ensemble predictions.

    Attributes
    ----------
    affinity_pred_value : float
        Predicted binding affinity from the ensemble model.
    affinity_probability_binary : float
        Predicted binding likelihood from the ensemble model.
    per_model : dict of str to float
        Dictionary containing predictions from each individual model in the ensemble.
        Keys are of the form 'affinity_pred_valueN' and 'affinity_probability_binaryN',
        where N is the model index (e.g., 1, 2, 3, ...).

    Notes
    -----
    Use the `parse_obj_with_models` class method to construct this object from a raw output
    dictionary, which will automatically separate ensemble-level and per-model predictions.
    """

    affinity_pred_value: float
    affinity_probability_binary: float
    # Catch all other per-model fields
    per_model: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def parse_obj_with_models(cls, obj: dict):
        # Extract fixed fields
        fixed = {
            "affinity_pred_value": obj.pop("affinity_pred_value"),
            "affinity_probability_binary": obj.pop("affinity_probability_binary"),
        }
        # Everything else goes into per_model
        return cls(**fixed, per_model=obj)
