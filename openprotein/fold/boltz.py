"""Community-based Boltz models for complex structure prediction with ligands/dna/rna."""

import warnings
from typing import Sequence

from pydantic import BaseModel, Field, TypeAdapter, model_validator

from openprotein.align import AlignAPI, MSAFuture
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.fold.common import normalize_inputs, serialize_input
from openprotein.molecules import Complex, Ligand, Protein

from . import api
from .complex import id_generator
from .future import FoldResultFuture
from .models import FoldModel


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
        sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 3,
        num_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
        **kwargs,
    ) -> FoldResultFuture:
        """
        Request structure prediction with boltz model.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldResultFuture
            Future for the folding complex result.
        """
        # migrate old parameter
        if (recycling_steps := kwargs.get("recycling_steps")) is not None:
            num_recycles = recycling_steps
            warnings.warn(
                "`recycling_steps` has been updated to `num_recycles`. The parameter will be auto-corrected for now but raise an exception in the future."
            )
        if (sampling_steps := kwargs.get("sampling_steps")) is not None:
            num_steps = sampling_steps
            warnings.warn(
                "`sampling_steps` has been updated to `num_steps`. The parameter will be auto-corrected for now but raise an exception in the future."
            )
        # validate constraints
        if constraints is not None:
            TypeAdapter(list[BoltzConstraint]).validate_python(constraints)

        # build the normalized_models from msa
        if isinstance(sequences, MSAFuture):
            id_gen = id_generator()
            align_api = getattr(self.session, "align", None)
            assert isinstance(align_api, AlignAPI)
            msa = sequences  # rename
            seed = align_api.get_seed(job_id=msa.job.job_id)
            _proteins: dict[str, Protein] = {}
            for seq in seed.split(":"):
                protein = Protein(sequence=seq)
                id = next(id_gen)
                protein.msa = msa.id
                _proteins[id] = protein
            normalized_complexes = [Complex(chains=_proteins)]

        else:
            normalized_complexes = normalize_inputs(sequences)

        _complexes = serialize_input(self.session, normalized_complexes, needs_msa=True)

        if len(_complexes) == 0:
            raise TypeError(
                "Expected either non-empty list of proteins/models/sequences or MSAFuture"
            )

        return FoldResultFuture(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=_complexes,
                diffusion_samples=diffusion_samples,
                num_recycles=num_recycles,
                num_steps=num_steps,
                step_scale=step_scale,
                constraints=constraints,
                use_potentials=use_potentials,
                **kwargs,
            ),
            complexes=normalized_complexes,
        )


class Boltz2Model(BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-2 structure prediction model which jointly models complex structures and binding affinities.
    """

    model_id = "boltz-2"

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 3,
        num_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
        templates: list[dict] | None = None,
        properties: list[dict] | None = None,
        method: str | None = None,
    ) -> FoldResultFuture:
        """
        Request structure prediction with Boltz-2 model.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
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
        FoldResultFuture
            Future for the folding result.
        """

        if templates is not None:
            raise ValueError("`templates` not yet supported!")

        # validate properties
        if properties is not None:
            props = TypeAdapter(list[BoltzProperty]).validate_python(properties)
            # Only allow affinity for ligands, and check binder refers to a ligand chain_id (str, not list)
            ligand_chain_ids = set()
            if isinstance(sequences, list):
                for protein in sequences:
                    if isinstance(protein, Complex):
                        complex = protein
                        for id, chain in complex.get_chains().items():
                            if isinstance(chain, Ligand):
                                ligand_chain_ids.add(id)
            for prop in props:
                if hasattr(prop, "affinity") and prop.affinity is not None:
                    binder_id = prop.affinity.binder
                    if binder_id not in ligand_chain_ids:
                        raise ValueError(
                            f"Affinity property binder '{binder_id}' does not match any ligand chain_id (must be a ligand with a single chain_id)."
                        )

        return super().fold(
            sequences=sequences,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            constraints=constraints,
            templates=templates,
            properties=properties,
            method=method,
        )


class Boltz1Model(BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-1 open-source structure prediction model.
    """

    model_id = "boltz-1"

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 3,
        num_steps: int = 200,
        step_scale: float = 1.638,
        use_potentials: bool = False,
        constraints: list[dict] | None = None,
    ) -> FoldResultFuture:
        """
        Request structure prediction with Boltz-1 model.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        use_potentials: bool = False.
            Whether or not to use potentials.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldResultFuture
            Future for the folding complex result.
        """
        if constraints is not None:
            pocket_constraints = []
            for constraint in constraints:
                if "contact" in constraint:
                    raise ValueError("Boltz-1(x) doesn't support contact constraints")

                if "pocket" in constraint:
                    pocket_constraint = constraint["pocket"]
                    if len(pocket_constraints) > 0:
                        msg = f"Only one pocket binders is supported in Boltz-1!"
                        raise ValueError(msg)

                    max_distance = constraint["pocket"].get("max_distance", 6.0)
                    if max_distance != 6.0:
                        msg = f"Max distance != 6.0 is not supported in Boltz-1!"
                        raise ValueError(msg)
                    pocket_constraints.append(pocket_constraint)

        return super().fold(
            sequences=sequences,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            step_scale=step_scale,
            use_potentials=use_potentials,
            constraints=constraints,
        )


class Boltz1xModel(Boltz1Model, BoltzModel, FoldModel):
    """
    Class providing inference endpoints for Boltz-1x open-source structure prediction model, which adds the use of inference potentials to improve performance.
    """

    model_id = "boltz-1x"

    def fold(
        self,
        sequences: Sequence[Complex | Protein | str | bytes] | MSAFuture,
        diffusion_samples: int = 1,
        num_recycles: int = 3,
        num_steps: int = 200,
        step_scale: float = 1.638,
        constraints: list[dict] | None = None,
    ) -> FoldResultFuture:
        """
        Request structure prediction with Boltz-1x model. Uses potentials with Boltz-1 model.

        Parameters
        ----------
        sequences : Sequence[Complex | Protein | str | bytes] | MSAFuture
            List of protein sequences to include in folded output. `Protein` objects must be tagged with an `msa`, which can be a `Protein.single_sequence_mode` for single sequence mode. Alternatively, supply an `MSAFuture` to use all query sequences as a multimer.
        diffusion_samples: int
            Number of diffusion samples to use
        num_recycles : int
            Number of recycling steps to use
        num_steps : int
            Number of sampling steps to use
        step_scale : float
            Scaling factor for diffusion steps.
        constraints : Optional[List[dict]]
            List of constraints.

        Returns
        -------
        FoldResultFuture
            Future for the folding complex result.
        """

        return super().fold(
            sequences=sequences,
            diffusion_samples=diffusion_samples,
            num_recycles=num_recycles,
            num_steps=num_steps,
            step_scale=step_scale,
            use_potentials=True,
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
    **kwargs:
        Extra keys of the form 'affinity_pred_valueN' and 'affinity_probability_binaryN',
        where N is the model index (e.g., 1, 2, 3, ...).
    """

    affinity_pred_value: float
    affinity_probability_binary: float

    class Config:
        extra = "allow"  # Allow extra fields
