"""RFdiffusion model for protein structure and sequence design."""

from typing import BinaryIO, Literal

from pydantic import BaseModel, Field

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.common.model_metadata import ModelDescription
from openprotein.jobs import Future, Job
from openprotein.jobs.futures import MappedFuture
from openprotein.jobs.jobs import JobsAPI
from openprotein.models.base import ProteinModel
from openprotein.models.structure_generation import StructureGenerationFuture
from openprotein.molecules import Protein, Complex
from openprotein.prompt import PromptAPI, Query


class Contig(BaseModel):
    """Defines a contig segment for protein design."""

    length: str = Field(..., description="Length range, e.g., '10-20' or '100'")
    chain: str | None = Field(None, description="Chain to sample from")


class Hotspot(BaseModel):
    """Specifies a hotspot residue constraint."""

    res_id: str = Field(
        ..., description="Residue identifier, e.g., 'A100' for chain A, residue 100"
    )


class RFdiffusionRequest(BaseModel):
    "Specification for an RFdiffusion request."

    N: int = 1
    query_id: str | None = None
    structure_text: str | None = None
    # contigs: list[Contig]
    contigs: str | None = None
    inpaint_seq: str | None = None
    provide_seq: str | None = None
    # hotspots: list[Hotspot]
    hotspot: str | None = None
    T: int | None = None
    partial_T: int | None = None
    use_active_site_model: bool | None = None
    use_beta_model: bool | None = None

    # Simplified symmetry options
    symmetry: Literal["cyclic", "dihedral", "tetrahedral"] | None = None
    order: int | None = None
    add_potential: bool | None = None

    # Fold conditioning
    scaffold_target_structure_text: str | None = None
    scaffold_target_use_struct: bool = False


class RFdiffusionJob(Job):
    """Job schema for an RFdiffusion request."""

    job_type: Literal["/models/rfdiffusion"]


class RFdiffusionFuture(StructureGenerationFuture):
    """Future for handling the results of an RFdiffusion job."""

    job: RFdiffusionJob

    def get_item(self, replicate: int = 0) -> Complex:
        """
        Retrieve the output Complex for a specific design.

        Args:
            replicate (int): The 0-based index of the design to retrieve.

        Returns:
            Complex: The designed Complex.
        """
        pdb = _rfdiffusion_api_result_get(
            session=self.session, job_id=self.id, replicate=replicate
        )
        m = Complex.from_string(pdb, format="pdb")
        return m


def _rfdiffusion_api_post(
    session: APISession, request: RFdiffusionRequest, **kwargs
) -> RFdiffusionJob:
    """
    POST a request for RFdiffusion design.

    Returns a Job object that can be used to retrieve results later.
    """
    endpoint = "v1/design/models/rfdiffusion"
    body = request.model_dump(exclude_none=True)
    body.update(kwargs)
    response = session.post(endpoint, json=body)
    return RFdiffusionJob.model_validate(response.json())


def _rfdiffusion_api_get_metadata(session: APISession) -> ModelMetadata:
    """
    POST a request for RFdiffusion design.

    Returns a Job object that can be used to retrieve results later.
    """
    endpoint = f"v1/design/models/rfdiffusion"
    response = session.get(endpoint)
    return ModelMetadata.model_validate(response.json())


def _rfdiffusion_api_result_get(
    session: APISession, job_id: str, replicate: int = 0
) -> str:
    """
    POST a request for RFdiffusion design.

    # Returns a Job object that can be used to retrieve results later.
    """
    endpoint = f"v1/design/{job_id}/results"
    response = session.get(endpoint, params={"replicate": replicate})
    return response.text


class RFdiffusionModel(ProteinModel):
    """
    RFdiffusion model for generating de novo protein structures.

    This model supports functionalities like unconditional design, scaffolding,
    and binder design.
    """

    model_id: str = "rfdiffusion"

    def __init__(self, session: APISession, model_id: str = "rfdiffusion"):
        # The model_id from the API might be more specific, e.g., "rfdiffusion-v1.1"
        super().__init__(session, model_id)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="rfdiffusion",
            description=ModelDescription(summary="RFdiffusion"),
            dimension=0,
            output_types=["pdb"],
            input_tokens=[],
            token_descriptions=[[]],
        )

    def generate(
        self,
        query: str | bytes | Protein | Complex | Query | None = None,
        contigs: int | str | None = None,
        structure_file: str | bytes | BinaryIO | None = None,
        N: int = 1,
        inpaint_seq: str | None = None,
        provide_seq: str | None = None,
        # binding sites
        hotspot: str | None = None,
        # diffusion timesteps
        T: int | None = None,
        partial_T: int | None = None,
        # model options
        use_active_site_model: bool | None = None,
        use_beta_model: bool | None = None,
        # Symmetry options
        symmetry: Literal["cyclic", "dihedral", "tetrahedral"] | None = None,
        order: int | None = None,
        add_potential: bool | None = None,
        # Fold conditioning
        # TODO: provide query as a way to do fold conditioning
        scaffold_target_structure_file: str | bytes | BinaryIO | None = None,
        scaffold_target_use_struct: bool = False,
        **kwargs,
    ) -> RFdiffusionFuture:
        """
        Run a protein structure generate job using RFdiffusion.

        Parameters
        ----------
        query : str or bytes or Protein or Complex or Query, optional
            A query representing the design specification. Use either `query` or `contigs`
            for default design. Or provide `scaffold_target_structure_file`
            for scaffold guided design.
            `query` provides a unified way to represent design specifications on the
            OpenProtein platform. In this case, the structure mask of the containing Complex
            proteins are specified to be designed. Other parameters like binding are passed
            as hotspots to RFdiffusion.
        contigs : int, str, optional
            Defines the lengths and connectivity of chain segments for the desired
            structure, specified in RFdiffusion's contig string format.
            Required for most design tasks. Example: 150, '10-20/A100-110/10-20' for a
            binder design.
        structure_file : BinaryIO, optional
            An input PDB file (as a file-like object) used for inpainting or other
            guided design tasks where parts of an existing structure are provided.
        n : int, optional
            The number of unique design trajectories to run (default is 1).
        inpaint_seq : str, optional
            A string specifying the regions in the input structure to mask for
            in-painting. Example: 'A1-A10/A30-40'.
        provide_seq : str, optional
            A string specifying which segments of the contig have a provided
            sequence. Example: 'A1-A10/A30-40'.
        hotspot : str, optional
            A string specifying hotspot residues to constrain during design,
            typically for functional sites. Example: 'A10,A12,A14'.
        T : int, optional
            The number of timesteps for the diffusion process.
        partial_T : int, optional
            The number of timesteps for partial diffusion.
        use_active_site_model : bool, optional
            If True, uses the active site model checkpoint, which has been finetuned to
            better keep very small motifs in place in the output for motif scaffolding
            (default is False).
        use_beta_model : bool, optional
            If True, uses the complex beta model checkpoint, which generates a
            greater diversity of topologies but has not been extensively
            experimentally validated (default is False).
        symmetry : {"cyclic", "dihedral", "tetrahedral"}, optional
            The type of symmetry to apply to the design.
        order : int, optional
            The order of the symmetry (e.g., 3 for C3 or D3 symmetry).
            Must be provided if `symmetry` is set.
        add_potential : bool, optional
            A flag to toggle an additional potential to guide the design.
            This defaults to true in the case of symmetric design.
        scaffold_target_structure_file : str, bytes, BinaryIO, optional
            A PDB file (which can be the text string or bytes or the file-like
            object) containing a scaffold structure to be used as a structural
            guide. It could also be used as a target when doing scaffold guided
            binder design with `scaffold_target_use_struct`.
        scaffold_target_use_struct : bool, optional
            Whether or not to use the provided scaffold structure as a target.
            Otherwise, it is used only as a topology guide.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword args that are passed directly to the rfdiffusion
            inference script. Overwrites any preceding options.

        Returns
        -------
        RFdiffusionFuture
            A future object that can be used to retrieve the results of the design
            job upon completion.
        """
        if query is None and contigs is None and scaffold_target_structure_file is None:
            raise ValueError(
                "Expected either `query`, `contigs` or `scaffold_target_structure_file`"
            )
        if query is not None:
            prompt_api = getattr(self.session, "prompt", None)
            assert isinstance(prompt_api, PromptAPI)
            query_id = prompt_api._resolve_query(
                query=query, force_structure=True
            )  # ensure we have a structure query
        else:
            query_id = None

        if isinstance(contigs, int):
            contigs = f"{contigs}-{contigs}"
        request = RFdiffusionRequest(
            N=N,
            query_id=query_id,
            contigs=contigs,
            inpaint_seq=inpaint_seq,
            provide_seq=provide_seq,
            hotspot=hotspot,
            T=T,
            partial_T=partial_T,
            use_active_site_model=use_active_site_model,
            use_beta_model=use_beta_model,
            symmetry=symmetry,
            order=order,
            add_potential=add_potential,
            scaffold_target_use_struct=scaffold_target_use_struct,
        )
        if structure_file is not None:
            if isinstance(structure_file, bytes):
                structure_text = structure_file.decode()
            elif isinstance(structure_file, str):
                structure_text = structure_file
            else:
                structure_text = structure_file.read().decode()
            request.structure_text = structure_text
        if scaffold_target_structure_file is not None:
            if isinstance(scaffold_target_structure_file, bytes):
                scaffold_target_structure_text = scaffold_target_structure_file.decode()
            elif isinstance(scaffold_target_structure_file, str):
                scaffold_target_structure_text = scaffold_target_structure_file
            else:
                scaffold_target_structure_text = (
                    scaffold_target_structure_file.read().decode()
                )
            request.scaffold_target_structure_text = scaffold_target_structure_text

        # Submit the job via the private API function
        job = _rfdiffusion_api_post(
            session=self.session,
            request=request,
            **kwargs,
        )

        # Return the future object
        return RFdiffusionFuture(session=self.session, job=job, N=request.N)

    predict = generate
