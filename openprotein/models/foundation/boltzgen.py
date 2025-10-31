"""BoltzGen model for protein structure and sequence design."""

from typing import Any, BinaryIO, Literal

from pydantic import BaseModel, Field

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.common.model_metadata import ModelDescription
from openprotein.jobs import Future, Job
from openprotein.models.base import ProteinModel
from openprotein.protein import Protein


class BoltzGenRequest(BaseModel):
    "Specification for an BoltzGen request."

    n: int = 1
    # protein: Protein
    structure_text: str | None = None
    design_spec: dict[str, Any]
    diffusion_batch_size: int | None = None
    step_scale: float | None = None
    noise_scale: float | None = None


class BoltzGenJob(Job):
    """Job schema for an BoltzGen request."""

    job_type: Literal["/models/boltzgen"]


class BoltzGenFuture(Future):
    """Future for handling the results of an BoltzGen job."""

    job: BoltzGenJob

    def get_pdb(self, replicate: int = 0) -> str:
        """
        Retrieve the PDB file for a specific design.

        Args:
            design_index (int): The 0-based index of the design to retrieve.

        Returns:
            str: The content of the PDB file as a string.
        """
        return _boltzgen_api_result_get(
            session=self.session, job_id=self.id, replicate=replicate
        )

    def get(self, replicate: int = 0):
        """Default result accessor, returns the first PDB."""
        # TODO handle different design index
        return self.get_pdb(replicate=replicate)


def _boltzgen_api_post(
    session: APISession, request: BoltzGenRequest, **kwargs
) -> BoltzGenJob:
    """
    POST a request for BoltzGen design.

    Returns a Job object that can be used to retrieve results later.
    """
    endpoint = "v1/design/models/boltzgen"
    body = request.model_dump(exclude_none=True)
    body.update(kwargs)
    response = session.post(endpoint, json=body)
    return BoltzGenJob.model_validate(response.json())


def _boltzgen_api_get_metadata(session: APISession) -> ModelMetadata:
    """
    POST a request for BoltzGen design.

    Returns a Job object that can be used to retrieve results later.
    """
    endpoint = f"v1/design/models/boltzgen"
    response = session.get(endpoint)
    return ModelMetadata.model_validate(response.json())


def _boltzgen_api_result_get(
    session: APISession, job_id: str, replicate: int = 0
) -> str:
    """
    POST a request for BoltzGen design.

    # Returns a Job object that can be used to retrieve results later.
    """
    endpoint = f"v1/design/{job_id}/results"
    response = session.get(endpoint, params={"replicate": replicate})
    return response.text


class BoltzGenModel(ProteinModel):
    """
    BoltzGen model for generating de novo protein structures.

    This model supports functionalities like unconditional design, scaffolding,
    and binder design.
    """

    model_id: str = "boltzgen"

    def __init__(self, session: APISession, model_id: str = "boltzgen"):
        # The model_id from the API might be more specific, e.g., "boltzgen-v1.1"
        super().__init__(session, model_id)

    def get_metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="boltzgen",
            description=ModelDescription(summary="BoltzGen"),
            dimension=0,
            output_types=["pdb"],
            input_tokens=[],
            token_descriptions=[[]],
        )

    def generate(
        self,
        design_spec: dict[str, Any],
        structure_file: str | bytes | BinaryIO | None = None,
        n: int = 1,
        diffusion_batch_size: int | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        **kwargs,
    ) -> BoltzGenFuture:
        """
        Run a protein structure generate job using BoltzGen.

        Parameters
        ----------
        design_spec : dict[str, Any]
            The BoltzGen design specification to run. This is the Python representation
            of the BoltzGen yaml request specification.
        structure_file : BinaryIO, optional
            An input PDB file (as a file-like object) used for inpainting or other
            guided design tasks where parts of an existing structure are provided.
        n : int, optional
            The number of unique design trajectories to run (default is 1).
        diffusion_batch_size : int, optional
            The batch size for diffusion sampling. Controls how many samples are
            processed in parallel during the diffusion process.
        step_scale : float, optional
            Scaling factor for the number of diffusion steps. Higher values may
            improve quality at the cost of longer generation time.
        noise_scale : float, optional
            Scaling factor for the noise schedule during diffusion. Controls the
            amount of noise added at each step of the reverse diffusion process.

        Other Parameters
        ----------------
        **kwargs : dict
            Additional keyword args that are passed directly to the boltzgen
            inference script. Overwrites any preceding options.

        Returns
        -------
        BoltzGenFuture
            A future object that can be used to retrieve the results of the design
            job upon completion.
        """
        request = BoltzGenRequest(
            n=n,
            design_spec=design_spec,
            diffusion_batch_size=diffusion_batch_size,
            step_scale=step_scale,
            noise_scale=noise_scale,
        )
        if structure_file is not None:
            if isinstance(structure_file, bytes):
                structure_text = structure_file.decode()
            elif isinstance(structure_file, str):
                structure_text = structure_file
            else:
                structure_text = structure_file.read().decode()
            request.structure_text = structure_text

        # Submit the job via the private API function
        job = _boltzgen_api_post(
            session=self.session,
            request=request,
            **kwargs,
        )

        # Return the future object
        return BoltzGenFuture(session=self.session, job=job)

    predict = generate
