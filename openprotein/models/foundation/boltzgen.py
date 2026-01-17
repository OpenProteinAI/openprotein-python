"""BoltzGen model for protein structure and sequence design."""

import base64
import gzip
import io
import tarfile
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.jobs import Future, Job
from openprotein.models.base import ProteinModel
from openprotein.models.structure_generation import StructureGenerationFuture
from openprotein.molecules import Protein, Complex
from openprotein.prompt import PromptAPI, Query
from openprotein.scaffolds import Scaffolds

from .boltzgen_schema import BoltzGenDesignSpec


def _create_assets_archive(
    scaffolds: dict[str, str | bytes | BinaryIO] | None = None,
    extra_files: dict[str, str | bytes | BinaryIO] | None = None,
) -> bytes | None:
    """
    Create a gzipped tar archive from scaffolds and extra files.

    Returns base64-encoded gzipped tar bytes, or None if no files provided.
    """
    if not scaffolds and not extra_files:
        return None

    # Create in-memory tar.gz
    tar_buffer = io.BytesIO()

    with gzip.GzipFile(fileobj=tar_buffer, mode="wb") as gz:
        with tarfile.open(fileobj=gz, mode="w") as tar:
            # Add scaffolds
            if scaffolds:
                for filename, content in scaffolds.items():
                    # Read content
                    if isinstance(content, bytes):
                        data = content
                    elif isinstance(content, str):
                        # Assume it's a file path
                        with open(content, "rb") as f:
                            data = f.read()
                    else:
                        # BinaryIO
                        data = content.read()

                    # Add to tar
                    info = tarfile.TarInfo(name=f"scaffolds/{filename}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

            # Add extra files
            if extra_files:
                for filename, content in extra_files.items():
                    # Read content
                    if isinstance(content, bytes):
                        data = content
                    elif isinstance(content, str):
                        with open(content, "rb") as f:
                            data = f.read()
                    else:
                        data = content.read()

                    # Add to tar
                    info = tarfile.TarInfo(name=f"extra/{filename}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))

    # Get the gzipped tar bytes
    tar_buffer.seek(0)
    return tar_buffer.read()


class BoltzGenRequest(BaseModel):
    "Specification for an BoltzGen request."

    N: int = 1
    query_id: str | None = None
    design_spec: BoltzGenDesignSpec | dict[str, Any] | None = None
    structure_text: str | None = None
    diffusion_batch_size: int | None = None
    step_scale: float | None = None
    noise_scale: float | None = None
    assets: str | None = None  # base64-encoded gzipped tar
    scaffold_set: str | None = None


class BoltzGenJob(Job):
    """Job schema for an BoltzGen request."""

    job_type: Literal["/models/boltzgen"]


class BoltzGenFuture(StructureGenerationFuture):
    """Future for handling the results of an RFdiffusion job."""

    job: BoltzGenJob

    def get_item(self, replicate: int = 0) -> Complex:
        """
        Retrieve the output Complex for a specific design.

        Args:
            replicate (int): The 0-based index of the design to retrieve.

        Returns:
            Complex: The designed Complex.
        """
        pdb = _boltzgen_api_result_get(
            session=self.session, job_id=self.id, replicate=replicate
        )
        m = Complex.from_string(pdb, format="cif")
        return m


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
        return _boltzgen_api_get_metadata(session=self.session)

    def generate(
        self,
        query: str | bytes | Protein | Complex | Query | None = None,
        design_spec: BoltzGenDesignSpec | dict[str, Any] | None = None,
        structure_file: str | bytes | BinaryIO | None = None,
        N: int = 1,
        diffusion_batch_size: int | None = None,
        step_scale: float | None = None,
        noise_scale: float | None = None,
        # scaffolds that can be provided for design
        scaffolds: dict[str, str | bytes | BinaryIO] | None = None,
        scaffold_set: Scaffolds | str | None = None,
        # extra structures that can be bundled together as assets
        extra_structure_files: dict[str, str | bytes | BinaryIO] | None = None,
        **kwargs,
    ) -> BoltzGenFuture:
        """
        Run a protein structure generate job using BoltzGen.

        Parameters
        ----------
        query : str or bytes or Protein or Complex or Query, optional
            A query representing the design specification. Either `query` or `design_spec`
            must be provided.
            `query` provides a unified way to represent design specifications on the
            OpenProtein platform. In this case, the structure mask of the containing Complex
            proteins are specified to be designed. Other parameters like binding, group,
            secondary structures, etc. are also passed through to BoltzGen.
        design_spec : BoltzGenDesignSpec | dict[str, Any] | None, optional
            The BoltzGen design specification to run. Either `query` or `design_spec`
            must be provided.
            `design_spec` exposes a low-level interface to using BoltzGen by accepting the YAML
            specification used by official BoltzGen examples.
            Can be a typed BoltzGenDesignSpec object or a dict representing the
            BoltzGen yaml request specification.
            Note: If the design_spec includes file paths, provide
            these extra files either using `scaffolds` or `extra_structure_files`.
        structure_file : str | bytes | BinaryIO | None, optional
            (Deprecated: use `extra_structure_files`)
            An input PDB/CIF file used for inpainting or other guided design tasks
            where parts of an existing structure are provided. This parameter provides
            the actual structure content that corresponds to any FileEntity `path`
            fields in the design_spec. Can be:
            - A file path (str) to read from
            - Raw file content (bytes)
            - A file-like object (BinaryIO)
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
        scaffolds : dict[str, str | bytes | BinaryIO] | None, optional
            Dictionary mapping scaffold filenames to their content. Each value can be:
            - A file path (str) to read from
            - Raw file content (bytes)
            - A file-like object (BinaryIO)
            These files will be packaged into a gzipped tar archive and made available
            to the design process under the 'scaffolds/' directory.
        scaffold_set : Scaffolds | str | None, optional
            A pre-defined scaffold set object. Alternative to providing individual
            scaffold files via the `scaffolds` parameter.
        extra_structure_files : dict[str, str | bytes | BinaryIO] | None, optional
            Dictionary mapping additional structure filenames to their content, with
            the same format options as `scaffolds`. These files will be packaged into
            the same archive under the 'extra/' directory and can be referenced in
            the design specification.

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
        # Ensure only query or design_spec is provided
        if (query is None and design_spec is None) or (
            query is not None and design_spec is not None
        ):
            raise ValueError("Expected either `query` or `design_spec`")

        if query is not None:
            prompt_api = getattr(self.session, "prompt", None)
            assert isinstance(prompt_api, PromptAPI)
            query_id = prompt_api._resolve_query(
                query=query, force_structure=True
            )  # ensure we have a structure query
        else:
            query_id = None

        # Validate design_spec if it's a dict
        if isinstance(design_spec, dict):
            design_spec = BoltzGenDesignSpec.model_validate(design_spec)

        # Extract the string
        if isinstance(scaffold_set, Scaffolds):
            scaffold_set = scaffold_set.value

        request = BoltzGenRequest(
            N=N,
            query_id=query_id,
            design_spec=design_spec,
            diffusion_batch_size=diffusion_batch_size,
            step_scale=step_scale,
            noise_scale=noise_scale,
            scaffold_set=scaffold_set,
        )

        # Handle structure_file
        if structure_file is not None:
            raise ValueError(
                "structure_file no longer accepted. use extra_structure_files instead to provide multiple structure files."
            )

        # Create assets archive from scaffolds and extra files
        assets_bytes = _create_assets_archive(
            scaffolds=scaffolds, extra_files=extra_structure_files
        )
        if assets_bytes:
            request.assets = base64.b64encode(assets_bytes).decode("utf-8")

        # Submit the job
        job = _boltzgen_api_post(
            session=self.session,
            request=request,
            **kwargs,
        )

        return BoltzGenFuture(session=self.session, job=job, N=request.N)

    predict = generate
