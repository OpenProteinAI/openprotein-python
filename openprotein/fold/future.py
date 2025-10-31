"""Fold prediction results represented as futures."""

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic.type_adapter import TypeAdapter
from typing_extensions import Self

from openprotein import config
from openprotein.base import APISession
from openprotein.chains import DNA, RNA, Ligand
from openprotein.jobs import Future, JobsAPI, MappedFuture
from openprotein.protein import Protein

from . import api
from .schemas import FoldJob, FoldMetadata

if TYPE_CHECKING:
    from .boltz import BoltzAffinity, BoltzConfidence


class FoldResultFuture(MappedFuture, Future):
    """
    Fold results represented as a future.

    Attributes
    ----------
    job : FoldJob
        The fold job associated with this future.
    """

    job: FoldJob

    def __init__(
        self,
        session: APISession,
        job: FoldJob | None = None,
        metadata: FoldMetadata | None = None,
        sequences: list[bytes] | None = None,
        max_workers: int = config.MAX_CONCURRENT_WORKERS,
    ):
        """
        Initialize a FoldResultFuture instance.

        Takes in either a fold job, or the fold job metadata.

        :meta private:
        """
        # initialize the fold job metadata
        if metadata is None:
            if job is None or job.job_id is None:
                raise ValueError("Expected fold metadata or job")
            metadata = api.fold_get(session=session, job_id=job.job_id)
        self._metadata = metadata
        if job is None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = FoldJob.create(jobs_api.get_job(job_id=metadata.job_id))
        if sequences is None:
            sequences = api.fold_get_sequences(session=session, job_id=job.job_id)
        self._sequences = sequences
        super().__init__(session, job, max_workers)

    @classmethod
    def create(
        cls: type[Self],
        session: APISession,
        job: FoldJob | None = None,
        metadata: FoldMetadata | None = None,
        **kwargs,
    ) -> "Self | FoldComplexResultFuture":
        """
        Factory method to create a FoldResultFuture or FoldComplexResultFuture.

        Parameters
        ----------
        session : APISession
            The API session to use for requests.
        job : FoldJob
            The fold job associated with this future.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        FoldResultFuture or FoldComplexResultFuture
            An instance of FoldResultFuture or FoldComplexResultFuture depending on the model.
        """
        if job is not None:
            job_id = job.job_id
        elif metadata is not None:
            job_id = metadata.job_id
        else:
            raise ValueError("Expected fold metadata or job")
        model_id = api.fold_get(session=session, job_id=job_id).model_id
        if (
            model_id.startswith("boltz")
            or model_id.startswith("alphafold")
            or model_id.startswith("rosettafold")
        ):
            return FoldComplexResultFuture(session=session, job=job, **kwargs)
        else:
            return cls(session=session, job=job, **kwargs)

    @property
    def sequences(self) -> list[bytes]:
        """
        Get the sequences submitted for the fold request.

        Returns
        -------
        list[bytes]
            List of sequences.
        """
        if self._sequences is None:
            self._sequences = api.fold_get_sequences(self.session, self.job.job_id)
        return self._sequences

    @property
    def id(self):
        """
        Get the ID of the fold request.

        Returns
        -------
        str
            Fold job ID.
        """
        return self.job.job_id

    @property
    def metadata(self) -> FoldMetadata:
        """The fold metadata."""
        return self._metadata

    @property
    def model_id(self) -> str:
        """The fold model used."""
        return self._metadata.model_id

    def __keys__(self):
        """
        Get the list of sequences submitted for the fold request.

        Returns
        -------
        list of bytes
            List of sequences.
        """
        return self.sequences

    def get(self, verbose=False) -> list[tuple[str, bytes]]:
        """
        Retrieve the fold results as a list of tuples mapping sequence to PDB-encoded string.

        Parameters
        ----------
        verbose : bool, optional
            If True, print verbose output. Default is False.

        Returns
        -------
        list[tuple[str, str]]
            List of tuples mapping sequence to PDB-encoded string.
        """
        return super().get(verbose=verbose)

    def get_item(self, sequence: bytes) -> bytes:
        """
        Get fold results for a specified sequence.

        Parameters
        ----------
        sequence : bytes
            Sequence to fetch results for.

        Returns
        -------
        bytes
            Fold result for the specified sequence.
        """
        data = api.fold_get_sequence_result(self.session, self.job.job_id, sequence)
        return data


class FoldComplexResultFuture(Future):
    """
    Future for manipulating results of a fold complex request.

    Attributes
    ----------
    job : FoldJob
        The fold job associated with this future.
    """

    job: FoldJob

    def __init__(
        self,
        session: APISession,
        job: FoldJob | None = None,
        metadata: FoldMetadata | None = None,
        model_id: str | None = None,
        proteins: list[Protein] | None = None,
        ligands: list[Ligand] | None = None,
        dnas: list[DNA] | None = None,
        rnas: list[RNA] | None = None,
    ):
        """
        Initialize a FoldComplexResultFuture instance.

        Parameters
        ----------
        session : APISession
            The API session to use for requests.
        job : FoldJob
            The fold job associated with this future.
        model_id : str, optional
            Model ID used for the fold request.
        proteins : list[Protein], optional
            List of proteins submitted for fold request.
        ligands : list[Ligand], optional
            List of ligands submitted for fold request.
        dnas : list[DNA], optional
            List of DNAs submitted for fold request.
        rnas : list[RNA], optional
            List of RNAs submitted for fold request.
        """
        # initialize the fold job metadata
        if metadata is None:
            if job is None or job.job_id is None:
                raise ValueError("Expected fold metadata or job")
            metadata = api.fold_get(session, job.job_id)
        self._metadata = metadata
        if job is None:
            jobs_api = getattr(session, "jobs", None)
            assert isinstance(jobs_api, JobsAPI)
            job = FoldJob.create(jobs_api.get_job(job_id=metadata.job_id))
        super().__init__(session, job)
        self._model_id = model_id
        self._proteins = proteins
        self._ligands = ligands
        self._dnas = dnas
        self._rnas = rnas
        self._initialized = not (proteins == ligands == dnas == rnas == None)
        self._pae: np.ndarray | None = None
        self._pde: np.ndarray | None = None
        self._plddt: np.ndarray | None = None
        self._score: pd.DataFrame | None = None
        self._metrics: pd.DataFrame | None = None
        self._confidence: list["BoltzConfidence"] | None = None
        self._affinity: "BoltzAffinity | None" = None

    @property
    def metadata(self) -> FoldMetadata:
        """The fold metadata."""
        return self._metadata

    @property
    def model_id(self) -> str:
        """
        Get the model ID used for the fold request.

        Returns
        -------
        str
            Model ID.
        """
        if self._model_id is None:
            self._model_id = api.fold_get(
                session=self.session, job_id=self.job.job_id
            ).model_id
        return self._model_id

    def __get_chains(self):
        """
        Internal method to initialize chain objects (proteins, dnas, rnas, ligands)
        from the fold job arguments.
        """
        args = api.fold_get(session=self.session, job_id=self.job.job_id).args
        assert args is not None and "sequences" in args
        for chain in args["sequences"]:
            assert isinstance(chain, dict)
            for chain_type, chain_info in chain:
                if chain_type == "protein":
                    self._proteins = self._proteins or []
                    protein = Protein(sequence=chain_info["sequence"])
                    protein.chain_id = chain_info.get("id")
                    protein.msa = chain_info.get("msa_id")
                    self._proteins.append(protein)
                elif chain_type == "dna":
                    self._dnas = self._dnas or []
                    dna = DNA(sequence=chain_info["sequence"])
                    dna.chain_id = chain_info.get("id")
                    self._dnas.append(dna)
                elif chain_type == "rna":
                    self._rnas = self._rnas or []
                    rna = RNA(sequence=chain_info["sequence"])
                    rna.chain_id = chain_info.get("id")
                    self._rnas.append(rna)
                elif chain_type == "ligand":
                    self._ligands = self._ligands or []
                    ligand = Ligand(
                        chain_id=chain_info.get("id"),
                        ccd=chain_info.get("ccd"),
                        smiles=chain_info.get("smiles"),
                    )
                    self._ligands.append(ligand)
                else:
                    pass
        self._initialized = True

    @property
    def proteins(self) -> list[Protein] | None:
        """
        Get the proteins submitted for the fold request.

        Returns
        -------
        list[Protein] or None
            List of Protein objects or None.
        """
        if not self._initialized:
            self.__get_chains()
        return self._proteins

    @property
    def dnas(self) -> list[DNA] | None:
        """
        Get the DNAs submitted for the fold request.

        Returns
        -------
        list[DNA] or None
            List of DNA objects or None.
        """
        if not self._initialized:
            self.__get_chains()
        return self._dnas

    @property
    def rnas(self) -> list[RNA] | None:
        """
        Get the RNAs submitted for the fold request.

        Returns
        -------
        list[RNA] or None
            List of RNA objects or None.
        """
        if not self._initialized:
            self.__get_chains()
        return self._rnas

    @property
    def ligands(self) -> list[Ligand] | None:
        """
        Get the ligands submitted for the fold request.

        Returns
        -------
        list[Ligand] or None
            List of Ligand objects or None.
        """
        if not self._initialized:
            self.__get_chains()
        return self._ligands

    @property
    def pae(self) -> np.ndarray:
        """
        Get the Predicted Aligned Error (PAE) matrix.

        Returns
        -------
        np.ndarray
            PAE matrix.

        Raises
        ------
        AttributeError
            If PAE is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("pae not supported for non-Boltz model")
        if self._pae is None:
            pae = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="pae"
            )
            assert isinstance(pae, np.ndarray)
            self._pae = pae
        return self._pae

    @property
    def pde(self) -> np.ndarray:
        """
        Get the Predicted Distance Error (PDE) matrix.

        Returns
        -------
        np.ndarray
            PDE matrix.

        Raises
        ------
        AttributeError
            If PDE is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("pde not supported for non-Boltz model")
        if self._pde is None:
            pde = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="pde"
            )
            assert isinstance(pde, np.ndarray)
            self._pde = pde
        return self._pde

    @property
    def plddt(self) -> np.ndarray:
        """
        Get the Predicted Local Distance Difference Test (pLDDT) scores.

        Returns
        -------
        np.ndarray
            pLDDT scores.

        Raises
        ------
        AttributeError
            If pLDDT is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("plddt not supported for non-Boltz model")
        if self._plddt is None:
            plddt = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="plddt"
            )
            assert isinstance(plddt, np.ndarray)
            self._plddt = plddt
        return self._plddt

    @property
    def score(self) -> pd.DataFrame:
        """
        Get the predicted scores.

        Returns
        -------
        pd.DataFrame
            Structure prediction scores.

        Raises
        ------
        AttributeError
            If score is not supported for the model.
        """
        if self.model_id not in {"rosettafold-3"}:
            raise AttributeError("score not supported for non-RosettaFold model")
        if self._score is None:
            score = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="score"
            )
            assert isinstance(score, pd.DataFrame)
            self._score = score
        return self._score

    @property
    def metrics(self) -> pd.DataFrame:
        """
        Get the predicted metrics.

        Returns
        -------
        pd.DataFrame
            Structure prediction metrics.

        Raises
        ------
        AttributeError
            If metrics is not supported for the model.
        """
        if self.model_id not in {"rosettafold-3"}:
            raise AttributeError("metrics not supported for non-RosettaFold model")
        if self._metrics is None:
            metrics = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="metrics"
            )
            assert isinstance(metrics, pd.DataFrame)
            self._metrics = metrics
        return self._metrics

    @property
    def confidence(self) -> list["BoltzConfidence"]:
        """
        Retrieve the confidences of the structure prediction.

        Note
        ----
        This is only currently supported for Boltz models.

        Returns
        -------
        list[BoltzConfidence]
            List of BoltzConfidence objects.

        Raises
        ------
        AttributeError
            If confidence is not supported for the model.
        """
        from .boltz import BoltzConfidence

        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("confidence not supported for non-Boltz model")
        if self._confidence is None:
            confidence = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="confidence"
            )
            assert isinstance(confidence, list)
            self._confidence = TypeAdapter(list[BoltzConfidence]).validate_python(
                confidence
            )
        return self._confidence

    @property
    def affinity(self) -> "BoltzAffinity":
        """
        Retrieve the predicted binding affinities.

        Note
        ----
        This is only currently supported for Boltz models.

        Returns
        -------
        BoltzAffinity
            BoltzAffinity object containing the predicted affinities.

        Raises
        ------
        AttributeError
            If affinity is not supported for the model.
        """
        from .boltz import BoltzAffinity

        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("affinity not supported for non-Boltz model")
        if self._affinity is None:
            affinity = api.fold_get_complex_extra_result(
                session=self.session, job_id=self.job.job_id, key="affinity"
            )
            assert isinstance(affinity, dict)
            self._affinity = BoltzAffinity.parse_obj_with_models(affinity)
        return self._affinity

    @property
    def id(self):
        """
        Get the ID of the fold request.

        Returns
        -------
        str
            Fold job ID.
        """
        return self.job.job_id

    def get(self, format: Literal["pdb", "mmcif"] = "mmcif", verbose=False) -> bytes:
        """
        Retrieve the fold results as a single bytestring.

        Defaults to mmCIF for complexes. Additional predicted properties like plddt and pae should be accessed from their respective properties, i.e. `.plddt` and `.pae`.

        Parameters
        ----------
        format : {'pdb', 'mmcif'}, optional
            Output format. Default is 'mmcif'.
        verbose : bool, optional
            If True, print verbose output. Default is False.

        Returns
        -------
        bytes
            Fold result as a bytestring.
        """
        return api.fold_get_complex_result(
            session=self.session, job_id=self.id, format=format
        )
