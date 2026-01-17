"""Fold prediction results represented as futures."""

import copy
import typing
from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np
import pandas as pd
from pydantic.type_adapter import TypeAdapter
from typing_extensions import Self

from openprotein import config
from openprotein.base import APISession
from openprotein.fold.complex import id_generator
from openprotein.jobs import JobsAPI, MappedFuture
from openprotein.molecules import DNA, RNA, Complex, Ligand, Protein, Structure
from openprotein.utils.numpy import readonly_view

from . import api
from .schemas import FoldJob, FoldMetadata

if TYPE_CHECKING:
    from .boltz import BoltzAffinity, BoltzConfidence

FoldResult: typing.TypeAlias = (
    "Structure | np.ndarray | pd.DataFrame | BoltzAffinity | list[BoltzConfidence]"
)


class FoldResultFuture(
    MappedFuture[
        bytes,
        FoldResult,
    ]
):
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
        complexes: list[Complex] | None = None,
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
        self._complexes = complexes
        self.reverse_map = {s: i for i, s in enumerate(self._sequences)}
        super().__init__(session, job, max_workers)

    @classmethod
    def create(
        cls: type[Self],
        session: APISession,
        job: FoldJob | None = None,
        metadata: FoldMetadata | None = None,
        **kwargs,
    ) -> "Self":
        """
        Factory method to create a FoldResultFuture.

        Parameters
        ----------
        session : APISession
            The API session to use for requests.
        job : FoldJob
            The fold job associated with this future.

            Additional keyword arguments.

        Returns
        -------
        FoldResultFuture
            An instance of FoldResultFuture.
        """
        if job is not None:
            job_id = job.job_id
        elif metadata is not None:
            job_id = metadata.job_id
        else:
            raise ValueError("Expected fold metadata or job")
        # model_id = api.fold_get(session=session, job_id=job_id).model_id
        # create different future - not used now
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
        import warnings

        warnings.warn(
            "`sequences` for fold jobs of complexes will show ':'-delimited protein sequences but omit the ligands and other chain entities"
        )
        if self._sequences is None:
            self._sequences = api.fold_get_sequences(self.session, self.job.job_id)
        return self._sequences

    @property
    def complexes(self) -> list[Complex]:
        """
        Get the molecular complexes submitted for the fold request.

        Returns
        -------
        list[Complex]
            List of complexes.
        """
        if self._complexes is not None:
            return copy.deepcopy(self._complexes)
        complexes: list[Complex] = []
        if self.metadata.sequences is None:
            # make from self.sequences instead
            # all proteins
            id_gen = id_generator()
            for seq in self.sequences:
                proteins = {}
                for monomer in seq.split(b":"):
                    chain_id = next(id_gen)
                    protein = Protein(sequence=monomer)
                    proteins[chain_id] = protein
                model = Complex(chains=proteins)
                complexes.append(model)
        else:
            # collate used ids
            used_ids = []
            for complex_dicts in self.metadata.sequences:
                for complex_dict in complex_dicts:
                    for entity_dict in complex_dict.values():
                        if (id := entity_dict.get("id")) is not None:
                            if isinstance(id, str):
                                used_ids.append(id)
                            elif isinstance(id, list):
                                used_ids.extend(id)
            id_gen = id_generator(used_ids)
            for complex_dicts in self.metadata.sequences:
                chains: dict = {}
                for complex_dict in complex_dicts:
                    for entity_type, entity_dict in complex_dict.items():
                        if entity_type == "protein":
                            chain_id = entity_dict.get("id") or next(id_gen)
                            protein = Protein(sequence=entity_dict["sequence"])
                            if (msa_id := entity_dict.get("msa_id")) is not None:
                                protein.msa = msa_id
                            if isinstance(chain_id, list):
                                for id in chain_id:
                                    chains[id] = protein
                            else:
                                chains[chain_id] = protein
                        elif entity_type == "dna":
                            chain_id = entity_dict.get("id") or next(id_gen)
                            dna = DNA(
                                sequence=entity_dict["sequence"],
                                cyclic=entity_dict.get("cyclic"),
                            )
                            if isinstance(chain_id, list):
                                for id in chain_id:
                                    chains[id] = dna
                            else:
                                chains[chain_id] = dna
                        elif entity_type == "rna":
                            chain_id = entity_dict.get("id") or next(id_gen)
                            rna = RNA(
                                sequence=entity_dict["sequence"],
                                cyclic=entity_dict.get("cyclic"),
                            )
                            if isinstance(chain_id, list):
                                for id in chain_id:
                                    chains[id] = rna
                            else:
                                chains[chain_id] = rna
                        elif entity_type == "ligand":
                            chain_id = entity_dict.get("id") or next(id_gen)
                            ligand = Ligand(
                                smiles=entity_dict.get("smiles"),
                                ccd=entity_dict.get("ccd"),
                            )
                            if isinstance(chain_id, list):
                                for id in chain_id:
                                    chains[id] = ligand
                            else:
                                chains[chain_id] = ligand
                complexes.append(Complex(chains=chains))
        self._complexes = complexes
        return copy.deepcopy(self._complexes)

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
        return list(range(len(self._sequences)))

    @typing.overload
    def get_item(
        self,
        index: int,
        key: None = None,
    ) -> Structure: ...

    @typing.overload
    def get_item(
        self,
        index: int,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
            ]
            | None
        ) = None,
    ) -> np.ndarray: ...

    @typing.overload
    def get_item(
        self,
        index: int,
        key: Literal["affinity"],
    ) -> "BoltzAffinity": ...

    @typing.overload
    def get_item(
        self,
        index: int,
        key: Literal["confidence"],
    ) -> "list[BoltzConfidence]": ...

    @typing.overload
    def get_item(
        self,
        index: int,
        key: (
            Literal[
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> pd.DataFrame: ...

    def get_item(
        self,
        index: int,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
                "confidence",
                "affinity",
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> FoldResult:
        """
        Get fold results for a specified sequence.

        Parameters
        ----------
        sequence : bytes
            Sequence to fetch results for.

        Returns
        -------
        Complex
            Complex containing the folded structure.
        """
        if key is None:
            data = api.fold_get_sequence_result(self.session, self.job.job_id, index)
            model = Structure.from_string(data.decode(), format="cif")
            return model
        else:
            data = api.fold_get_extra_result(self.session, self.job.job_id, index, key)
            if key == "affinity":
                from .boltz import BoltzAffinity

                data = TypeAdapter(BoltzAffinity).validate_python(data)
            elif key == "confidence":
                from .boltz import BoltzConfidence

                data = TypeAdapter(list[BoltzConfidence]).validate_python(data)
            return data  # type: ignore - converted by adapter

    @typing.overload
    def stream(
        self,
        key: None = None,
    ) -> Iterator[Structure]: ...

    @typing.overload
    def stream(
        self,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
            ]
            | None
        ) = None,
    ) -> Iterator[np.ndarray]: ...

    @typing.overload
    def stream(
        self,
        key: Literal["affinity"],
    ) -> "Iterator[BoltzAffinity]": ...

    @typing.overload
    def stream(
        self,
        key: Literal["confidence"],
    ) -> "Iterator[list[BoltzConfidence]]": ...

    @typing.overload
    def stream(
        self,
        key: (
            Literal[
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> Iterator[pd.DataFrame]: ...

    # NOTE: ensure we only return the complex without the tuple
    def stream(
        self,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
                "confidence",
                "affinity",
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> "Iterator[Structure] | Iterator[np.ndarray] | Iterator[pd.DataFrame] | Iterator[BoltzAffinity] | Iterator[list[BoltzConfidence]]":
        for _, v in super().stream(key=key):
            yield v  # type: ignore - homogenous

    @typing.overload
    def get(
        self,
        verbose: bool = False,
        key: None = None,
    ) -> list[Structure]: ...

    @typing.overload
    def get(
        self,
        verbose: bool = False,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
            ]
            | None
        ) = None,
    ) -> list[np.ndarray]: ...

    @typing.overload
    def get(
        self,
        verbose: bool = False,
        key: Literal["affinity"] | None = None,
    ) -> "list[BoltzAffinity]": ...

    @typing.overload
    def get(
        self,
        verbose: bool = False,
        key: Literal["confidence"] | None = None,
    ) -> "list[list[BoltzConfidence]]": ...

    @typing.overload
    def get(
        self,
        verbose: bool = False,
        key: (
            Literal[
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> list[pd.DataFrame]: ...

    def get(
        self,
        verbose: bool = False,
        key: (
            Literal[
                "pae",
                "pde",
                "plddt",
                "ptm",
                "confidence",
                "affinity",
                "score",
                "metrics",
            ]
            | None
        ) = None,
    ) -> "list[Structure] | list[np.ndarray] | list[pd.DataFrame] | list[list[BoltzConfidence]] | list[BoltzAffinity]":
        return super().get(verbose, key=key)  # type: ignore - homogenous

    def get_pae(self) -> list[np.ndarray]:
        """
        Get the Predicted Aligned Error (PAE) matrix for all outputs.

        Returns
        -------
        list[np.ndarray]
            PAE matrix.

        Raises
        ------
        AttributeError
            If PAE is not supported for the model.
        """
        if self.model_id not in {
            "boltz-1",
            "boltz-1x",
            "boltz-2",
            "alphafold2",
            "esmfold",
        }:
            raise AttributeError("pae not supported for this model")
        if not hasattr(self, "_pae"):
            self._pae = None
        if self._pae is None:
            pae = self.get(key="pae")
            self._pae = pae
        return [readonly_view(x) for x in self._pae]

    def get_pde(self) -> list[np.ndarray]:
        """
        Get the Predicted Distance Error (PDE) matrix.

        Returns
        -------
        list[np.ndarray]
            PDE matrix.

        Raises
        ------
        AttributeError
            If PDE is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("pde not supported for this model")
        if not hasattr(self, "_pde"):
            self._pde = None
        if self._pde is None:
            pde = self.get(key="pde")
            self._pde = pde
        return [readonly_view(x) for x in self._pde]

    def get_plddt(self) -> list[np.ndarray]:
        """
        Get the Predicted Local Distance Difference Test (pLDDT) scores.

        Returns
        -------
        list[np.ndarray]
            pLDDT scores.

        Raises
        ------
        AttributeError
            If pLDDT is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2", "alphafold2"}:
            raise AttributeError("plddt not supported for this model")
        if not hasattr(self, "_plddt"):
            self._plddt = None
        if self._plddt is None:
            plddt = self.get(key="plddt")
            self._plddt = plddt
        return [readonly_view(x) for x in self._plddt]

    def get_ptm(self) -> list[np.ndarray]:
        """
        Get the Predicted TM (pTM) scores.

        Returns
        -------
        list[np.ndarray]
            pTM scores.

        Raises
        ------
        AttributeError
            If pTM is not supported for the model.
        """
        if self.model_id not in {"alphafold2"}:
            raise AttributeError("ptm not supported for this model")
        if not hasattr(self, "_ptm"):
            self._ptm = None
        if self._ptm is None:
            ptm = self.get(key="ptm")
            self._ptm = ptm
        return [readonly_view(x) for x in self._ptm]

    def get_score(self) -> list[pd.DataFrame]:
        """
        Get the predicted scores.

        Returns
        -------
        list[pd.DataFrame]
            Structure prediction scores.

        Raises
        ------
        AttributeError
            If score is not supported for the model.
        """
        if self.model_id not in {"rosettafold-3"}:
            raise AttributeError("score not supported for this model")
        if not hasattr(self, "_score"):
            self._score = None
        if self._score is None:
            score = self.get(key="score")
            self._score = score
        return copy.deepcopy(self._score)

    def get_metrics(self) -> list[pd.DataFrame]:
        """
        Get the predicted metrics.

        Returns
        -------
        list[pd.DataFrame]
            Structure prediction metrics.

        Raises
        ------
        AttributeError
            If metrics is not supported for the model.
        """
        if self.model_id not in {"rosettafold-3"}:
            raise AttributeError("metrics not supported for this model")
        if not hasattr(self, "_metrics"):
            self._metrics = None
        if self._metrics is None:
            metrics = self.get(key="metrics")
            self._metrics = metrics
        return copy.deepcopy(self._metrics)

    def get_confidence(self) -> list[list["BoltzConfidence"]]:
        """
        Retrieve the confidences of the structure prediction.

        Note
        ----
        This is only currently supported for Boltz models.

        Returns
        -------
        list[list[BoltzConfidence]]
            List of list of BoltzConfidence objects.

        Raises
        ------
        AttributeError
            If confidence is not supported for the model.
        """
        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("confidence not supported for non-Boltz model")
        if not hasattr(self, "_confidence"):
            self._confidence = None
        if self._confidence is None:
            confidence = self.get(key="confidence")
            self._confidence = confidence
        return copy.deepcopy(self._confidence)

    def get_affinity(self) -> list["BoltzAffinity"]:
        """
        Retrieve the predicted binding affinities.

        Note
        ----
        This is only currently supported for Boltz models.

        Returns
        -------
        list[list[BoltzAffinity]]
            BoltzAffinity object containing the predicted affinities.

        Raises
        ------
        AttributeError
            If affinity is not supported for the model.
        """
        from .boltz import BoltzAffinity

        if self.model_id not in {"boltz-1", "boltz-1x", "boltz-2"}:
            raise AttributeError("affinity not supported for non-Boltz model")
        if not hasattr(self, "_affinity"):
            self._affinity = None
        if self._affinity is None:
            affinity = self.get(key="affinity")
            self._affinity = affinity
        return copy.deepcopy(self._affinity)
