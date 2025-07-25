"""Align API interface for creating alignments and MSAs (multiple sequence alignments) which can be used for other protein tasks."""

from collections.abc import Sequence
from io import BytesIO
from typing import BinaryIO, Iterator

from openprotein.base import APISession
from openprotein.errors import DeprecationError
from openprotein.jobs import Job
from openprotein.protein import Protein

from . import api
from .msa import MSAFuture
from .schemas import AbNumberScheme, AlignType


class AlignAPI:
    """Align API interface for creating alignments and MSAs (multiple sequence alignments) which can be used for other protein tasks."""

    def __init__(self, session: APISession):
        self.session = session

    def mafft(
        self,
        sequences: Sequence[bytes | str],
        names: Sequence[str] | None = None,
        auto: bool = True,
        ep: float | None = None,
        op: float | None = None,
    ) -> MSAFuture:
        """
        Align sequences using the `mafft` algorithm.

        Set `auto` to True to automatically attempt the best parameters. Leave a parameter as None to use system defaults.

        Parameters
        ----------
        sequences : Sequence[bytes or str]
            Sequences to align.
        names : Sequence[str], optional
            Optional list of sequence names, must be the same length as sequences if provided.
        auto : bool, default=True
            Set to True to automatically set algorithm parameters.
        ep : float, optional
            MAFFT "ep" parameter. Sets the offset value for the scoring matrix; lower values make gap opening more difficult. If None, uses system default.
        op : float, optional
            MAFFT "op" parameter. Sets the gap opening penalty; higher values increase the cost of opening gaps. If None, uses system default.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        Exception
            If names and sequences are not the same length.
        """
        if names is not None and len(names) != len(sequences):
            raise Exception(
                f"Names and sequences must be same length, but were {len(names)} and {len(sequences)}"
            )
        lines = []
        if names is None:
            # as CSV
            lines = [s.encode() if isinstance(s, str) else s for s in sequences]
        else:
            # as fasta
            for name, sequence in zip(names, sequences):
                if isinstance(name, str):
                    name = name.encode()
                if isinstance(sequence, str):
                    sequence = sequence.encode()
                lines.append(b">" + name)
                lines.append(sequence)
        content = b"\n".join(lines)
        stream = BytesIO(content)
        return self.mafft_file(stream, auto=auto, ep=ep, op=op)

    def mafft_file(self, file, auto=True, ep=None, op=None) -> MSAFuture:
        """
        Align sequences using the `mafft` algorithm. Sequences can be provided as FASTA or CSV formats.
        If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Set `auto` to True to automatically attempt the best parameters. Leave a parameter as None to use system defaults.

        Parameters
        ----------
        file : file-like object
            Sequences to align in FASTA or CSV format.
        auto : bool, default=True
            Set to True to automatically set algorithm parameters.
        ep : float, optional
            MAFFT "ep" parameter. Sets the offset value for the scoring matrix; lower values make gap opening more difficult. If None, uses system default.
        op : float, optional
            MAFFT "op" parameter. Sets the gap opening penalty; higher values increase the cost of opening gaps. If None, uses system default.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = api.mafft_post(self.session, file, auto=auto, ep=ep, op=op)
        return MSAFuture.create(session=self.session, job=job)

    def clustalo(
        self,
        sequences: Sequence[bytes | str],
        names: Sequence[str] | None = None,
        clustersize: int | None = None,
        iterations: int | None = None,
    ) -> MSAFuture:
        """
        Align sequences using the `clustal omega` algorithm.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Parameters
        ----------
        sequences : Sequence[bytes or str]
            Sequences to align.
        names : Sequence[str], optional
            Optional list of sequence names, must be the same length as sequences if provided.
        clustersize : int, optional
            Maximum number of sequences per cluster during guide tree generation. If None, uses the default value.
        iterations : int, optional
            Number of refinement iterations performed during alignment. If None, uses the default value.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        Exception
            If names and sequences are not the same length.
        """
        if names is not None and len(names) != len(sequences):
            raise Exception(
                f"Names and sequences must be same length, but were {len(names)} and {len(sequences)}"
            )
        lines = []
        if names is None:
            # as CSV
            lines = [s.encode() if isinstance(s, str) else s for s in sequences]
        else:
            # as fasta
            for name, sequence in zip(names, sequences):
                if isinstance(name, str):
                    name = name.encode()
                if isinstance(sequence, str):
                    sequence = sequence.encode()
                lines.append(b">" + name)
                lines.append(sequence)
        content = b"\n".join(lines)
        stream = BytesIO(content)
        return self.clustalo_file(
            stream, clustersize=clustersize, iterations=iterations
        )

    def clustalo_file(self, file, clustersize=None, iterations=None) -> MSAFuture:
        """
        Align sequences using the `clustal omega` algorithm.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Parameters
        ----------
        file : file-like object
            Sequences to align in FASTA or CSV format.
        clustersize : int, optional
            Maximum number of sequences per cluster during guide tree generation. If None, uses the default value.
        iterations : int, optional
            Number of refinement iterations performed during alignment. If None, uses the default value.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = api.clustalo_post(
            self.session, file, clustersize=clustersize, iterations=iterations
        )
        return MSAFuture.create(session=self.session, job=job)

    def abnumber(
        self,
        sequences: Sequence[bytes | str],
        names: Sequence[str] | None = None,
        scheme: AbNumberScheme = AbNumberScheme.CHOTHIA,
    ) -> MSAFuture:
        """
        Align antibody sequences using `AbNumber`.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        The antibody numbering scheme can be specified.

        Parameters
        ----------
        sequences : Sequence[bytes or str]
            Sequences to align.
        names : Sequence[str], optional
            Optional list of sequence names, must be the same length as sequences if provided.
        scheme : AbNumberScheme, default=AbNumberScheme.CHOTHIA
            Antibody numbering scheme.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        Exception
            If names and sequences are not the same length.
        """
        if names is not None and len(names) != len(sequences):
            raise Exception(
                f"Names and sequences must be same length, but were {len(names)} and {len(sequences)}"
            )
        lines = []
        if names is None:
            # as CSV
            lines = [s.encode() if isinstance(s, str) else s for s in sequences]
        else:
            # as fasta
            for name, sequence in zip(names, sequences):
                if isinstance(name, str):
                    name = name.encode()
                if isinstance(sequence, str):
                    sequence = sequence.encode()
                lines.append(b">" + name)
                lines.append(sequence)
        content = b"\n".join(lines)
        stream = BytesIO(content)
        return self.abnumber_file(stream, scheme=scheme)

    def abnumber_file(
        self, file, scheme: AbNumberScheme = AbNumberScheme.CHOTHIA
    ) -> MSAFuture:
        """
        Align antibody sequences using `AbNumber`.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        The antibody numbering scheme can be specified.

        Parameters
        ----------
        file : file-like object
            Sequences to align in FASTA or CSV format.
        scheme : AbNumberScheme, default=AbNumberScheme.CHOTHIA
            Antibody numbering scheme.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = api.abnumber_post(self.session, file, scheme=scheme)
        return MSAFuture.create(session=self.session, job=job)

    def upload_msa(self, msa_file: BinaryIO) -> MSAFuture:
        """
        Upload an MSA from a file.

        Parameters
        ----------
        msa_file : str
            Path to a ready-made MSA file.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        APIError
            If there is an issue with the API request.
        """
        return MSAFuture.create(
            session=self.session, job=api.msa_post(self.session, msa_file=msa_file)
        )

    def create_msa(self, seed: bytes) -> MSAFuture:
        """
        Construct an MSA via homology search with the seed sequence.

        Parameters
        ----------
        seed : bytes
            Seed sequence for the MSA construction.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        APIError
            If there is an issue with the API request.
        """
        return MSAFuture.create(
            session=self.session, job=api.msa_post(self.session, seed=seed)
        )

    def upload_prompt(self, prompt_file: BinaryIO):
        """
        Directly upload a prompt.

        This method is deprecated. Use `create_prompt` on the `prompt` module instead.

        Parameters
        ----------
        prompt_file : BinaryIO
            Binary I/O object representing the prompt file.

        Returns
        -------
        PromptJob
            An object representing the status and results of the prompt job.

        Raises
        ------
        DeprecationError
            This method is no longer supported.
        """
        raise DeprecationError(
            "This method is no longer supported! Use `create_prompt` on the `prompt` module instead."
        )

    def get_prompt(
        self, job: Job, prompt_index: int | None = None
    ) -> Iterator[list[str]]:
        """
        Get prompts for a given job.

        This method is deprecated. Use `get_prompt` on the `prompt` module instead.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.
        prompt_index : int, optional
            The replicate number for the prompt (input_type=-PROMPT only).

        Returns
        -------
        Iterator[list[str]]
            An iterator over rows of the prompt data.

        Raises
        ------
        DeprecationError
            This method is no longer supported.
        """
        raise DeprecationError(
            "This method is no longer supported! Use `get_prompt` on the `prompt` module instead."
        )

    def get_seed(self, job_id: str) -> str:
        """
        Get seed sequence for a given MSA job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        str
            Seed sequence that was used to generate the MSA.
        """
        return api.get_seed(session=self.session, job_id=job_id)

    def get_msa(self, job_id: str) -> Iterator[tuple[str, str]]:
        """
        Get generated MSA for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        Iterator[tuple[str, str]]
            An iterator over names and sequences of the MSA data.
        """
        return api.get_msa(session=self.session, job_id=job_id)
