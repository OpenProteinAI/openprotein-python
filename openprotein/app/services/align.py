from collections.abc import Sequence
from io import BytesIO
from typing import BinaryIO, Iterator

from openprotein.api import align
from openprotein.app.models import MSAFuture
from openprotein.base import APISession
from openprotein.errors import DeprecationError
from openprotein.schemas import AlignType, Job


class AlignAPI:
    """API interface for calling Poet and Align endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    # TODO - document the `ep` and `op` parameters
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

        Set auto to True to automatically attempt the best params. Leave a parameter as None to use system defaults.

        Parameters
        ----------
        sequences : Sequence[bytes | str]
            Sequences to align
        names : Sequence[string], optional
            Optional list of sequence names, must be same length as sequences if provided.
        auto : bool = True, optional
            Set to true to automatically set algorithm parameters.
        ep : float, optional
            mafft parameter
        op : float, optional
            mafft parameter

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
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

    # TODO - document the `ep` and `op` parameters
    def mafft_file(self, file, auto=True, ep=None, op=None) -> MSAFuture:
        """
        Align sequences using the `mafft` algorithm. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

        Set auto to True to automatically attempt the best params. Leave a parameter as None to use system defaults.

        Parameters
        ----------
        file : File
            Sequences to align in fasta or csv format.
        auto : bool = True, optional
            Set to true to automatically set algorithm parameters.
        ep : float, optional
            mafft parameter
        op : float, optional
            mafft parameter

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = align.mafft_post(self.session, file, auto=auto, ep=ep, op=op)
        return MSAFuture.create(session=self.session, job=job)

    # TODO - document the parameters
    def clustalo(
        self, sequences, names=None, clustersize=None, iterations=None
    ) -> MSAFuture:
        """
        Align sequences using the `clustal omega` algorithm. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

        Leave a parameters as None to use system defaults.

        Parameters
        ----------
        sequences : List[bytes]
            Sequences to align
        names : List[string], optional
            Optional list of sequence names, must be same length as sequences if provided.
        clustersize : int, optional
            clustal omega parameter
        iterations : int, optional
            clustal omega parameter

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        if names is not None and len(names) != len(sequences):
            raise Exception(
                f"Names and sequences must be same length, but were {len(names)} and {len(sequences)}"
            )
        lines = []
        if names is None:
            # as CSV
            lines = sequences
        else:
            # as fasta
            for name, sequence in zip(names, sequences):
                if type(name) is str:
                    name = name.encode()
                lines.append(b">" + name)
                lines.append(sequence)
        content = b"\n".join(lines)
        stream = BytesIO(content)
        return self.clustalo_file(
            stream, clustersize=clustersize, iterations=iterations
        )

    # TODO - document the parameters
    def clustalo_file(self, file, clustersize=None, iterations=None) -> MSAFuture:
        """
        Align sequences using the `clustal omega` algorithm. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

        Leave a parameters as None to use system defaults.

        Parameters
        ----------
        file : File
            Sequences to align in fasta or csv format.
        clustersize : int, optional
            clustal omega parameter
        iterations : int, optional
            clustal omega parameter

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = align.clustalo_post(
            self.session, file, clustersize=clustersize, iterations=iterations
        )
        return MSAFuture.create(session=self.session, job=job)

    def abnumber(self, sequences, names=None, scheme="imgt") -> MSAFuture:
        """
        Align antibody using `AbNumber`. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

        The antibody numbering scheme can be specified from `imgt` (default), `chothia`, `kabat`, or `aho`.

        Parameters
        ----------
        sequences : List[bytes]
            Sequences to align
        names : List[string], optional
            Optional list of sequence names, must be same length as sequences if provided.
        scheme : str = 'imgt'
            Antibody numbering scheme. Can be one of 'imgt', 'chothia', 'kabat', or 'aho'

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        if names is not None and len(names) != len(sequences):
            raise Exception(
                f"Names and sequences must be same length, but were {len(names)} and {len(sequences)}"
            )
        lines = []
        if names is None:
            # as CSV
            lines = sequences
        else:
            # as fasta
            for name, sequence in zip(names, sequences):
                if type(name) is str:
                    name = name.encode()
                lines.append(b">" + name)
                lines.append(sequence)
        content = b"\n".join(lines)
        stream = BytesIO(content)
        return self.abnumber_file(stream, scheme=scheme)

    # TODO - properly test me and add new AbNumberFuture to support additional GET endpoint
    def abnumber_file(self, file, scheme="imgt") -> MSAFuture:
        """
        Align antibody using `AbNumber`. Sequences can be provided as `fasta` or `csv` formats. If `csv`, the file must be headerless with either a single sequence column or name, sequence columns.

        The antibody numbering scheme can be specified from `imgt` (default), `chothia`, `kabat`, or `aho`.

        Parameters
        ----------
        file : File
            Sequences to align in fasta or csv format.
        scheme : str = 'imgt'
            Antibody numbering scheme. Can be one of 'imgt', 'chothia', 'kabat', or 'aho'

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = align.abnumber_post(self.session, file, scheme=scheme)
        return MSAFuture.create(session=self.session, job=job)

    def upload_msa(self, msa_file) -> MSAFuture:
        """
        Upload an MSA from file.

        Parameters
        ----------
        msa_file : str, optional
            Ready-made MSA. If not provided, default value is None.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        return MSAFuture.create(
            session=self.session, job=align.msa_post(self.session, msa_file=msa_file)
        )

    def create_msa(self, seed: bytes) -> MSAFuture:
        """
        Construct an MSA via homology search with the seed sequence.

        Parameters
        ----------
        seed : bytes
            Seed sequence for the MSA construction.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        MSAJob
            Job object containing the details of the MSA construction.
        """
        return MSAFuture.create(
            session=self.session, job=align.msa_post(self.session, seed=seed)
        )

    def upload_prompt(self, prompt_file: BinaryIO):
        """
        Directly upload a prompt.

        Bypass post_msa and prompt_post steps entirely. In this case PoET will use the prompt as is.
        You can specify multiple prompts (one per replicate) with an <END_PROMPT> and newline between CSVs.

        Parameters
        ----------
        prompt_file : BinaryIO
            Binary I/O object representing the prompt file.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        PromptJob
            An object representing the status and results of the prompt job.
        """
        raise DeprecationError(
            "This method is no longer supported! Use `create_prompt` on the `prompt` module instead."
        )

    def get_prompt(
        self, job: Job, prompt_index: int | None = None
    ) -> Iterator[list[str]]:
        """
        Get prompts for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.
        prompt_index : Optional[int]
            The replicate number for the prompt (input_type=-PROMPT only)

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        raise DeprecationError(
            "This method is no longer supported! Use `get_prompt` on the `prompt` module instead."
        )

    def get_seed(self, job: Job) -> Iterator[list[str]]:
        """
        Get input data for a given msa job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return align.get_input(
            session=self.session, job=job, input_type=AlignType.INPUT
        )

    def get_msa(self, job: Job) -> Iterator[list[str]]:
        """
        Get generated MSA for a given job.

        Parameters
        ----------
        job : Job
            The job for which to retrieve data.

        Returns
        -------
        csv.reader
            A CSV reader for the response data.
        """
        return align.get_input(session=self.session, job=job, input_type=AlignType.MSA)
