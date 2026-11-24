"""Align API interface for creating alignments and MSAs (multiple sequence alignments) which can be used for other protein tasks."""

from collections.abc import Sequence
from typing import BinaryIO, Iterator

from openprotein import _deprecation
from openprotein.base import APISession
from openprotein.errors import DeprecationError
from openprotein.jobs import Job

from . import api
from ._files import read_file_content, read_sequences
from .msa import MSAFuture
from .schemas import AbNumberScheme, Pairing, UnnumberedPolicy

__all__ = ["AlignAPI"]


def _as_sequences(sequences: Sequence[bytes | str]) -> list[str]:
    """Coerce the caller's sequences to the JSON array the align endpoints take."""
    return [s.decode() if isinstance(s, bytes) else s for s in sequences]


def _as_names(names: Sequence[str] | None) -> list[str] | None:
    """Coerce optional row labels to a list, leaving ``None`` alone."""
    return None if names is None else list(names)


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
            Row labels for the aligned output, same length as sequences. Omit them
            and the aligner labels the rows itself.
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
        job = api.mafft_post(
            self.session,
            _as_sequences(sequences),
            names=_as_names(names),
            auto=auto,
            ep=ep,
            op=op,
        )
        return MSAFuture.create(session=self.session, job=job)

    def mafft_file(
        self,
        file,
        auto: bool = True,
        ep: float | None = None,
        op: float | None = None,
    ) -> MSAFuture:
        """
        Align sequences from a file using the `mafft` algorithm.

        .. deprecated:: 0.18.0
            Use `AlignAPI.mafft` with sequences and names instead.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Set `auto` to True to automatically attempt the best parameters. Leave a parameter as None to use system defaults.

        Parameters
        ----------
        file : file-like object, bytes, bytearray, str, tuple or list
            Sequences to align in FASTA or CSV format. A tuple or list is
            ``(filename, fileobj)`` or ``(filename, fileobj, content_type)``.
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
        InvalidParameterError
            If the file has no sequences or a CSV has more than two columns.
        """
        _deprecation.warn_deprecated(
            "AlignAPI.mafft_file",
            "`AlignAPI.mafft` with sequences and names",
            since="0.18.0",
            removed_in="0.19.0",
        )
        sequences, names = read_sequences(read_file_content(file))
        return self.mafft(sequences, names, auto=auto, ep=ep, op=op)

    def clustalo(
        self,
        sequences: Sequence[bytes | str],
        names: Sequence[str] | None = None,
        clustersize: int | None = None,
        iterations: int | None = None,
    ) -> MSAFuture:
        """
        Align sequences using the `clustal omega` algorithm.

        Parameters
        ----------
        sequences : Sequence[bytes or str]
            Sequences to align.
        names : Sequence[str], optional
            Row labels for the aligned output, same length as sequences. Omit them
            and the aligner labels the rows itself.
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
            self.session,
            _as_sequences(sequences),
            names=_as_names(names),
            clustersize=clustersize,
            iterations=iterations,
        )
        return MSAFuture.create(session=self.session, job=job)

    def clustalo_file(
        self,
        file,
        clustersize: int | None = None,
        iterations: int | None = None,
    ) -> MSAFuture:
        """
        Align sequences from a file using the `clustal omega` algorithm.

        .. deprecated:: 0.18.0
            Use `AlignAPI.clustalo` with sequences and names instead.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Parameters
        ----------
        file : file-like object, bytes, bytearray, str, tuple or list
            Sequences to align in FASTA or CSV format. A tuple or list is
            ``(filename, fileobj)`` or ``(filename, fileobj, content_type)``.
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
        InvalidParameterError
            If the file has no sequences or a CSV has more than two columns.
        """
        _deprecation.warn_deprecated(
            "AlignAPI.clustalo_file",
            "`AlignAPI.clustalo` with sequences and names",
            since="0.18.0",
            removed_in="0.19.0",
        )
        sequences, names = read_sequences(read_file_content(file))
        return self.clustalo(
            sequences, names, clustersize=clustersize, iterations=iterations
        )

    def abnumber(
        self,
        sequences: Sequence[bytes | str],
        names: Sequence[str] | None = None,
        scheme: AbNumberScheme = AbNumberScheme.CHOTHIA,
        drop_minority_chains: bool = False,
        unnumbered_policy: UnnumberedPolicy = UnnumberedPolicy.ERROR,
    ) -> MSAFuture:
        """
        Align antibody sequences using `AbNumber`.

        The antibody numbering scheme can be specified.

        Parameters
        ----------
        sequences : Sequence[bytes or str]
            Sequences to align.
        names : Sequence[str], optional
            Row labels for the aligned output, same length as sequences. Omit them
            and the aligner labels the rows itself.
        scheme : AbNumberScheme, default=AbNumberScheme.CHOTHIA
            Antibody numbering scheme.
        drop_minority_chains : bool, default=False
            If True, drop sequences belonging to chain types that are in the
            minority (e.g. heavy vs light) so the resulting alignment contains
            only the dominant chain type.
        unnumbered_policy : UnnumberedPolicy, default=UnnumberedPolicy.ERROR
            What to do with residues outside the numbered variable domain --
            a constant/Fc region, an N-terminal tag. ERROR fails the job, DROP
            aligns the numbered residues only, RETAIN keeps them padded against
            the numbered-domain boundary.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.
        """
        job = api.abnumber_post(
            self.session,
            _as_sequences(sequences),
            names=_as_names(names),
            scheme=scheme,
            drop_minority_chains=drop_minority_chains,
            unnumbered_policy=unnumbered_policy,
        )
        return MSAFuture.create(session=self.session, job=job)

    def abnumber_file(
        self,
        file,
        scheme: AbNumberScheme = AbNumberScheme.CHOTHIA,
        drop_minority_chains: bool = False,
        unnumbered_policy: UnnumberedPolicy = UnnumberedPolicy.ERROR,
    ) -> MSAFuture:
        """
        Align antibody sequences from a file using `AbNumber`.

        .. deprecated:: 0.18.0
            Use `AlignAPI.abnumber` with sequences and names instead.

        Sequences can be provided as FASTA or CSV formats. If CSV, the file must be headerless with either a single sequence column or name, sequence columns.

        Parameters
        ----------
        file : file-like object, bytes, bytearray, str, tuple or list
            Sequences to align in FASTA or CSV format. A tuple or list is
            ``(filename, fileobj)`` or ``(filename, fileobj, content_type)``.
        scheme : AbNumberScheme, default=AbNumberScheme.CHOTHIA
            Antibody numbering scheme.
        drop_minority_chains : bool, default=False
            If True, drop sequences belonging to chain types that are in the
            minority (e.g. heavy vs light) so the resulting alignment contains
            only the dominant chain type.
        unnumbered_policy : UnnumberedPolicy, default=UnnumberedPolicy.ERROR
            What to do with residues outside the numbered variable domain --
            a constant/Fc region, an N-terminal tag. ERROR fails the job, DROP
            aligns the numbered residues only, RETAIN keeps them padded against
            the numbered-domain boundary.

        Returns
        -------
        MSAFuture
            Future object awaiting the contents of the MSA upload.

        Raises
        ------
        InvalidParameterError
            If the file has no sequences or a CSV has more than two columns.
        """
        _deprecation.warn_deprecated(
            "AlignAPI.abnumber_file",
            "`AlignAPI.abnumber` with sequences and names",
            since="0.18.0",
            removed_in="0.19.0",
        )
        sequences, names = read_sequences(read_file_content(file))
        return self.abnumber(
            sequences,
            names,
            scheme=scheme,
            drop_minority_chains=drop_minority_chains,
            unnumbered_policy=unnumbered_policy,
        )

    def upload_msa(self, msa_file: BinaryIO) -> MSAFuture:
        """
        Upload a ready-made MSA.

        .. deprecated:: 0.18.0
            Use `session.data.MSA.create` and `MSA.sample_prompt`
            instead.

        The returned future samples prompts with ``sample_prompt``. For a
        complex, join the chains of each row with ``:``, or use the a3m
        ``#<lengths>\t<cardinalities>`` header.

        Parameters
        ----------
        msa_file : BinaryIO
            The MSA as FASTA, a3m or a headerless CSV of names and sequences.

        Returns
        -------
        MSAFuture
            Future for the upload.

        Raises
        ------
        APIError
            If there is an issue with the API request.
        """
        _deprecation.warn_deprecated(
            "AlignAPI.upload_msa",
            "`session.data.MSA.create` and `MSA.sample_prompt`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return MSAFuture.create(
            session=self.session, job=api._msa_post(self.session, msa_file=msa_file)
        )

    def create_msa(
        self, seed: bytes | str, pairing: Pairing = Pairing.BOTH
    ) -> MSAFuture:
        """
        Construct an MSA via homology search with the seed sequence.

        Parameters
        ----------
        seed : bytes
            Seed sequence for the MSA construction.
        pairing : Pairing, optional
            Cross-chain pairing strategy for a complex (`:`-separated) seed;
            has no effect on a single-chain seed. Defaults to `both`.

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
            session=self.session,
            job=api._msa_mmseqs_search_post(self.session, seed=seed, pairing=pairing),
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
            Always. Use `create_prompt` on the `prompt` module instead.
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
            Always. Use `get_prompt` on the `prompt` module instead.
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
