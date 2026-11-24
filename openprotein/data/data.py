import builtins
from collections.abc import Sequence

import pandas as pd

from openprotein._deprecation import warn_deprecated
from openprotein.base import APISession

from . import assaydataset
from .assaydataset import AssayDataset
from .msa import msa as stored_msa
from .prompts import prompts as stored_prompts

__all__ = ["DataAPI"]


class DataAPI:
    """Stored data: assays, MSAs and prompts, each through its class."""

    _assay_class = None
    _msa_class = None
    _prompt_class = None

    def __init__(self, session: APISession):
        self.session = session

    @property
    def Assay(self) -> type[assaydataset.Assay]:
        """Assays: ``Assay.create`` uploads, ``Assay.list`` lists, ``Assay(id=...)`` loads."""
        if self._assay_class is None:
            self._assay_class = assaydataset.Assay._bind(self.session)
        return self._assay_class

    @property
    def MSA(self) -> type[stored_msa.MSA]:
        """Stored MSAs: ``MSA.create`` uploads, ``MSA.list`` lists, ``MSA(id=...)`` loads."""
        if self._msa_class is None:
            self._msa_class = stored_msa.MSA._bind(self.session)
        return self._msa_class

    @property
    def Prompt(self) -> type[stored_prompts.Prompt]:
        """Stored prompts: ``Prompt.create`` uploads, ``Prompt.list`` lists, ``Prompt(id=...)`` loads."""
        if self._prompt_class is None:
            self._prompt_class = stored_prompts.Prompt._bind(self.session)
        return self._prompt_class

    def list(
        self, limit: int | None = None, offset: int | None = None
    ) -> builtins.list[AssayDataset]:
        """
        List the caller's assays.

        .. deprecated:: 0.18.0
            Use ``session.data.Assay.list``.

        Parameters
        ----------
        limit : int, optional
            Limit the number of assays to return.
        offset : int, optional
            Offset of assays to retrieve. Useful with limit.

        Returns
        -------
        List[AssayDataset]
            The assays from every page.
        """
        warn_deprecated(
            "session.data.list",
            "`session.data.Assay.list`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return builtins.list(self.Assay.list(limit=limit, offset=offset))

    def create(
        self,
        table: pd.DataFrame,
        name: str,
        description: str | None = None,
        metadata_columns: Sequence[str] | None = None,
    ) -> AssayDataset:
        """
        Create a new assay dataset.

        .. deprecated:: 0.18.0
            Use ``session.data.Assay.create``.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame containing the assay data.
        name : str
            Name of the assay dataset.
        description : str, optional
            Description of the assay dataset, by default None.
        metadata_columns : Sequence[str], optional
            Columns of free text, such as notes, kept per row as metadata.
            They are not measurements and cannot be used to train models.

        Returns
        -------
        AssayDataset
            Created assay dataset.
        """
        warn_deprecated(
            "session.data.create",
            "`session.data.Assay.create`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return self.Assay.create(
            table, name, description=description, metadata_columns=metadata_columns
        )

    def get(self, assay_id: str, verbose: bool = False) -> AssayDataset:
        """
        Get an assay dataset by its ID.

        .. deprecated:: 0.18.0
            Use ``session.data.Assay(id=...)``.

        Parameters
        ----------
        assay_id : str
            ID of the assay dataset.

        Returns
        -------
        AssayDataset
            Assay dataset with the specified ID.
        """
        warn_deprecated(
            "session.data.get",
            "`session.data.Assay(id=...)`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return self.Assay(id=assay_id)

    def load_assay(self, assay_id: str, verbose: bool = False) -> AssayDataset:
        """
        Get an assay dataset by its ID.

        .. deprecated:: 0.18.0
            Use ``session.data.Assay(id=...)``.
        """
        warn_deprecated(
            "session.data.load_assay",
            "`session.data.Assay(id=...)`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return self.Assay(id=assay_id)

    def __len__(self) -> int:
        """
        The number of the caller's assays.

        .. deprecated:: 0.18.0
            Use ``len(session.data.Assay.list())``.
        """
        warn_deprecated(
            "len(session.data)",
            "`len(session.data.Assay.list())`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return len(self.Assay.list())
