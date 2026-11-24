"""Assays (``session.data.Assay``).

>>> assay = session.data.Assay.create(table, "stability")
>>> assay = session.data.Assay(id=assay.id)
>>> first_rows = assay.get_slice(0, 10)
"""

from __future__ import annotations

import builtins
import io
from collections.abc import Sequence
from datetime import datetime

import pandas as pd
from typing_extensions import Self

from openprotein import config
from openprotein._deprecation import warn_deprecated
from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.jobs.schemas import JobStatus

from . import api
from ._resource import _Resource
from .schemas import AssayMetadata

__all__ = ["Assay", "AssayDataset"]

_LIST_PAGE = 1000


class Assay(_Resource[AssayMetadata]):
    """An assay: your sequences and their measurements, which predictors train on.

    ``session.data.Assay.create`` uploads one, ``session.data.Assay.list`` lists
    the caller's and ``session.data.Assay(id=...)`` loads one.
    """

    def __init__(
        self,
        session: APISession | None = None,
        metadata: AssayMetadata | None = None,
        *,
        id: str | None = None,
    ):
        super().__init__(session, metadata, id=id)
        self.page_size = min(config.BASE_PAGE_SIZE, 1000)

    @classmethod
    def _load(cls, session: APISession, id: str) -> AssayMetadata:
        return api.get_assay_metadata(session, id)

    @classmethod
    def create(
        cls,
        table: pd.DataFrame,
        name: str,
        description: str | None = None,
        metadata_columns: Sequence[str] | None = None,
    ) -> Self:
        """
        Upload a table as an assay.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame containing the assay data.
        name : str
            Name of the assay.
        description : str, optional
            Description of the assay, by default None.
        metadata_columns : Sequence[str], optional
            Columns of free text, such as notes, kept per row as metadata.
            They are not measurements and cannot be used to train models.

        Returns
        -------
        Assay
            The stored assay.
        """
        session = cls._bound()
        stream = io.BytesIO()
        table.to_csv(stream, index=False)
        stream.seek(0)
        metadata = api.assaydata_post(
            session,
            stream,
            name,
            assay_description=description,
            metadata_columns=metadata_columns,
        )
        return cls(session, metadata)

    @classmethod
    def list(
        cls, limit: int | None = None, offset: int | None = None
    ) -> builtins.list[Self]:
        """The caller's assays from every page, at most ``limit``, starting at ``offset``."""
        session = cls._bound()
        start = offset or 0
        assays: builtins.list[Self] = []
        while limit is None or len(assays) < limit:
            size = _LIST_PAGE if limit is None else min(_LIST_PAGE, limit - len(assays))
            page = api.assaydata_list(session, limit=size, offset=start + len(assays))
            assays.extend(cls(session, metadata) for metadata in page)
            if len(page) < size:
                break
        return assays

    @property
    def id(self):
        return self.metadata.assay_id

    @property
    def name(self):
        return self.metadata.assay_name

    @property
    def description(self):
        return self.metadata.assay_description

    @property
    def status(self) -> JobStatus:
        """``SUCCESS`` once the dataset is ready, else the status of the job producing it."""
        return self.metadata.status

    @property
    def job_id(self) -> str | None:
        """The job that produces the dataset, such as an alignment; ``None`` for an upload."""
        return self.metadata.align_job_id

    @property
    def created_at(self) -> datetime:
        return self.metadata.created_date

    @property
    def measurement_names(self):
        return self.metadata.measurement_names

    @property
    def metadata_names(self) -> builtins.list[str]:
        return self.metadata.metadata_names or []

    @property
    def sequence_length(self):
        return self.metadata.sequence_length

    def __len__(self):
        return self.metadata.num_rows

    def __bool__(self) -> bool:
        return len(self) != 0

    @property
    def shape(self):
        return (len(self), len(self.measurement_names) + 1)

    def list_models(self):
        """
        List models assoicated with assay.

        Returns
        -------
        List
            List of models
        """
        return api.list_models(self.session, self.id)

    def update(
        self,
        name: str | None = None,
        description: str | None = None,
        *,
        assay_name: str | None = None,
        assay_description: str | None = None,
    ) -> Self:
        """Update the name or description; only the ones given change.

        .. deprecated:: 0.18.0
            ``assay_name`` and ``assay_description``. Use ``name`` and
            ``description``.
        """
        if assay_name is not None or assay_description is not None:
            if (name is not None and assay_name is not None) or (
                description is not None and assay_description is not None
            ):
                raise TypeError(
                    "pass name and description, without assay_name or assay_description"
                )
            warn_deprecated(
                "Assay.update(assay_name=, assay_description=)",
                "`Assay.update(name=, description=)`",
                since="0.18.0",
                removed_in="0.19.0",
            )
            name = assay_name if name is None else name
            description = assay_description if description is None else description
        updated = api.assaydata_put(
            self.session, self.id, assay_name=name, assay_description=description
        )
        if updated.align_job_id is None and self.job_id is not None:
            # The update's answer leaves out the job, so its status reads SUCCESS.
            updated = self._load(self.session, self.id)
        self.metadata = updated
        return self

    def delete(self) -> None:
        """Delete this assay."""
        api.assaydata_delete(self.session, self.id)

    def _get_all(self, verbose: bool = False) -> pd.DataFrame:
        """
        Get all assay data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing all assay data.
        """
        step = self.page_size

        results = []
        num_returned = step
        offset = 0

        while num_returned >= step:
            try:
                result = self.get_slice(offset, offset + step)
                results.append(result)
                num_returned = len(result)
                offset += num_returned
            except APIError as exc:
                if verbose:
                    print(f"Failed to get results: {exc}")
                return pd.concat(results)
        return pd.concat(results)

    def get_first(self) -> pd.DataFrame:
        """
        Get head slice of assay data.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the slice of assay data.
        """
        rows = []
        entries = api.assaydata_page_get(
            self.session, self.id, page_offset=0, page_size=1
        )
        for row in entries.assaydata:
            row = [row.mut_sequence] + row.measurement_values
            rows.append(row)
        table = pd.DataFrame(rows, columns=["sequence"] + self.measurement_names)
        return table

    def get_slice(
        self, start: int, end: int, include_metadata: bool = False
    ) -> pd.DataFrame:
        """
        Get a slice of assay data.

        Parameters
        ----------
        start : int
            Start index of the slice.
        end : int
            End index of the slice.
        include_metadata : bool, optional
            Append the metadata columns after the measurements, by default False.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the slice of assay data.
        """
        rows = []
        page_size = self.page_size
        for i in range(start, end, page_size):
            # the last page might be smaller than the page size
            current_page_size = min(page_size, end - i)

            entries = api.assaydata_page_get(
                self.session,
                self.id,
                page_offset=i,
                page_size=current_page_size,
                include_metadata=include_metadata,
            )

            for row in entries.assaydata:
                values = [row.mut_sequence] + row.measurement_values
                if include_metadata:
                    metadata = list(row.metadata_values or [])
                    values += metadata + [None] * (
                        len(self.metadata_names) - len(metadata)
                    )
                rows.append(values)

        columns = ["sequence"] + self.measurement_names
        if include_metadata:
            columns += self.metadata_names
        table = pd.DataFrame(rows, columns=columns)
        if include_metadata:
            # object dtype keeps a missing metadata value as None; pandas 3 infers
            # its str dtype, which reads one as NaN
            start = len(columns) - len(self.metadata_names)
            for i, name in enumerate(self.metadata_names, start=start):
                table[name] = pd.Series([row[i] for row in rows], dtype=object)
        return table


AssayDataset = Assay
