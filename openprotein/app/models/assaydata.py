import pandas as pd
from openprotein import config
from openprotein.api import assaydata
from openprotein.base import APISession
from openprotein.errors import APIError
from openprotein.schemas import AssayDataPage, AssayMetadata


class AssayDataset:
    """Future Job for manipulating results"""

    def __init__(self, session: APISession, metadata: AssayMetadata):
        """
        init for AssayDataset.

        Parameters
        ----------
        session : APISession
            Session object for API communication.
        metadata : AssayMetadata
            Metadata object of the assay data.
        """
        self.session = session
        self.metadata = metadata
        self.page_size = config.BASE_PAGE_SIZE
        if self.page_size > 1000:
            self.page_size = 1000

    def __str__(self) -> str:
        return str(self.metadata)

    def __repr__(self) -> str:
        return repr(self.metadata)

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
    def measurement_names(self):
        return self.metadata.measurement_names

    @property
    def sequence_length(self):
        return self.metadata.sequence_length

    def __len__(self):
        return self.metadata.num_rows

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
        return assaydata.list_models(self.session, self.id)

    def update(
        self, assay_name: str | None = None, assay_description: str | None = None
    ) -> None:
        """
        Update the assay metadata.

        Parameters
        ----------
        assay_name : str, optional
            New name of the assay, by default None.
        assay_description : str, optional
            New description of the assay, by default None.

        Returns
        -------
        None
        """
        metadata = assaydata.assaydata_put(
            self.session,
            self.id,
            assay_name=assay_name,
            assay_description=assay_description,
        )
        self.metadata = metadata

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
        entries = assaydata.assaydata_page_get(
            self.session, self.id, page_offset=0, page_size=1
        )
        for row in entries.assaydata:
            row = [row.mut_sequence] + row.measurement_values
            rows.append(row)
        table = pd.DataFrame(rows, columns=["sequence"] + self.measurement_names)  # type: ignore
        return table

    def get_slice(self, start: int, end: int) -> pd.DataFrame:
        """
        Get a slice of assay data.

        Parameters
        ----------
        start : int
            Start index of the slice.
        end : int
            End index of the slice.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the slice of assay data.
        """
        rows = []
        page_size = self.page_size
        # loop over the range
        for i in range(start, end, page_size):
            # the last page might be smaller than the page size
            current_page_size = min(page_size, end - i)

            entries = assaydata.assaydata_page_get(
                self.session, self.id, page_offset=i, page_size=current_page_size
            )

            for row in entries.assaydata:
                row = [row.mut_sequence] + row.measurement_values
                rows.append(row)

        table = pd.DataFrame(rows, columns=["sequence"] + self.measurement_names)  # type: ignore
        return table
