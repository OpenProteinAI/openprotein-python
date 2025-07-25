import io

import pandas as pd

from openprotein.base import APISession

from . import api
from .assaydataset import AssayDataset


class DataAPI:
    """API interface for calling AssayData endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def list(self) -> list[AssayDataset]:
        """
        List all assay datasets.

        Returns
        -------
        List[AssayDataset]
            List of all assay datasets.
        """
        metadata = api.assaydata_list(self.session)
        return [AssayDataset(self.session, x) for x in metadata]

    def create(
        self, table: pd.DataFrame, name: str, description: str | None = None
    ) -> AssayDataset:
        """
        Create a new assay dataset.

        Parameters
        ----------
        table : pd.DataFrame
            DataFrame containing the assay data.
        name : str
            Name of the assay dataset.
        description : str, optional
            Description of the assay dataset, by default None.

        Returns
        -------
        AssayDataset
            Created assay dataset.
        """
        stream = io.BytesIO()
        table.to_csv(stream, index=False)
        stream.seek(0)
        metadata = api.assaydata_post(
            self.session, stream, name, assay_description=description
        )
        metadata.sequence_length = len(table["sequence"].values[0])
        return AssayDataset(self.session, metadata)

    def get(self, assay_id: str, verbose: bool = False) -> AssayDataset:
        """
        Get an assay dataset by its ID.

        Parameters
        ----------
        assay_id : str
            ID of the assay dataset.

        Returns
        -------
        AssayDataset
            Assay dataset with the specified ID.

        Raises
        ------
        KeyError
            If no assay dataset with the given ID is found.
        """
        return AssayDataset(
            session=self.session,
            metadata=api.get_assay_metadata(self.session, assay_id),
        )

    load_assay = get

    def __len__(self) -> int:
        """
        Get the number of assay datasets.

        Returns
        -------
        int
            Number of assay datasets.
        """
        return len(self.list())
