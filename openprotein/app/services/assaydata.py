import io

import pandas as pd
from openprotein.api import assaydata
from openprotein.app.models import AssayDataset, AssayMetadata
from openprotein.base import APISession


class AssayDataAPI:
    """API interface for calling AssayData endpoints"""

    def __init__(self, session: APISession):
        """
        init the DataAPI.

        Parameters
        ----------
        session : APISession
            Session object for API communication.
        """
        self.session = session

    def list(self) -> list[AssayDataset]:
        """
        List all assay datasets.

        Returns
        -------
        List[AssayDataset]
            List of all assay datasets.
        """
        metadata = assaydata.assaydata_list(self.session)
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
        metadata = assaydata.assaydata_post(
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
            self.session, assaydata.get_assay_metadata(self.session, assay_id)
        )

    def load_assay(self, assay_id: str) -> AssayDataset:
        """
        Reload a Submitted job to resume from where you left off!


        Parameters
        ----------
        assay_id : str
            The identifier of the job whose details are to be loaded.

        Returns
        -------
        Job
            Job

        Raises
        ------
        HTTPError
            If the request to the server fails.
        InvalidJob
            If the Job is of the wrong type

        """
        metadata = self.get(assay_id)
        # if job_details.job_type != JobType.train:
        #    raise InvalidJob(f"Job {job_id} is not of type {JobType.train}")
        return AssayDataset(
            self.session,
            metadata,
        )

    def __len__(self) -> int:
        """
        Get the number of assay datasets.

        Returns
        -------
        int
            Number of assay datasets.
        """
        return len(self.list())
