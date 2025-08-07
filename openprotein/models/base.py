"""Base protein models for working with proteins."""

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.jobs import Future


class ProteinModel:
    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        self.__doc__ = self.__fmt_doc()

    def __fmt_doc(self):
        summary = str(self.metadata.description.summary)
        return f"""\t{summary}
        \t max_sequence_length = {self.metadata.max_sequence_length}
        \t supported outputs = {self.metadata.output_types}
        \t supported tokens = {self.metadata.input_tokens} 
        """

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    @property
    def metadata(self):
        """
        ModelMetadata for this model.

        Returns
        -------
        ModelMetadata
            The metadata associated with this model.
        """
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for this model.

        Returns
        -------
        ModelMetadata
            The metadata associated with this model.
        """
        raise NotImplementedError("`get_metadata` not implemented for this model")

    def predict(self, *args, **kwargs) -> Future:
        """
        Alias for the `design` method to conform to the base ProteinModel.
        """
        raise NotImplementedError("`predict` not implemented for this model")
