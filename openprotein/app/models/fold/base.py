from abc import ABC, abstractmethod

from openprotein.api import fold
from openprotein.base import APISession
from openprotein.schemas import ModelMetadata


class FoldModel(ABC):
    # overridden by subclasses
    # get correct fold model
    model_id: list[str] | str = "protfold"

    def __init__(
        self, session: APISession, model_id: str, metadata: ModelMetadata | None = None
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata

    @classmethod
    def get_model(cls):
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id

    @staticmethod
    def create(
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
        default: type["FoldModel"] | None = None,
    ):
        """
        Create and return an instance of the appropriate Future class based on the job type.

        Returns:
        - An instance of the appropriate Future class.
        """
        # Dynamically discover all subclasses of FutureBase
        model_classes = FoldModel.__subclasses__()

        # Find the FoldModel class that matches the job type
        for model_class in model_classes:
            if model_id in model_class.get_model():
                return model_class(
                    session=session, model_id=model_id, metadata=metadata
                )
        # default to FoldModel
        if default is not None:
            try:
                return default(session=session, model_id=model_id, metadata=metadata)
            except:
                pass
        raise ValueError(f"Unsupported model_id type: {model_id}")

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    @property
    def metadata(self):
        return self.get_metadata()

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata for this model.

        Returns
        -------
            ModelMetadata
        """
        if self._metadata is not None:
            return self._metadata
        self._metadata = fold.fold_model_get(self.session, self.id)
        return self._metadata

    @abstractmethod
    def fold(self, sequence: str, **kwargs):
        pass
