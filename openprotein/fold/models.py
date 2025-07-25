"""Fold model representations which can be used directly for creating structure predictions."""

from openprotein.base import APISession
from openprotein.common import ModelMetadata

from . import api
from .future import FoldComplexResultFuture, FoldResultFuture


class FoldModel:

    # overridden by subclasses
    # used to get correct fold model
    model_id: list[str] | str

    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.id

    @classmethod
    def get_model(cls):
        """
        Get the model_id(s) for this FoldModel subclass.

        Returns
        -------
        list of str
            List of model_id strings associated with this class.
        """
        if isinstance(cls.model_id, str):
            return [cls.model_id]
        return cls.model_id

    @classmethod
    def create(
        cls,
        session: APISession,
        model_id: str,
        default: type["FoldModel"] | None = None,
        **kwargs,
    ):
        """
        Create and return an instance of the appropriate FoldModel subclass based on the model_id.

        Parameters
        ----------
        session : APISession
            The API session to use.
        model_id : str
            The model identifier.
        default : type[FoldModel] or None, optional
            Default FoldModel subclass to use if no match is found.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the model constructor.

        Returns
        -------
        FoldModel
            An instance of the appropriate FoldModel subclass.

        Raises
        ------
        ValueError
            If no suitable FoldModel subclass is found and no default is provided.
        """
        # Dynamically discover all subclasses of FoldModel
        model_classes = FoldModel.__subclasses__()

        # Find the FoldModel class that matches the model_id
        for model_class in model_classes:
            if model_id in model_class.get_model():
                return model_class(session=session, model_id=model_id, **kwargs)
        # default to FoldModel
        if default is not None:
            try:
                return default(session=session, model_id=model_id, **kwargs)
            except:
                pass
        raise ValueError(f"Unsupported model_id type: {model_id}")

    @property
    def metadata(self):
        """
        ModelMetadata : Model metadata for this model.
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
        return api.fold_model_get(self.session, self.id)

    def fold(self, **kwargs) -> FoldResultFuture | FoldComplexResultFuture:
        """
        Fold a sequence using this model.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying
            `fold` request.

        Returns
        -------
        FoldResultFuture or FoldComplexResultFuture
            Future object representing the fold result.
        """
        return FoldResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=(
                    model_id
                    if isinstance(model_id := self.model_id, str)
                    else model_id[0]
                ),
                **kwargs,
            ),
        )
