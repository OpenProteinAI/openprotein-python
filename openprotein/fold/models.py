"""Fold model representations which can be used directly for creating structure predictions."""

from collections.abc import Iterator

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.molecules import Complex, Protein

from . import api
from .common import FoldInput, _as_entities, normalize_inputs, serialize_input
from .future import FoldResultFuture

__all__ = ["FoldModel"]


def _attaches_msa(sequences: FoldInput) -> bool:
    """Whether any given protein has an MSA or single sequence mode attached."""
    for entity in _as_entities(sequences):
        if isinstance(entity, Complex):
            proteins = list(entity.get_proteins().values())
        elif isinstance(entity, Protein):
            proteins = [entity]
        else:
            continue
        if any(protein.msa is not None for protein in proteins):
            return True
    return False


def _subclasses(cls: type) -> Iterator[type]:
    """Every subclass of ``cls``, at any depth. ``__subclasses__`` is direct only."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


class FoldModel:
    """A fold model that predicts a structure from sequences."""

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
        model_id = cls.model_id
        if isinstance(model_id, str):
            return [model_id]
        if isinstance(model_id, list):
            return model_id
        # No id declared, so this class serves nothing.
        return []

    @property
    def model_id(self) -> str:
        # self.id, not self.model_id - that would call this property again.
        return self.id

    @classmethod
    def create(
        cls,
        session: APISession,
        model_id: str,
        default: type["FoldModel"] | None = None,
        **kwargs,
    ):
        """
        Create and return an instance of the subclass serving ``model_id``.

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
        for model_class in _subclasses(FoldModel):
            if model_id in model_class.get_model():
                return model_class(session=session, model_id=model_id, **kwargs)
        if default is not None:
            try:
                return default(session=session, model_id=model_id, **kwargs)
            except Exception:
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

    def fold(
        self,
        sequences: FoldInput,
        **kwargs,
    ) -> FoldResultFuture:
        """
        Fold sequences using this model.

        Parameters
        ----------
        sequences : FoldInput
            Sequences, proteins or complexes to fold. Every protein is sent
            with an ``msa_id`` when any of them has an MSA or
            ``Protein.single_sequence_mode`` attached. An empty list sends no
            sequences, for models whose inputs are other parameters.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the underlying
            `fold` request.

        Returns
        -------
        FoldResultFuture
            Future object representing the fold result.
        """
        complexes = normalize_inputs(sequences)
        serialized = (
            serialize_input(self.session, complexes, needs_msa=_attaches_msa(sequences))
            if complexes
            else None
        )
        return FoldResultFuture.create(
            session=self.session,
            job=api.fold_models_post(
                session=self.session,
                model_id=self.model_id,
                sequences=serialized,
                **kwargs,
            ),
            complexes=complexes or None,
        )
