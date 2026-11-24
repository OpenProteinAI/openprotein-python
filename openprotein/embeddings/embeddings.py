"""Embeddings API providing the interface for creating embeddings using protein language models."""

from openprotein.base import APISession

from . import api
from .ablang import AbLang2Model
from .esm import ESMModel
from .future import EmbeddingsResultFuture
from .models import EmbeddingModel
from .openprotein import OpenProteinModel
from .poet import PoETModel
from .poet2 import PoET2Model

__all__ = ["EmbeddingsAPI"]


_LEGACY_MODELS: dict[str, tuple[type[EmbeddingModel], str]] = {
    "poet2": (PoET2Model, "poet-2"),
    "poet": (PoETModel, "poet"),
    "prot_seq": (OpenProteinModel, "prot-seq"),
    "rotaprot_large_uniref50w": (OpenProteinModel, "rotaprot-large-uniref50w"),
    "rotaprot_large_uniref90_ft": (OpenProteinModel, "rotaprot-large-uniref90-ft"),
    "ablang2": (AbLang2Model, "ablang2"),
    **{name.replace("-", "_"): (ESMModel, name) for name in ESMModel.get_model()},
}
_LEGACY_ALIASES = {
    "poet_2": "poet2",
    "esm1b": "esm1b_t33_650M_UR50S",
    "esm1v": "esm1v_t33_650M_UR90S_1",
    "esm2": "esm2_t33_650M_UR50D",
}


class EmbeddingsAPI:
    """
    Embeddings API providing the interface for creating embeddings using protein language models.

    You can access all our models either via :meth:`get_model` or directly through the session's embedding attribute using the model's ID and the desired method. For example, to use the attention method on the protein sequence model, you would use ``session.embedding.prot_seq.attn()``.

    Examples
    --------
    Accessing a model's method:

    .. code-block:: python

        # To call the attention method on the protein sequence model:
        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.prot_seq.attn()

    Using the `get_model` method:

    .. code-block:: python

        # Get a model instance by name:
        import openprotein
        session = openprotein.connect(username="user", password="password")
        # list available models:
        print(session.embedding.list_models() )
        # init model by name
        model = session.embedding.get_model('prot-seq')
    """

    # Static declarations for typing and autocomplete.

    #: PoET-2 model
    poet2: PoET2Model
    #: PoET model
    poet: PoETModel
    #: Prot-seq model
    prot_seq: OpenProteinModel
    #: Rotaprot model trained on UniRef50
    rotaprot_large_uniref50w: OpenProteinModel
    #: Rotaprot model trained on UniRef90
    rotaprot_large_uniref90_ft: OpenProteinModel
    poet_2: PoET2Model
    ablang2: AbLang2Model

    #: ESM1b model
    esm1b: ESMModel  # alias
    esm1b_t33_650M_UR50S: ESMModel

    #: ESM1v model
    esm1v: ESMModel  # alias
    esm1v_t33_650M_UR90S_1: ESMModel
    esm1v_t33_650M_UR90S_2: ESMModel
    esm1v_t33_650M_UR90S_3: ESMModel
    esm1v_t33_650M_UR90S_4: ESMModel
    esm1v_t33_650M_UR90S_5: ESMModel

    #: ESM2 model
    esm2: ESMModel  # alias
    esm2_t12_35M_UR50D: ESMModel
    esm2_t30_150M_UR50D: ESMModel
    esm2_t33_650M_UR50D: ESMModel
    esm2_t36_3B_UR50D: ESMModel
    esm2_t6_8M_UR50D: ESMModel

    #: ESM-C (Cambrian) model
    esmc_300m: ESMModel
    esmc_600m: ESMModel
    esmc_6b: ESMModel

    def __init__(self, session: APISession):
        self.session = session

    def __getattr__(self, name: str) -> EmbeddingModel:
        """Resolve a model handle under the embeddings namespace.

        Names without a typed handle resolve against :meth:`list_models`.
        """
        if name.startswith("_") or "session" not in self.__dict__:
            raise AttributeError(name)
        canonical = _LEGACY_ALIASES.get(name, name)
        if canonical not in self.__dict__:
            self.__dict__[canonical] = self._resolve(canonical)
        return self.__dict__[canonical]

    def _resolve(self, name: str) -> EmbeddingModel:
        entry = _LEGACY_MODELS.get(name)
        if entry is not None:
            cls, model_id = entry
            return cls(self.session, model_id)
        for model in self.list_models():
            if model.id.replace("-", "_") == name:
                return model
        raise AttributeError(name)

    def __dir__(self):
        return sorted(
            set(super().__dir__()) | _LEGACY_MODELS.keys() | _LEGACY_ALIASES.keys()
        )

    def list_models(self) -> list[EmbeddingModel]:
        """list models available for creating embeddings of your sequences"""
        models = []
        for metadata in api.list_models(self.session, verbose=True):
            models.append(
                EmbeddingModel.create(
                    session=self.session,
                    model_id=metadata.id,
                    default=EmbeddingModel,
                    metadata=metadata,
                )
            )
        return models

    def get_model(self, name: str) -> EmbeddingModel:
        """
        Get model by model_id. 

        ProtembedModel allows all the usual job manipulation: \
            e.g. making POST and GET requests for this model specifically. 


        Parameters
        ----------
        model_id : str
            the model identifier

        Returns
        -------
        ProtembedModel
            The model

        Raises
        ------
        AttributeError
            If no listed model has that name.
        HTTPError
            If the GET request does not succeed.
        """
        model_name = name.replace("-", "_")
        return getattr(self, model_name)

    def __get_results(self, job) -> EmbeddingsResultFuture:
        """
        Retrieves the results of an embedding job.

        Parameters
        ----------
        job : Job
            The embedding job whose results are to be retrieved.

        Returns
        -------
        EmbeddingResultFuture
            An instance of EmbeddingResultFuture
        """
        return EmbeddingsResultFuture(job=job, session=self.session)
