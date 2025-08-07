"""Embeddings API providing the interface for creating embeddings using protein language models."""

from openprotein.base import APISession

from . import api
from .esm import ESMModel
from .future import EmbeddingsResultFuture
from .models import EmbeddingModel
from .openprotein import OpenProteinModel
from .poet import PoETModel
from .poet2 import PoET2Model


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

    # added for static typing, eg pylance, for autocomplete
    # at init these are all overwritten.

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

    def __init__(self, session: APISession):
        self.session = session
        # dynamically add models from  api list
        self._load_models()

    def _load_models(self):
        # Dynamically add model instances as attributes - precludes any drift
        models = self.list_models()
        for model in models:
            model_name = model.id.replace("-", "_")  # hyphens out
            setattr(self, model_name, model)
        # Setup aliases safely
        if getattr(self, "esm1b_t33_650M_UR50S", None):
            self.esm1b = self.esm1b_t33_650M_UR50S
        if getattr(self, "esm1v_t33_650M_UR90S_1", None):
            self.esm1v = self.esm1v_t33_650M_UR90S_1
        if getattr(self, "esm2_t33_650M_UR50D", None):
            self.esm2 = self.esm2_t33_650M_UR50D
        if getattr(self, "poet_2", None):
            self.poet2 = self.poet_2

    def list_models(self) -> list[EmbeddingModel]:
        """list models available for creating embeddings of your sequences"""
        models = []
        for model_id in api.list_models(self.session):
            models.append(
                EmbeddingModel.create(
                    session=self.session, model_id=model_id, default=EmbeddingModel
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
