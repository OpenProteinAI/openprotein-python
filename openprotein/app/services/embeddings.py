from openprotein.base import APISession
from openprotein.app.models.embeddings import EmbeddingModel, OpenProteinModel, ESMModel, PoETModel, EmbeddingResultFuture
from openprotein.api import embedding

class EmbeddingsAPI:
    """
    This class defines a high level interface for accessing the embeddings API.

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
    prot_seq: OpenProteinModel
    rotaprot_large_uniref50w: OpenProteinModel
    rotaprot_large_uniref90_ft: OpenProteinModel
    poet: PoETModel

    esm1b: ESMModel # alias
    esm1b_t33_650M_UR50S: ESMModel

    esm1v: ESMModel # alias
    esm1v_t33_650M_UR90S_1: ESMModel
    esm1v_t33_650M_UR90S_2: ESMModel
    esm1v_t33_650M_UR90S_3: ESMModel
    esm1v_t33_650M_UR90S_4: ESMModel
    esm1v_t33_650M_UR90S_5: ESMModel

    esm2: ESMModel # alias
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
        # Setup aliases
        self.esm1b = self.esm1b_t33_650M_UR50S
        self.esm1v = self.esm1v_t33_650M_UR90S_1
        self.esm2 = self.esm2_t33_650M_UR50D

    def list_models(self) -> list[EmbeddingModel]:
        """list models available for creating embeddings of your sequences"""
        models = []
        for model_id in embedding.list_models(self.session):
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

    def __get_results(self, job) -> EmbeddingResultFuture:
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
        return EmbeddingResultFuture(job=job, session=self.session)
