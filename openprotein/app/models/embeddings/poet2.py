from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.protein import Protein
from openprotein.schemas import ModelMetadata, ReductionType
from openprotein.utils import uuid

from ..assaydata import AssayDataset, AssayMetadata
from ..prompt import Prompt, Query
from .base import EmbeddingModel
from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)
from .poet import PoETModel

if TYPE_CHECKING:
    from openprotein import OpenProtein

    from ..predictor import PredictorModel
    from ..svd import SVDModel
    from ..umap import UMAPModel


class PoET2Model(PoETModel, EmbeddingModel):
    """
    Class for OpenProtein's foundation model PoET 2 - NB. PoET functions are dependent on a prompt supplied via the align endpoints.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.poet2.<embeddings_method>


    """

    model_id = "poet-2"

    # TODO - Add model to explicitly require prompt_id
    def __init__(
        self,
        session: "OpenProtein",
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        # could add prompt here?

    def __resolve_query(
        self,
        query: str | bytes | Protein | Query | None = None,
    ) -> str | None:
        if query is None:
            query_id = None
        elif (
            isinstance(query, Protein)
            or isinstance(query, bytes)
            or (isinstance(query, str) and not uuid.is_valid_uuid(query))
        ):
            query_ = self.session.prompt.create_query(query=query)
            query_id = query_.id
        else:
            query_id = query if isinstance(query, str) else query.id
        return query_id

    def embed(
        self,
        sequences: list[bytes],
        reduction: ReductionType | None = ReductionType.MEAN,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> EmbeddingsResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        sequence : bytes
            Sequence to embed.
        reduction: str
            embeddings reduction to use (e.g. mean)
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional

        Returns
        -------
        EmbeddingResultFuture
            A future object that returns the embeddings of the submitted sequences.
        """
        query_id = self.__resolve_query(query=query)
        return super().embed(
            sequences=sequences,
            reduction=reduction,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def logits(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> EmbeddingsResultFuture:
        """
        logit embeddings for sequences using this model.

        Parameters
        ----------
        sequence : bytes
            Sequence to analyse.
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional

        Returns
        -------
        EmbeddingResultFuture
            A future object that returns the logits of the submitted sequences.
        """
        query_id = self.__resolve_query(query=query)
        return super().logits(
            sequences=sequences,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def score(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> EmbeddingsScoreFuture:
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        sequence: list[bytes]
            Sequences to score.
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the submitted sequences.
        """
        query_id = self.__resolve_query(query=query)
        return super().score(
            sequences=sequences,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def single_site(
        self,
        sequence: bytes,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> EmbeddingsScoreFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        sequence: bytes
            Sequence to analyse.
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the mutated sequence.
        """
        query_id = self.__resolve_query(query=query)
        return super().single_site(
            sequence=sequence,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def generate(
        self,
        prompt: str | Prompt,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        num_samples: int = 100,
        temperature: float = 1.0,
        topk: float | None = None,
        topp: float | None = None,
        max_length: int = 1000,
        seed: int | None = None,
    ) -> EmbeddingsGenerateFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional
        num_samples: int, optional
            The number of samples to generate, by default 100.
        temperature: float, optional
            The temperature for sampling. Higher values produce more random outputs, by default 1.0.
        topk: int, optional
            The number of top-k residues to consider during sampling, by default None.
        topp: float, optional
            The cumulative probability threshold for top-p sampling, by default None.
        max_length: int, optional
            The maximum length of generated proteins, by default 1000.
        seed: int, optional
            Seed for random number generation, by default a random number.

        Returns
        -------
        EmbeddingsGenerateFuture
            A future object representing the status and information about the generation job.
        """
        query_id = self.__resolve_query(query=query)
        return super().generate(
            prompt=prompt,
            num_samples=num_samples,
            temperature=temperature,
            topk=topk,
            topp=topp,
            max_length=max_length,
            seed=seed,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def fit_svd(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> "SVDModel":
        """
        Fit an SVD on the embedding results of PoET. 

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional
        sequences : List[bytes] 
            sequences to SVD
        n_components: int 
            number of components in SVD. Will determine output shapes
        reduction: str
            embeddings reduction to use (e.g. mean)


        Returns
        -------
        SVDModel
            A future that represents the fitted SVD model.
        """
        query_id = self.__resolve_query(query=query)
        return super().fit_svd(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def fit_umap(
        self,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 2,
        reduction: ReductionType | None = ReductionType.MEAN,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
    ) -> "UMAPModel":
        """
        Fit a UMAP on assay using PoET and hyperparameters.

        This function will create a UMAP based on the embeddings from this PoET model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
        sequences : list[bytes] | None
            Optional sequences to fit UMAP with. Either use sequences or assay. sequences is preferred.
        assay: AssayDataset | None
            Optional assay containing sequences to fit UMAP with. Either use sequences or assay. Ignored if sequences are provided.
        n_components: int
            Number of components in UMAP fit. Will determine output shapes. Defaults to 2.
        reduction: ReductionType | None
            Embeddings reduction to use (e.g. mean). Defaults to MEAN.

        Returns
        -------
        UMAPModel
            A future that represents the fitted UMAP model.
        """
        query_id = self.__resolve_query(query=query)
        return super().fit_umap(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
        )

    def fit_gp(
        self,
        assay: AssayMetadata | AssayDataset | str,
        properties: list[str],
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        **kwargs,
    ) -> "PredictorModel":
        """
        Fit a GP on assay using this embedding model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata | str
            Assay to fit GP on.
        properties: list[str]
            Properties in the assay to fit the gp on.
        reduction : str
            Type of embedding reduction to use for computing features. PLM must use reduction.
        query: str | bytes | Protein | Query | None
            Query to use with prompt. Optional

        Returns
        -------
        PredictorModel
            A future that represents the trained predictor model.
        """
        query_id = self.__resolve_query(query=query)
        return super().fit_gp(
            assay=assay,
            properties=properties,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
            **kwargs,
        )
