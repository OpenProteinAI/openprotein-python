"""Proprietary PoET-2 model providing top-class performance on protein engineering tasks."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np

from openprotein.base import APISession
from openprotein.common import ModelMetadata, ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.prompt import Prompt, PromptAPI, Query
from openprotein.protein import Protein
from openprotein.utils import uuid

from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)
from .models import EmbeddingModel
from .poet import PoETModel

if TYPE_CHECKING:
    from openprotein.predictor import PredictorModel
    from openprotein.svd import SVDModel
    from openprotein.umap import UMAPModel


class PoET2Model(PoETModel, EmbeddingModel):
    """
    Class for OpenProtein's foundation model PoET 2.

    PoET functions are dependent on a prompt supplied via the prompt endpoints.

    Examples
    --------
    View specific model details (including supported tokens) with the `?` operator.

    Examples
    --------
    .. code-block:: python

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.embedding.poet2?
    """

    model_id = "poet-2"

    # TODO - Add model to explicitly require prompt_id
    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

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
            prompt_api = getattr(self.session, "prompt", None)
            assert isinstance(prompt_api, PromptAPI)
            query_ = prompt_api.create_query(query=query)
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
        decoder_type: Literal["mlm", "clm"] | None = None,
    ) -> EmbeddingsResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to embed.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g. mean). Default is ReductionType.MEAN.
        prompt : str or Prompt or None, optional
            Prompt or prompt_id or prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        decoder_type : {'mlm', 'clm'} or None, optional
            Decoder type. Default is None.

        Returns
        -------
        EmbeddingsResultFuture
            A future object that returns the embeddings of the submitted sequences.
        """
        query_id = self.__resolve_query(query=query)
        return super().embed(
            sequences=sequences,
            reduction=reduction,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
            decoder_type=decoder_type,
        )

    def logits(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        decoder_type: Literal["mlm", "clm"] | None = None,
    ) -> EmbeddingsResultFuture:
        """
        Compute logit embeddings for sequences using this model.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to analyze.
        prompt : str or Prompt or None, optional
            Prompt or prompt_id or prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        decoder_type : {'mlm', 'clm'} or None, optional
            Decoder type. Default is None.

        Returns
        -------
        EmbeddingsResultFuture
            A future object that returns the logits of the submitted sequences.
        """
        query_id = self.__resolve_query(query=query)
        return super().logits(
            sequences=sequences,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
            decoder_type=decoder_type,
        )

    def score(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        decoder_type: Literal["mlm", "clm"] | None = None,
    ) -> EmbeddingsScoreFuture:
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to score.
        prompt : str or Prompt or None, optional
            Prompt or prompt_id or prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        decoder_type : {'mlm', 'clm'} or None, optional
            Decoder type. Default is None.

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
            decoder_type=decoder_type,
        )

    def indel(
        self,
        sequence: bytes,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        decoder_type: Literal["mlm", "clm"] | None = None,
        insert: str | None = None,
        delete: list[int] | None = None,
        **kwargs,
    ) -> EmbeddingsScoreFuture:
        """
        Score all indels of the query sequence using the specified prompt.

        Parameters
        ----------
        sequence : bytes
            Sequence to analyze.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        decoder_type : {'mlm', 'clm'} or None, optional
            Decoder type. Default is None.
        insert : str or None, optional
            Insertion fragment at each site.
        delete : list of int or None, optional
            Range of size of fragment to delete at each site.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the indel-ed sequence.

        Raises
        ------
        ValueError
            If neither insert nor delete is provided.
        """
        query_id = self.__resolve_query(query=query)
        return super().indel(
            sequence=sequence,
            prompt=prompt,
            query_id=query_id,
            use_query_structure_in_decoder=use_query_structure_in_decoder,
            decoder_type=decoder_type,
            insert=insert,
            delete=delete,
        )

    def single_site(
        self,
        sequence: bytes,
        prompt: str | Prompt | None = None,
        query: str | bytes | Protein | Query | None = None,
        use_query_structure_in_decoder: bool = True,
        decoder_type: Literal["mlm", "clm"] | None = None,
    ) -> EmbeddingsScoreFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        sequence : bytes
            Sequence to analyze.
        prompt : str or Prompt or None, optional
            Prompt or prompt_id or prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        decoder_type : {'mlm', 'clm'} or None, optional
            Decoder type. Default is None.

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
            decoder_type=decoder_type,
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
        ensemble_weights: Sequence[float] | None = None,
        ensemble_method: Literal["arithmetic", "geometric"] | None = None,
    ) -> EmbeddingsGenerateFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt : str or Prompt
            Prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        num_samples : int, optional
            The number of samples to generate. Default is 100.
        temperature : float, optional
            The temperature for sampling. Higher values produce more random outputs. Default is 1.0.
        topk : float or None, optional
            The number of top-k residues to consider during sampling. Default is None.
        topp : float or None, optional
            The cumulative probability threshold for top-p sampling. Default is None.
        max_length : int, optional
            The maximum length of generated proteins. Default is 1000.
        seed : int or None, optional
            Seed for random number generation. Default is None.
        ensemble_weights : Sequence of float or None, optional
            Weights for combining likelihoods from multiple prompts in the ensemble.
            The length of this sequence must match the number of prompts.
            All weights must be finite. If ensemble_method is "arithmetic", then weights
            must also be non-negative, and have a non-zero sum.
        ensemble_method : {'arithmetic', 'geometric'} or None, optional
            Method used to combine likelihoods from multiple prompts in the ensemble.
            If "arithmetic", the weighted mean is used; if "geometric", the weighted
            geometric mean is used. If None (default), the method defaults to
            "arithmetic", but this behavior may change in the future.

        Returns
        -------
        EmbeddingsGenerateFuture
            A future object representing the status and information about the generation job.
        """
        query_id = self.__resolve_query(query=query)
        if ensemble_weights is not None:
            # NB: for now, ensemble_method is None -> ensemble_method == "arithmetic"
            if ensemble_method is None or (ensemble_method == "arithmetic"):
                assert all(w >= 0 for w in ensemble_weights)
                assert sum(ensemble_weights) >= 0
            assert np.isfinite(np.array(ensemble_weights)).all()
            if isinstance(prompt, Prompt):
                assert len(ensemble_weights) == prompt.num_replicates, (
                    f"Number of ensemble weights ({len(ensemble_weights)}) must be "
                    f"equal to the number of prompts ({prompt.num_replicates})"
                )
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
            ensemble_weights=ensemble_weights,
            ensemble_method=ensemble_method,
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

        This function will create an SVDModel based on the embeddings from this model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        sequences : list of bytes or list of str or None, optional
            Sequences to fit SVD. If None, assay must be provided.
        assay : AssayDataset or None, optional
            Assay containing sequences to fit SVD. Ignored if sequences are provided.
        n_components : int, optional
            Number of components in SVD. Determines output shapes. Default is 1024.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g. mean).
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.

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

        This function will create a UMAP based on the embeddings from this PoET model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        sequences : list of bytes or list of str or None, optional
            Sequences to fit UMAP. If None, assay must be provided.
        assay : AssayDataset or None, optional
            Assay containing sequences to fit UMAP. Ignored if sequences are provided.
        n_components : int, optional
            Number of components in UMAP fit. Determines output shapes. Default is 2.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g. mean). Default is ReductionType.MEAN.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.

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
        Fit a Gaussian Process (GP) on assay using this embedding model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata or AssayDataset or str
            Assay to fit GP on.
        properties : list of str
            Properties in the assay to fit the GP on.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition PoET model.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
        use_query_structure_in_decoder : bool, optional
            Whether to use query structure in decoder. Default is True.
        **kwargs
            Additional keyword arguments.

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
