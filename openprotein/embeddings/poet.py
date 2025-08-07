"""Original PoET model handling various protein engineering tasks."""

from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.common import ModelMetadata, ReductionType
from openprotein.data import AssayDataset, AssayMetadata
from openprotein.prompt import Prompt

from . import api
from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)
from .models import EmbeddingModel

if TYPE_CHECKING:
    from openprotein.predictor import PredictorModel
    from openprotein.svd import SVDModel
    from openprotein.umap import UMAPModel


class PoETModel(EmbeddingModel):
    """
    Class for OpenProtein's foundation model PoET.

    Note
    ----
    PoET functions are dependent on a prompt supplied via the prompt endpoints.

    Examples
    --------
    View specific model details (including supported tokens) with the `?` operator.

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.embedding.poet.<embeddings_method>
    """

    model_id = "poet"

    # TODO - Add model to explicitly require prompt_id
    def __init__(
        self,
        session: APISession,
        model_id: str,
        metadata: ModelMetadata | None = None,
    ):
        super().__init__(session=session, model_id=model_id, metadata=metadata)

    def embed(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        reduction: ReductionType | None = ReductionType.MEAN,
        **kwargs,
    ) -> EmbeddingsResultFuture:
        """
        Embed sequences using the PoET model.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to embed.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g., mean). Default is ReductionType.MEAN.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsResultFuture
            Future object that returns the embeddings of the submitted sequences.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().embed(
            sequences=sequences,
            reduction=reduction,
            prompt_id=prompt_id,
            **kwargs,
        )

    def logits(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        **kwargs,
    ) -> EmbeddingsResultFuture:
        """
        Compute logits for sequences using the PoET model.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to analyze.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsResultFuture
            Future object that returns the logits of the submitted sequences.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().logits(sequences=sequences, prompt_id=prompt_id, **kwargs)

    def attn(self):
        """
        Attention is not available for PoET.

        Raises
        ------
        ValueError
            Always raised, as attention is not supported for PoET.

        :meta private:
        """
        raise ValueError("Attn not yet supported for poet")

    def score(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        **kwargs,
    ) -> EmbeddingsScoreFuture:
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to score.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsScoreFuture
            Future object that returns the scores of the submitted sequences.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=api.request_score_post(
                session=self.session,
                model_id=self.id,
                prompt_id=prompt_id,
                sequences=sequences,
                **kwargs,
            ),
        )

    def indel(
        self,
        sequence: bytes,
        prompt: str | Prompt | None = None,
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
        insert : str or None, optional
            Insertion fragment at each site.
        delete : list of int or None, optional
            Range of size of fragment to delete at each site.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsScoreFuture
            Future object that returns the scores of the indel-ed sequence.

        Raises
        ------
        ValueError
            If neither insert nor delete is provided.
        """
        if not insert and not delete:
            raise ValueError("Expected insert and/or delete to be provided")
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=api.request_score_indel_post(
                session=self.session,
                model_id=self.id,
                base_sequence=sequence,
                prompt_id=prompt_id,
                insert=insert,
                delete=delete,
                **kwargs,
            ),
        )

    def single_site(
        self,
        sequence: bytes,
        prompt: str | Prompt | None = None,
        **kwargs,
    ) -> EmbeddingsScoreFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        sequence : bytes
            Sequence to analyze.
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsScoreFuture
            Future object that returns the scores of the mutated sequence.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=api.request_score_single_site_post(
                session=self.session,
                model_id=self.id,
                base_sequence=sequence,
                prompt_id=prompt_id,
                **kwargs,
            ),
        )

    def generate(
        self,
        prompt: str | Prompt,
        num_samples: int = 100,
        temperature: float = 1.0,
        topk: float | None = None,
        topp: float | None = None,
        max_length: int = 1000,
        seed: int | None = None,
        **kwargs,
    ) -> EmbeddingsGenerateFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt : str or Prompt
            Prompt from an align workflow to condition the PoET model.
        num_samples : int, optional
            Number of samples to generate. Default is 100.
        temperature : float, optional
            Temperature for sampling. Higher values produce more random outputs. Default is 1.0.
        topk : float or None, optional
            Number of top-k residues to consider during sampling. Default is None.
        topp : float or None, optional
            Cumulative probability threshold for top-p sampling. Default is None.
        max_length : int, optional
            Maximum length of generated proteins. Default is 1000.
        seed : int or None, optional
            Seed for random number generation. Default is None.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        EmbeddingsGenerateFuture
            Future object representing the status and information about the generation job.
        """
        prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsGenerateFuture.create(
            session=self.session,
            job=api.request_generate_post(
                session=self.session,
                model_id=self.id,
                num_samples=num_samples,
                temperature=temperature,
                topk=topk,
                topp=topp,
                max_length=max_length,
                random_seed=seed,
                prompt_id=prompt_id,
                **kwargs,
            ),
        )

    def fit_svd(
        self,
        prompt: str | Prompt | None = None,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
        **kwargs,
    ) -> "SVDModel":
        """
        Fit an SVD on the embedding results of PoET.

        This function creates an SVDModel based on the embeddings from this model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        sequences : list of bytes or list of str or None, optional
            Sequences to use for SVD.
        assay : AssayDataset or None, optional
            Assay dataset to use for SVD.
        n_components : int, optional
            Number of components in SVD. Determines output shapes. Default is 1024.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g., mean).
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        SVDModel
            Future that represents the fitted SVD model.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().fit_svd(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt_id=prompt_id,
            **kwargs,
        )

    def fit_umap(
        self,
        prompt: str | Prompt | None = None,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 2,
        reduction: ReductionType | None = ReductionType.MEAN,
        **kwargs,
    ) -> "UMAPModel":
        """
        Fit a UMAP on assay using PoET and hyperparameters.

        This function creates a UMAP based on the embeddings from this PoET model
        as well as the hyperparameters specified in the arguments.

        Parameters
        ----------
        prompt : str or Prompt or None, optional
            Prompt from an align workflow to condition the PoET model.
        sequences : list of bytes or list of str or None, optional
            Optional sequences to fit UMAP with. Either use sequences or assay. Sequences is preferred.
        assay : AssayDataset or None, optional
            Optional assay containing sequences to fit UMAP with. Either use sequences or assay. Ignored if sequences are provided.
        n_components : int, optional
            Number of components in UMAP fit. Determines output shapes. Default is 2.
        reduction : ReductionType or None, optional
            Embeddings reduction to use (e.g., mean). Default is ReductionType.MEAN.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        UMAPModel
            Future that represents the fitted UMAP model.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().fit_umap(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt_id=prompt_id,
            **kwargs,
        )

    def fit_gp(
        self,
        assay: AssayMetadata | AssayDataset | str,
        properties: list[str],
        prompt: str | Prompt | None = None,
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
            Prompt from an align workflow to condition the PoET model.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        PredictorModel
            Future that represents the trained predictor model.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().fit_gp(
            assay=assay,
            properties=properties,
            prompt_id=prompt_id,
            **kwargs,
        )
