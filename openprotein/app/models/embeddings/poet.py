import warnings
from typing import TYPE_CHECKING

from openprotein.api import embedding
from openprotein.base import APISession
from openprotein.schemas import ModelMetadata, ReductionType

from ..assaydata import AssayDataset, AssayMetadata
from ..prompt import Prompt
from .base import EmbeddingModel
from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)

if TYPE_CHECKING:
    from ..predictor import PredictorModel
    from ..svd import SVDModel
    from ..umap import UMAPModel


class PoETModel(EmbeddingModel):
    """
    Class for OpenProtein's foundation model PoET - NB. PoET functions are dependent on a prompt supplied via the align endpoints.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.poet.<embeddings_method>


    """

    model_id = "poet"

    # TODO - Add model to explicitly require prompt_id
    def __init__(
        self, session: APISession, model_id: str, metadata: ModelMetadata | None = None
    ):
        self.session = session
        self.id = model_id
        self._metadata = metadata
        # could add prompt here?

    def embed(
        self,
        sequences: list[bytes],
        prompt: str | Prompt | None = None,
        reduction: ReductionType | None = ReductionType.MEAN,
        **kwargs,
    ) -> EmbeddingsResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to embed.
        reduction: str
            embeddings reduction to use (e.g. mean)

        Returns
        -------
        EmbeddingResultFuture
            A future object that returns the embeddings of the submitted sequences.
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
        logit embeddings for sequences using this model.

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.

        Returns
        -------
        EmbeddingResultFuture
            A future object that returns the logits of the submitted sequences.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return super().logits(sequences=sequences, prompt_id=prompt_id)

    def attn(self):
        """Not Available for Poet."""
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
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        sequence: list[bytes]
            Sequences to score.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the submitted sequences.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=embedding.request_score_post(
                session=self.session,
                model_id=self.id,
                prompt_id=prompt_id,
                sequences=sequences,
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
        prompt: str | Prompt
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        sequence: bytes
            Sequence to analyse.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the mutated sequence.
        """
        if prompt is None:
            prompt_id = None
        else:
            prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=embedding.request_score_single_site_post(
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
        prompt: str | Prompt
            Prompt from an align workflow to condition Poet model
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
        prompt_id = prompt if isinstance(prompt, str) else prompt.id
        return EmbeddingsGenerateFuture.create(
            session=self.session,
            job=embedding.request_generate_post(
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

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: str | Prompt
            prompt from an align workflow to condition Poet model
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
        Fit a GP on assay using this embedding model and hyperparameters.

        Parameters
        ----------
        assay : AssayMetadata | str
            Assay to fit GP on.
        properties: list[str]
            Properties in the assay to fit the gp on.
        reduction : str
            Type of embedding reduction to use for computing features. PLM must use reduction.

        Returns
        -------
        PredictorModel
            A future that represents the trained predictor model.
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
