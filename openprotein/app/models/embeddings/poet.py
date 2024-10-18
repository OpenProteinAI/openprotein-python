from typing import TYPE_CHECKING

from openprotein.api import embedding, poet
from openprotein.base import APISession
from openprotein.schemas import (
    ModelMetadata,
    PoetGenerateJob,
    PoetScoreJob,
    PoetSSPJob,
    ReductionType,
)

from ..align import PromptFuture
from ..assaydata import AssayDataset, AssayMetadata
from ..deprecated.poet import PoetGenerateFuture, PoetScoreFuture, PoetSingleSiteFuture
from ..futures import Future
from .base import EmbeddingModel
from .future import EmbeddingResultFuture, EmbeddingsScoreResultFuture

if TYPE_CHECKING:
    from ..predictor import PredictorModel
    from ..svd import SVDModel


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
        self.deprecated = self.Deprecated(session=session)
        # could add prompt here?

    def embed(
        self,
        prompt: str | PromptFuture,
        sequences: list[bytes],
        reduction: ReductionType | None = ReductionType.MEAN,
    ) -> EmbeddingResultFuture:
        """
        Embed sequences using this model.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to embed.
        reduction: str
            embeddings reduction to use (e.g. mean)
        Returns
        -------
            EmbeddingResultFuture
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().embed(
            sequences=sequences, reduction=reduction, prompt_id=prompt_id
        )

    def logits(
        self,
        prompt: str | PromptFuture,
        sequences: list[bytes],
    ) -> EmbeddingResultFuture:
        """
        logit embeddings for sequences using this model.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.

        Returns
        -------
            EmbeddingResultFuture
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().logits(
            sequences=sequences,
            prompt_id=prompt_id,
        )

    def attn(self):
        """Not Available for Poet."""
        raise ValueError("Attn not yet supported for poet")

    def score(
        self, prompt: str | PromptFuture, sequences: list[bytes]
    ) -> EmbeddingsScoreResultFuture:
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.
        Returns
        -------
            ScoreFuture
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return EmbeddingsScoreResultFuture.create(
            session=self.session,
            job=embedding.request_score_post(
                session=self.session,
                model_id=self.id,
                prompt_id=prompt_id,
                sequences=sequences,
            ),
        )

    def single_site(
        self, prompt: str | PromptFuture, sequence: bytes
    ) -> EmbeddingsScoreResultFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        sequence : bytes
            Sequence to analyse.
        Returns
        -------
        results
            The scores of the mutated sequence.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return EmbeddingsScoreResultFuture.create(
            session=self.session,
            job=embedding.request_score_single_site_post(
                session=self.session,
                model_id=self.id,
                base_sequence=sequence,
                prompt_id=prompt_id,
            ),
        )

    def generate(
        self,
        prompt: str | PromptFuture,
        num_samples: int = 100,
        temperature: float = 1.0,
        topk: float | None = None,
        topp: float | None = None,
        max_length: int = 1000,
        seed: int | None = None,
    ) -> EmbeddingsScoreResultFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
        num_samples : int, optional
            The number of samples to generate, by default 100.
        temperature : float, optional
            The temperature for sampling. Higher values produce more random outputs, by default 1.0.
        topk : int, optional
            The number of top-k residues to consider during sampling, by default None.
        topp : float, optional
            The cumulative probability threshold for top-p sampling, by default None.
        max_length : int, optional
            The maximum length of generated proteins, by default 1000.
        seed : int, optional
            Seed for random number generation, by default a random number.

        Raises
        ------
        APIError
            If there is an issue with the API request.

        Returns
        -------
        Job
            An object representing the status and information about the generation job.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return EmbeddingsScoreResultFuture.create(
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
            ),
        )

    def fit_svd(
        self,
        prompt: str | PromptFuture,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 1024,
        reduction: ReductionType | None = None,
    ) -> "SVDModel":
        """
        Fit an SVD on the embedding results of this model. 

        This function will create an SVDModel based on the embeddings from this model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
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
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().fit_svd(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt_id=prompt_id,
        )

    def fit_gp(
        self,
        prompt: str | PromptFuture,
        assay: AssayMetadata | AssayDataset | str,
        properties: list[str],
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
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().fit_gp(
            assay=assay, properties=properties, prompt_id=prompt_id, **kwargs
        )

    class Deprecated:

        def __init__(self, session: APISession):
            self.session = session

        def score(
            self,
            prompt: str | PromptFuture,
            sequences: list[bytes],
        ) -> PoetScoreFuture:
            """
            Score query sequences using the specified prompt.

            Parameters
            ----------
            prompt: Union[str, PromptFuture]
                prompt from an align workflow to condition Poet model
            sequence : bytes
                Sequence to analyse.
            Returns
            -------
                PoetScoreFuture
            """
            prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
            # HACK - manually construct the job and future since job types have been overwritten
            return PoetScoreFuture(
                session=self.session,
                job=PoetScoreJob(
                    **poet.poet_score_post(
                        session=self.session,
                        prompt_id=prompt_id,
                        queries=sequences,
                    ).model_dump()
                ),
            )

        def single_site(
            self, prompt: str | PromptFuture, sequence: bytes
        ) -> PoetSingleSiteFuture:
            """
            Score query sequences using the specified prompt.

            Parameters
            ----------
            prompt: Union[str, PromptFuture]
                prompt from an align workflow to condition Poet model
            sequence : bytes
                Sequence to analyse.
            Returns
            -------
                ScoreFuture
            """
            prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
            # HACK - manually construct the job and future since job types have been overwritten
            return PoetSingleSiteFuture(
                session=self.session,
                job=PoetSSPJob(
                    **poet.poet_single_site_post(
                        session=self.session,
                        prompt_id=prompt_id,
                        variant=sequence,
                    ).model_dump()
                ),
            )

        def generate(
            self,
            prompt: str | PromptFuture,
            num_samples: int = 100,
            temperature: float = 1.0,
            topk: float | None = None,
            topp: float | None = None,
            max_length: int = 1000,
            seed: int | None = None,
        ) -> PoetGenerateFuture:
            """
            (Deprecated) Generate protein sequences conditioned on a prompt.

            Parameters
            ----------
            prompt: Union[str, PromptFuture]
                prompt from an align workflow to condition Poet model
            num_samples : int, optional
                The number of samples to generate, by default 100.
            temperature : float, optional
                The temperature for sampling. Higher values produce more random outputs, by default 1.0.
            topk : int, optional
                The number of top-k residues to consider during sampling, by default None.
            topp : float, optional
                The cumulative probability threshold for top-p sampling, by default None.
            max_length : int, optional
                The maximum length of generated proteins, by default 1000.
            seed : int, optional
                Seed for random number generation, by default a random number.

            Raises
            ------
            APIError
                If there is an issue with the API request.

            Returns
            -------
            Job
                An object representing the status and information about the generation job.
            """
            prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
            # HACK - manually construct the job and future since job types have been overwritten
            return PoetGenerateFuture(
                session=self.session,
                job=PoetGenerateJob(
                    **poet.poet_generate_post(
                        session=self.session,
                        prompt_id=prompt_id,
                        num_samples=num_samples,
                        temperature=temperature,
                        topk=topk,
                        topp=topp,
                        max_length=max_length,
                        random_seed=seed,
                    ).model_dump()
                ),
            )
