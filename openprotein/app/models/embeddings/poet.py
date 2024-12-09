import warnings
from typing import TYPE_CHECKING

from openprotein.api import embedding
from openprotein.api.deprecated import poet
from openprotein.base import APISession
from openprotein.schemas import ModelMetadata, ReductionType
from openprotein.schemas.deprecated.poet import (
    PoetGenerateJob,
    PoetScoreJob,
    PoetSSPJob,
)

from ..align import PromptFuture
from ..assaydata import AssayDataset, AssayMetadata
from .base import EmbeddingModel
from .future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)

if TYPE_CHECKING:
    from ..deprecated import PoetGenerateFuture, PoetScoreFuture, PoetSingleSiteFuture
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
    _deprecated: "Deprecated | None" = None

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
        prompt: str | PromptFuture,
        sequences: list[bytes],
        reduction: ReductionType | None = ReductionType.MEAN,
    ) -> EmbeddingsResultFuture:
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
            A future object that returns the embeddings of the submitted sequences.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().embed(
            sequences=sequences, reduction=reduction, prompt_id=prompt_id
        )

    def logits(
        self,
        prompt: str | PromptFuture,
        sequences: list[bytes],
    ) -> EmbeddingsResultFuture:
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
            A future object that returns the logits of the submitted sequences.
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
    ) -> EmbeddingsScoreFuture:
        """
        Score query sequences using the specified prompt.

        Parameters
        ----------
        prompt: str | PromptFuture
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        sequence: list[bytes]
            Sequences to score.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the submitted sequences.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return EmbeddingsScoreFuture.create(
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
    ) -> EmbeddingsScoreFuture:
        """
        Score all single substitutions of the query sequence using the specified prompt.

        Parameters
        ----------
        prompt: str | PromptFuture
            Prompt or prompt_id or prompt from an align workflow to condition Poet model
        sequence: bytes
            Sequence to analyse.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the mutated sequence.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return EmbeddingsScoreFuture.create(
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
    ) -> EmbeddingsScoreFuture:
        """
        Generate protein sequences conditioned on a prompt.

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
            prompt from an align workflow to condition Poet model
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
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
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
        Fit an SVD on the embedding results of PoET. 

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
            A future that represents the fitted SVD model.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().fit_svd(
            sequences=sequences,
            assay=assay,
            n_components=n_components,
            reduction=reduction,
            prompt_id=prompt_id,
        )

    def fit_umap(
        self,
        prompt: str | PromptFuture,
        sequences: list[bytes] | list[str] | None = None,
        assay: AssayDataset | None = None,
        n_components: int = 2,
        reduction: ReductionType | None = ReductionType.MEAN,
    ) -> "UMAPModel":
        """
        Fit a UMAP on assay using PoET and hyperparameters.

        This function will create a UMAP based on the embeddings from this PoET model \
            as well as the hyperparameters specified in the args.  

        Parameters
        ----------
        prompt: Union[str, PromptFuture]
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
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().fit_umap(
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
            A future that represents the trained predictor model.
        """
        prompt_id = prompt.id if isinstance(prompt, PromptFuture) else prompt
        return super().fit_gp(
            assay=assay, properties=properties, prompt_id=prompt_id, **kwargs
        )

    @property
    def deprecated(self):
        if self._deprecated is None:
            warnings.warn(
                "The old interface to PoET is deprecated! Support will be dropped in the future. Please migrate your code to use the new interface."
            )
            from ..deprecated import (
                PoetGenerateFuture,
                PoetScoreFuture,
                PoetSingleSiteFuture,
            )

            self._deprecated = self.Deprecated(session=self.session)
        return self._deprecated

    class Deprecated:

        def __init__(self, session: APISession):
            self.session = session

        def score(
            self,
            prompt: str | PromptFuture,
            sequences: list[bytes],
        ) -> "PoetScoreFuture":
            """
            (Deprecated) Score query sequences using the specified prompt.

            Parameters
            ----------
            prompt: Union[str, PromptFuture]
                Prompt or prompt_id of prompt from an align workflow to condition Poet model
            sequences : list[bytes]
                Sequences to score.
            Returns
            -------
            PoetScoreFuture
                A future object that returns the scores of the submitted sequences.
            """
            from ..deprecated import PoetScoreFuture

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
        ) -> "PoetSingleSiteFuture":
            """
            (Deprecated) Score query sequences using the specified prompt.

            Parameters
            ----------
            prompt: str | PromptFuture
                Prompt or prompt_id of prompt from an align workflow to condition Poet model
            sequence: bytes
                Sequence to analyse.
            Returns
            -------
            PoetSingleSiteFuture
                A future object that returns the scores of the mutated sequence.
            """
            from ..deprecated import PoetSingleSiteFuture

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
        ) -> "PoetGenerateFuture":
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

            Returns
            -------
            PoetGenerateFuture
                A future object representing the status and information about the generation job.
            """
            from ..deprecated import PoetGenerateFuture

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
