"""ESM-IF1 model providing inverse-folding, scoring, and generation capabilities."""

from typing import TYPE_CHECKING

from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.embeddings import api as embeddings_api
from openprotein.embeddings.future import (
    EmbeddingsGenerateFuture,
    EmbeddingsScoreFuture,
    EmbeddingsScoreSingleSiteFuture,
)
from openprotein.models.base import ProteinModel
from openprotein.molecules import Complex, Protein
from openprotein.prompt import PromptAPI, Query

if TYPE_CHECKING:
    from openprotein.models.structure_generation import StructureGenerationFuture


class ESMIF1Model(ProteinModel):
    """
    Class for ESM Inverse Folding (ESM-IF1) model.

    Model inference requires an input structure which is provided by a `query`.

    Examples
    --------
    View specific model details (including supported tokens) with the `?` operator.

    Examples
    --------
    .. code-block:: ipython3

        >>> import openprotein
        >>> session = openprotein.connect()
        >>> session.models.esmif1?
    """

    model_id = "esm-if1"

    def __init__(self, session: APISession):
        super().__init__(session=session, model_id=self.model_id)

    def get_metadata(self) -> ModelMetadata:
        return embeddings_api.get_model(session=self.session, model_id=self.model_id)

    def score(
        self,
        sequences: list[bytes] | list[str],
        query: str | bytes | Protein | Complex | Query,
    ) -> EmbeddingsScoreFuture:
        """
        Score sequences based on the specified query.

        Parameters
        ----------
        sequences : list of bytes or str
            Sequences to score.
        query : str or bytes or Protein or Complex or Query
            Query to use with prompt.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the submitted sequences.
        """
        prompt_api = getattr(self.session, "prompt", None)
        assert isinstance(prompt_api, PromptAPI)
        query_id = prompt_api._resolve_query(query=query)
        return EmbeddingsScoreFuture.create(
            session=self.session,
            job=embeddings_api.request_score_post(
                session=self.session,
                model_id=self.id,
                sequences=sequences,
                query_id=query_id,
            ),
            sequences=sequences,
        )

    def single_site(
        self,
        sequence: bytes | str,
        query: str | bytes | Protein | Complex | Query,
    ) -> EmbeddingsScoreSingleSiteFuture:
        """
        Score all single substitutions of `sequence` using the specified `query`.

        Parameters
        ----------
        sequence : bytes or str
            Sequence to analyze.
        query : str or bytes or Protein or Complex or Query
            Query to use with prompt.

        Returns
        -------
        EmbeddingsScoreSingleSiteFuture
            A future object that returns the per-variant scores.
        """
        prompt_api = getattr(self.session, "prompt", None)
        assert isinstance(prompt_api, PromptAPI)
        query_id = prompt_api._resolve_query(query=query)
        return EmbeddingsScoreSingleSiteFuture.create(
            session=self.session,
            job=embeddings_api.request_score_single_site_post(
                session=self.session,
                model_id=self.id,
                base_sequence=sequence,
                query_id=query_id,
            ),
        )

    def indel(
        self,
        sequence: bytes | str,
        query: str | bytes | Protein | Complex | Query,
        insert: str | None = None,
        delete: list[int] | None = None,
    ) -> EmbeddingsScoreFuture:
        """
        Score all indels of `sequence` based on the specified `query`.

        Parameters
        ----------
        sequence : bytes or str
            Sequence to analyze.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.
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
        raise NotImplementedError("Score indel not yet implemented")

    def generate(
        self,
        query: (
            str
            | bytes
            | Protein
            | Complex
            | Query
            | list[str | bytes | Protein | Complex | Query]
            | None
        ) = None,
        design: "str | StructureGenerationFuture | None" = None,
        num_samples: int = 100,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> EmbeddingsGenerateFuture:
        """
        Generate protein sequences based on a masked input query.

        Parameters
        ----------
        query : str or bytes or Protein or Complex or Query or list of these or None, optional
            Query specifying the structure to generate sequences for.
        design : str or StructureGenerationFuture or None, optional
            Structure-generation design ID or future to condition generation from
            design outputs.
        num_samples : int, optional
            Number of sequences to sample. Default 100.
        temperature : float, optional
            Sampling temperature. Lower = more conservative; near-zero
            approximates greedy decoding. Default 1.0.
        seed : int, optional
            PRNG seed for reproducible sampling.

        Returns
        -------
        EmbeddingsGenerateFuture
            A future object representing the status and information about the generation job.
        """
        if query is None and design is None:
            raise ValueError("Expected either `query` or `design` to be provided")

        from openprotein.models.structure_generation import StructureGenerationFuture

        query_id = None
        if query is not None:
            prompt_api = getattr(self.session, "prompt", None)
            assert isinstance(prompt_api, PromptAPI)
            query_id = prompt_api._resolve_query(query=query)
        design_id = (
            design.job_id if isinstance(design, StructureGenerationFuture) else design
        )
        return EmbeddingsGenerateFuture.create(
            session=self.session,
            job=embeddings_api.request_generate_post(
                session=self.session,
                model_id=self.id,
                query_id=query_id,
                design_id=design_id,
                num_samples=num_samples,
                temperature=temperature,
                random_seed=seed,
            ),
        )
