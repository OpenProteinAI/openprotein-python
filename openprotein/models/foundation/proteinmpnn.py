"""ProteinMPNN model providing inverse-folding and scoring capabilities."""

from typing import TYPE_CHECKING

from openprotein import embeddings
from openprotein.base import APISession
from openprotein.common import ModelMetadata
from openprotein.embeddings import api as embeddings_api
from openprotein.embeddings.future import (
    EmbeddingsGenerateFuture,
    EmbeddingsResultFuture,
    EmbeddingsScoreFuture,
)
from openprotein.models.base import ProteinModel
from openprotein.prompt import PromptAPI, Query
from openprotein.protein import Protein
from openprotein.utils import uuid


class ProteinMPNNModel(ProteinModel):
    """
    Class for ProteinMPNN model.

    Model inference requires an input structure which is provided by a `query`.

    Examples
    --------
    View specific model details (including supported tokens) with the `?` operator.

    Examples
    --------
    .. code-block:: python

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.models.proteinmpnn?
    """

    model_id = "proteinmpnn"

    # TODO - Add model to explicitly require prompt_id
    def __init__(
        self,
        session: APISession,
    ):
        super().__init__(session=session, model_id=self.model_id)

    def get_metadata(self) -> ModelMetadata:
        return embeddings_api.get_model(session=self.session, model_id=self.model_id)

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

    def score(
        self,
        sequences: list[bytes],
        query: str | bytes | Protein | Query,
    ) -> EmbeddingsScoreFuture:
        """
        Score query sequences based on the specified query.

        Parameters
        ----------
        sequences : list of bytes
            Sequences to score.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the submitted sequences.
        """
        raise NotImplementedError("Score not yet implemented")

    def indel(
        self,
        sequence: bytes,
        query: str | bytes | Protein | Query,
        insert: str | None = None,
        delete: list[int] | None = None,
        **kwargs,
    ) -> EmbeddingsScoreFuture:
        """
        Score all indels of the query sequence based on the specified query.

        Parameters
        ----------
        sequence : bytes
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

    def single_site(
        self,
        sequence: bytes,
        query: str | bytes | Protein | Query,
    ) -> EmbeddingsScoreFuture:
        """
        Score all single substitutions of the query sequence using the specified query.

        Parameters
        ----------
        sequence : bytes
            Sequence to analyze.
        query : str or bytes or Protein or Query or None, optional
            Query to use with prompt.

        Returns
        -------
        EmbeddingsScoreFuture
            A future object that returns the scores of the mutated sequence.
        """
        raise NotImplementedError("Score indel not yet implemented")

    def generate(
        self,
        query: str | bytes | Protein | Query,
        num_samples: int = 100,
        temperature: float = 1.0,
        # topk: float | None = None,
        # topp: float | None = None,
        # max_length: int = 1000,
        # seed: int | None = None,
    ) -> EmbeddingsGenerateFuture:
        """
        Generate protein sequences based on a masked input query.

        Parameters
        ----------
        query : str or bytes or Protein or Query
            Query specifying the structure to generate sequences for.
        num_samples : int, optional
            The number of samples to generate. Default is 100.
        temperature : float, optional
            The temperature for sampling. Higher values produce more random outputs. Default is 1.0.

        Returns
        -------
        EmbeddingsGenerateFuture
            A future object representing the status and information about the generation job.
        """
        query_id = self.__resolve_query(query=query)
        return EmbeddingsGenerateFuture.create(
            session=self.session,
            job=embeddings_api.request_generate_post(
                session=self.session,
                model_id=self.id,
                query_id=query_id,
                num_samples=num_samples,
                temperature=temperature,
            ),
        )
