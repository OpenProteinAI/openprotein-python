from typing import List, Sequence

from openprotein.api.prompt import (
    Context,
    create_prompt,
    create_query,
    get_prompt_metadata,
    get_query_metadata,
    list_prompts,
)
from openprotein.app.models import Prompt, Query
from openprotein.base import APISession
from openprotein.protein import Protein


class PromptAPI:
    """API interface for calling prompt endpoints"""

    def __init__(self, session: APISession):
        self.session = session

    def create_prompt(
        self,
        context: Context | Sequence[Context],
        name: str | None = None,
        description: str | None = None,
    ) -> Prompt:
        """
        Create a prompt.

        Parameters
        ----------
        context : Context | Sequence[Context]
            context or list of contexts, where each context is a Sequence of str,
            bytes, and/or Protein
        query : Optional[bytes | str | Protein]
            Optional query provided as sequence/structure
        name : str
            Name of the prompt.
        description : Optional[str]
            Description of the prompt.

        Returns
        -------
        PromptMetadata
            Metadata of the created prompt.
        """
        return Prompt(
            session=self.session,
            metadata=create_prompt(
                session=self.session,
                context=context,
                name=name,
                description=description,
            ),
        )

    def get_prompt(self, prompt_id: str) -> Prompt:
        """
        Get the prompt for a given prompt ID.

        Parameters
        ----------
        prompt_id : str
            The prompt ID.

        Returns
        -------
        BinaryIO
            The prompt data in binary format.
        """
        return Prompt(
            session=self.session,
            metadata=get_prompt_metadata(session=self.session, prompt_id=prompt_id),
        )

    def list_prompts(self) -> List[Prompt]:
        """
        List all prompts.

        Returns
        -------
        List[PromptMetadata]
            List of prompt metadata.
        """
        return [
            Prompt(session=self.session, metadata=p)
            for p in list_prompts(session=self.session)
        ]

    def create_query(
        self,
        query: str | bytes | Protein,
    ) -> Query:
        """
        Create a query.

        Parameters
        ----------
        query : Optional[bytes | str | Protein]
            Optional query provided as sequence/structure

        Returns
        -------
        QueryMetadata
            Metadata of the created query.
        """
        return Query(
            session=self.session,
            metadata=create_query(
                session=self.session,
                query=query,
            ),
        )

    def get_query(self, query_id: str) -> Query:
        """
        Get the query for a given query ID.

        Parameters
        ----------
        query_id : str
            The query ID.

        Returns
        -------
        BinaryIO
            The query data in binary format.
        """
        return Query(
            session=self.session,
            metadata=get_query_metadata(session=self.session, query_id=query_id),
        )
