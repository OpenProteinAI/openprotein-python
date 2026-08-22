"""Prompt API providing the interface to create prompts for use with PoET models."""

from typing import List, Sequence

from openprotein.base import APISession
from openprotein.molecules import Protein, Complex
from openprotein.utils import uuid

from . import api
from .models import Prompt, Query
from .schemas import Context

HUMAN_ANTIBODY_VH_V1 = "c044b1b9-9842-536a-83e0-e81400199f67"
HUMAN_ANTIBODY_VH_VL_V1 = "0ddde5dc-d140-5d53-bebf-e69fff5a05c5"
HUMAN_ANTIBODY_VL_V1 = "3da24801-7032-5b79-a0b9-6e687581de60"
HUMAN_ANTIBODY_VL_VH_V1 = "6de556eb-a739-5986-ab20-cf574a780f69"


class PromptAPI:
    """Prompt API providing the interface to create prompts for use with PoET models."""

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
            Context or list of contexts. Each context is a sequence of entries
            where each entry is a raw sequence (``bytes``/``str``, optionally with
            ``:`` chain breaks for multichain), :py:class:`Protein`, or
            :py:class:`Complex`. Currently only protein chains are accepted;
            passing a Complex with DNA, RNA, or Ligand chains raises
            :py:class:`InvalidParameterError`. This restriction may be relaxed
            in the future.
        name : str
            Name of the prompt.
        description : Optional[str]
            Description of the prompt.

        Returns
        -------
        Prompt
            The created prompt.
        """
        return Prompt(
            session=self.session,
            metadata=api.create_prompt(
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
        Prompt
            The prompt.
        """
        return Prompt(
            session=self.session,
            metadata=api.get_prompt_metadata(session=self.session, prompt_id=prompt_id),
        )

    def list_prompts(self) -> List[Prompt]:
        """
        List all prompts.

        Returns
        -------
        List[Prompt]
            List of prompts.
        """
        return [
            Prompt(session=self.session, metadata=p)
            for p in api.list_prompts(session=self.session)
        ]

    def create_query(
        self,
        query: str | bytes | Protein | Complex,
        force_structure: bool = False,
    ) -> Query:
        """
        Create a query.

        Parameters
        ----------
        query : bytes or str or Protein or Complex
            A query protein or complex. Raw ``bytes``/``str`` inputs may include
            ``:`` chain breaks to denote a multichain protein. Currently only
            protein chains are accepted; passing a Complex with DNA, RNA, or
            Ligand chains raises :py:class:`InvalidParameterError`. This
            restriction may be relaxed in the future.
        force_structure : bool, optional
            Optionally force a query to be interpreted with a structure.
            Useful for creating structure prediction queries which can have
            no structure.

        Returns
        -------
        Query
            The created query.
        """
        return Query(
            session=self.session,
            metadata=api.create_query(
                session=self.session,
                query=query,
                force_structure=force_structure,
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
        Query
            The query.
        """
        return Query(
            session=self.session,
            metadata=api.get_query_metadata(session=self.session, query_id=query_id),
        )

    @property
    def human_antibody_vh(self) -> Prompt:
        """
        Curated prompt of naive human antibody heavy chains (VH).

        Tracks the latest version of this prompt, currently
        :py:attr:`human_antibody_vh_v1`. Use the versioned attribute directly to
        pin a version that will not change in a later release.
        """
        return self.human_antibody_vh_v1

    @property
    def human_antibody_vh_v1(self) -> Prompt:
        """
        Curated prompt of naive human antibody heavy chains (VH), version 1.

        An ensemble of 10 replicates, each a random sample of 200 naive human
        antibody heavy chain (VH) sequences from the OAS paired database,
        clustered at 70% sequence identity.
        """
        return self.get_prompt(HUMAN_ANTIBODY_VH_V1)

    @property
    def human_antibody_vl(self) -> Prompt:
        """
        Curated prompt of naive human antibody light chains (VL).

        Tracks the latest version of this prompt, currently
        :py:attr:`human_antibody_vl_v1`. Use the versioned attribute directly to
        pin a version that will not change in a later release.
        """
        return self.human_antibody_vl_v1

    @property
    def human_antibody_vl_v1(self) -> Prompt:
        """
        Curated prompt of naive human antibody light chains (VL), version 1.

        An ensemble of 10 replicates, each a random sample of 200 naive human
        antibody light chain (VL) sequences from the OAS paired database,
        clustered at 70% sequence identity.
        """
        return self.get_prompt(HUMAN_ANTIBODY_VL_V1)

    @property
    def human_antibody_vh_vl(self) -> Prompt:
        """
        Curated prompt of naive paired human antibodies, heavy chain first.

        Tracks the latest version of this prompt, currently
        :py:attr:`human_antibody_vh_vl_v1`. Use the versioned attribute directly
        to pin a version that will not change in a later release.
        """
        return self.human_antibody_vh_vl_v1

    @property
    def human_antibody_vh_vl_v1(self) -> Prompt:
        """
        Curated prompt of naive paired human antibodies (VH-VL), version 1.

        An ensemble of 10 replicates, each a random sample of 200 naive paired
        human antibody sequences from the OAS paired database, clustered at 70%
        sequence identity. Each complex is the heavy chain (VH) followed by the
        light chain (VL).
        """
        return self.get_prompt(HUMAN_ANTIBODY_VH_VL_V1)

    @property
    def human_antibody_vl_vh(self) -> Prompt:
        """
        Curated prompt of naive paired human antibodies, light chain first.

        Tracks the latest version of this prompt, currently
        :py:attr:`human_antibody_vl_vh_v1`. Use the versioned attribute directly
        to pin a version that will not change in a later release.
        """
        return self.human_antibody_vl_vh_v1

    @property
    def human_antibody_vl_vh_v1(self) -> Prompt:
        """
        Curated prompt of naive paired human antibodies (VL-VH), version 1.

        An ensemble of 10 replicates, each a random sample of 200 naive paired
        human antibody sequences from the OAS paired database, clustered at 70%
        sequence identity. Each complex is the light chain (VL) followed by the
        heavy chain (VH).
        """
        return self.get_prompt(HUMAN_ANTIBODY_VL_VH_V1)

    def _resolve_query(
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
        force_structure: bool = False,
    ) -> str | list[str] | None:
        if query is None:
            query_id = None
        elif isinstance(query, list):
            query_id = [
                self._resolve_query(query=q, force_structure=force_structure)
                for q in query
            ]
        elif (
            isinstance(query, Protein)
            or isinstance(query, Complex)
            or isinstance(query, bytes)
            or (isinstance(query, str) and not uuid.is_valid_uuid(query))
        ):
            query_ = self.create_query(query=query, force_structure=force_structure)
            query_id = query_.id
        else:
            query_id = query if isinstance(query, str) else query.id
        return query_id
