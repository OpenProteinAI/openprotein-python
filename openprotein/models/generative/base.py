"""Shared structure-generation query handling."""

from typing import Any, cast

from openprotein.jobs import futures

from .. import _inputs
from ..model import Model

__all__ = ["StructureGenerativeModel"]


class StructureGenerativeModel(Model):
    """A model that generates structures from design queries."""

    def generate(
        self, query: Any = None, **params: Any
    ) -> futures.StructureDesignFuture:
        """Generate structures from a query.

        Parameters
        ----------
        query : Protein, Complex, Query, str, or list, optional
            A design query or query id. Accepted contents depend on the model.
        **params
            Parameters declared by ``generate/structures``.

        Returns
        -------
        StructureDesignFuture
            The submitted structure-design job.
        """
        _inputs.check_inputs(params, query=query)
        self._method("generate", "generate/*")
        params = dict(params)
        many = _inputs.takes_list(
            self._params_schema("generate/structures"), "query_id"
        )
        _inputs.with_param(
            params,
            "query_id",
            _inputs.query_ids(self.session, query, force_structure=True, many=many),
            "query",
        )
        return cast(
            futures.StructureDesignFuture,
            super().generate(**params),
        )
