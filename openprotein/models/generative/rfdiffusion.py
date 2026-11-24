"""RFdiffusion."""

from .base import StructureGenerativeModel

__all__ = ["RFdiffusion"]


class RFdiffusion(StructureGenerativeModel):
    """RFdiffusion, for de novo backbones, scaffolding and binder design.

    ``Protein.from_expr(n)`` as the query designs ``n`` residues de novo. The
    query has a fixed length. Its checkpoint is ``rfdiffusion``.
    """

    _default_id = "rfdiffusion"
