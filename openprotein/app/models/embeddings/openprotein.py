"""OpenProtein-proprietary models."""

from .base import EmbeddingModel


class OpenProteinModel(EmbeddingModel):
    """
    Class providing inference endpoints for proprietary protein embedding models served by OpenProtein.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        import openprotein
        session = openprotein.connect(username="user", password="password")
        session.embedding.prot_seq?
    """

    model_id = ["prot-seq", "rotaprot-large-uniref50w", "rotaprot_large_uniref90_ft"]
