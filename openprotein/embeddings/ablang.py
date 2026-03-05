"""AbLang model."""

from .models import EmbeddingModel


class AbLang2Model(EmbeddingModel):
    """
    Community AbLang2 model that targets antibodies.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: python

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.embedding.ablang2?
    """

    model_id = ["ablang2"]
