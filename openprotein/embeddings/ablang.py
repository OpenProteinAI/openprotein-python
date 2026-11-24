"""AbLang model."""

from . import api
from .future import EmbeddingsScoreSingleSiteFuture
from .models import EmbeddingModel

__all__ = ["AbLang2Model"]


class AbLang2Model(EmbeddingModel):
    """
    Community AbLang2 model that targets antibodies.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: ipython3

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.embedding.ablang2?
    """

    model_id = ["ablang2"]

    def single_site(
        self,
        sequence: bytes | str,
    ) -> EmbeddingsScoreSingleSiteFuture:
        """
        Score all single substitutions of a paired antibody sequence.

        Masks each position of the sequence in turn and scores every
        substitution there against the wild-type residue, so the wild type
        scores 0 and a positive score means the model prefers the
        substitution. `sequence` is a paired VH:VL sequence; the ':'
        separator is kept and both chains are scanned. Gap ('-') positions
        are skipped and kept in every variant. `include_gaps` is not
        supported for this model.

        Parameters
        ----------
        sequence : bytes or str
            Paired VH:VL sequence to analyze.

        Returns
        -------
        EmbeddingsScoreSingleSiteFuture
            Future object that returns the per-variant scores of the single substitutions.
        """
        return EmbeddingsScoreSingleSiteFuture.create(
            session=self.session,
            job=api.request_score_single_site_post(
                session=self.session,
                model_id=self.id,
                base_sequence=sequence,
            ),
        )
