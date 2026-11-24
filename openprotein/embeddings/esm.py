"""Community-based ESM models."""

from . import api
from .future import EmbeddingsScoreSingleSiteFuture
from .models import AttnModel, EmbeddingModel

__all__ = ["ESMModel"]


class ESMModel(AttnModel, EmbeddingModel):
    """
    Class providing inference endpoints for Facebook's ESM protein language models.

    Covers ESM-1b, ESM-1v, ESM2 and ESM Cambrian (ESMC), and supports single-site
    scanning with `single_site`.

    Examples
    --------
    View specific model details (inc supported tokens) with the `?` operator.

    .. code-block:: ipython3

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.embedding.esm1b_t33_650M_UR50S?
    """

    model_id = [
        "esm1b_t33_650M_UR50S",
        "esm1v_t33_650M_UR90S_1",
        "esm1v_t33_650M_UR90S_2",
        "esm1v_t33_650M_UR90S_3",
        "esm1v_t33_650M_UR90S_4",
        "esm1v_t33_650M_UR90S_5",
        "esm2_t12_35M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t6_8M_UR50D",
        "esmc-300m",
        "esmc-600m",
        "esmc-6b",
    ]

    def single_site(self, sequence: bytes | str) -> EmbeddingsScoreSingleSiteFuture:
        """
        Score all single substitutions of a sequence.

        Masks each position in turn and scores every substitution there
        against the wild-type residue. The wild type therefore scores 0, and a
        positive score means the model prefers the substitution.

        Parameters
        ----------
        sequence : bytes or str
            Sequence to analyze.

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
