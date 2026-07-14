"""SolubleMPNN model: ProteinMPNN variant tuned for soluble protein targets."""

from .proteinmpnn import ProteinMPNNModel


class SolubleMPNNModel(ProteinMPNNModel):
    """
    Class for SolubleMPNN model.

    Soluble-protein variant of ProteinMPNN.

    Examples
    --------
    View specific model details (including supported tokens) with the `?` operator.

    Examples
    --------
    .. code-block:: ipython3

        >>> import openprotein
        >>> session = openprotein.connect(username="user", password="password")
        >>> session.models.solublempnn?
    """

    model_id = "solublempnn"
