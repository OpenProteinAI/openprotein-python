"""The ModelsAPI class, providing access to all protein models."""

from openprotein.base import APISession

from .foundation.rfdiffusion import RFdiffusionModel

# In the future, we would import other models here:
# from .foundation.esm import ESMModel
# from .foundation.alphafold import AlphaFoldModel
# from .custom.gp import GaussianProcessModel


class ModelsAPI:
    """
    API-like accessor that groups all available protein models.

    This class is attached to the main APISession and provides a single,
    consistent entry point for accessing various models.
    """

    def __init__(self, session: APISession):
        """
        Initializes the ModelsAPI and attaches instances of all available models.

        Args:
            session: The active APISession to be used by the models for API calls.
        """
        self.rfdiffusion = RFdiffusionModel(session)

        # To add new models, you would simply instantiate them here:
        # self.esm = ESMModel(session)
        # self.alphafold = AlphaFoldModel(session)
        # self.gp = GaussianProcessModel(session)
