"""The base handle of the structure-ensemble models."""

from typing import Any, cast

from openprotein.jobs import futures

from ..model import Model

__all__ = ["StructureEnsemblePredictionModel"]


class StructureEnsemblePredictionModel(Model):
    """A model that samples conformational ensembles.

    Each model's ``sample`` takes its own input type, available on the handle
    as ``Input``, one input per ensemble. It returns a
    ``StructurePredictionFuture`` whose ``get()`` gives one ``Structure`` per
    input, holding its sampled conformations.
    """

    def _sample(self, **params: Any) -> futures.StructurePredictionFuture:
        return cast(
            futures.StructurePredictionFuture,
            self._run("sample", "batch/structures", **params),
        )
