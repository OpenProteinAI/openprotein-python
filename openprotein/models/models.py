"""Models served through the models API."""

from functools import cached_property
from typing import Any, TypeVar

from openprotein import _deprecation
from openprotein.base import APISession
from openprotein.errors import HTTPError

from . import _api, estimators, gp, svd, umap
from .ensemble import AlphaFlow, BioEmu, ESMFlow
from .folding import (
    AlphaFold2,
    Boltz1,
    Boltz2,
    ESMFold,
    ESMFold2,
    MiniFold,
    OpenDDE,
    Protenix,
    ProtenixV2,
    RosettaFold3,
)
from .foundational import (
    ESM1,
    ESM2,
    ESMC,
    ESMIF1,
    AbLang2,
    PoET,
    PoET2,
    ProteinMPNN,
    ProtSeq,
    ProtT5,
    RotaProt,
)
from .foundational.proteinmpnn import _SolubleMPNN
from .generative import BoltzGen, RFdiffusion
from .gp import GPModel
from .model import Model
from .schemas import ModelMetadata
from .svd import SVDModel
from .umap import UMAPModel

__all__ = ["ModelsAPI"]


_H = TypeVar("_H", bound=Model)

_RENAMED = {
    "BoltzGenModel": BoltzGen,
    "ESMIF1Model": ESMIF1,
    "ProteinMPNNModel": ProteinMPNN,
    "RFdiffusionModel": RFdiffusion,
}


def __getattr__(name: str) -> Any:
    if name == "SolubleMPNNModel":
        _deprecation.warn_deprecated(
            "openprotein.models.models.SolubleMPNNModel",
            "`session.models.solublempnn`",
            since="0.18.0",
            removed_in="0.19.0",
        )
        return _SolubleMPNN
    if name not in _RENAMED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    new = _RENAMED[name]
    _deprecation.warn_deprecated(
        f"openprotein.models.models.{name}",
        f"`openprotein.models.{new.__name__}`",
        since="0.18.0",
        removed_in="0.19.0",
    )
    return new


class ModelsAPI:
    """Discover models and access their handles through ``session.models``.

    ``list_models`` lists available cards, and ``get_model`` loads a handle by
    model identifier. Named properties provide handles for common models and
    estimators for fitting SVD, UMAP and Gaussian process models.
    """

    def __init__(self, session: APISession):
        self.session = session
        self._handles: dict[str, Model] = {}

    def _handle(self, model_id: str, handle: type[_H]) -> _H:
        cached = self._handles.get(model_id)
        if not isinstance(cached, handle):
            cached = self._handles[model_id] = handle(self.session, model_id)
        return cached

    @cached_property
    def SVD(self) -> type[estimators.SVD]:
        """An SVD to fit, a fitted one by ``id``, or ``list`` the caller's."""
        return estimators.SVD._bind(self.session)

    @cached_property
    def UMAP(self) -> type[estimators.UMAP]:
        """A UMAP to fit, a fitted one by ``id``, or ``list`` the caller's."""
        return estimators.UMAP._bind(self.session)

    @cached_property
    def GP(self) -> type[estimators.GP]:
        """A GP to train, a trained one by ``id``, or ``list`` the caller's."""
        return estimators.GP._bind(self.session)

    @property
    def rfdiffusion(self) -> RFdiffusion:
        """``rfdiffusion``."""
        return self._handle("rfdiffusion", RFdiffusion)

    @property
    def proteinmpnn(self) -> ProteinMPNN:
        """``proteinmpnn``."""
        return self._handle("proteinmpnn", ProteinMPNN)

    @property
    def solublempnn(self) -> ProteinMPNN:
        """``solublempnn``."""
        return self._handle("solublempnn", ProteinMPNN)

    @property
    def boltzgen(self) -> BoltzGen:
        """``boltzgen``."""
        return self._handle("boltzgen", BoltzGen)

    @property
    def esmif1(self) -> ESMIF1:
        """``esm-if1``."""
        return self._handle("esm-if1", ESMIF1)

    @property
    def esm_if1(self) -> ESMIF1:
        """``esm-if1``."""
        return self.esmif1

    @property
    def poet2(self) -> PoET2:
        """``poet-2``."""
        return self._handle("poet-2", PoET2)

    @property
    def poet_2(self) -> PoET2:
        """``poet-2``."""
        return self.poet2

    @property
    def poet(self) -> PoET:
        """``poet``."""
        return self._handle("poet", PoET)

    @property
    def prot_seq(self) -> ProtSeq:
        """``prot-seq``."""
        return self._handle("prot-seq", ProtSeq)

    @property
    def rotaprot_large_uniref50w(self) -> RotaProt:
        """``rotaprot-large-uniref50w``."""
        return self._handle("rotaprot-large-uniref50w", RotaProt)

    @property
    def rotaprot_large_uniref90_ft(self) -> RotaProt:
        """``rotaprot-large-uniref90-ft``."""
        return self._handle("rotaprot-large-uniref90-ft", RotaProt)

    @property
    def ablang2(self) -> AbLang2:
        """``ablang2``."""
        return self._handle("ablang2", AbLang2)

    @property
    def esm1b_t33_650M_UR50S(self) -> ESM1:
        """``esm1b_t33_650M_UR50S``."""
        return self._handle("esm1b_t33_650M_UR50S", ESM1)

    @property
    def esm1b(self) -> ESM1:
        """``esm1b_t33_650M_UR50S``."""
        return self.esm1b_t33_650M_UR50S

    @property
    def esm1v_t33_650M_UR90S_1(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_1``."""
        return self._handle("esm1v_t33_650M_UR90S_1", ESM1)

    @property
    def esm1v(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_1``."""
        return self.esm1v_t33_650M_UR90S_1

    @property
    def esm1v_t33_650M_UR90S_2(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_2``."""
        return self._handle("esm1v_t33_650M_UR90S_2", ESM1)

    @property
    def esm1v_t33_650M_UR90S_3(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_3``."""
        return self._handle("esm1v_t33_650M_UR90S_3", ESM1)

    @property
    def esm1v_t33_650M_UR90S_4(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_4``."""
        return self._handle("esm1v_t33_650M_UR90S_4", ESM1)

    @property
    def esm1v_t33_650M_UR90S_5(self) -> ESM1:
        """``esm1v_t33_650M_UR90S_5``."""
        return self._handle("esm1v_t33_650M_UR90S_5", ESM1)

    @property
    def esm2_t12_35M_UR50D(self) -> ESM2:
        """``esm2_t12_35M_UR50D``."""
        return self._handle("esm2_t12_35M_UR50D", ESM2)

    @property
    def esm2_t30_150M_UR50D(self) -> ESM2:
        """``esm2_t30_150M_UR50D``."""
        return self._handle("esm2_t30_150M_UR50D", ESM2)

    @property
    def esm2_t33_650M_UR50D(self) -> ESM2:
        """``esm2_t33_650M_UR50D``."""
        return self._handle("esm2_t33_650M_UR50D", ESM2)

    @property
    def esm2(self) -> ESM2:
        """``esm2_t33_650M_UR50D``."""
        return self.esm2_t33_650M_UR50D

    @property
    def esm2_t36_3B_UR50D(self) -> ESM2:
        """``esm2_t36_3B_UR50D``."""
        return self._handle("esm2_t36_3B_UR50D", ESM2)

    @property
    def esm2_t6_8M_UR50D(self) -> ESM2:
        """``esm2_t6_8M_UR50D``."""
        return self._handle("esm2_t6_8M_UR50D", ESM2)

    @property
    def esmc_300m(self) -> ESMC:
        """``esmc-300m``."""
        return self._handle("esmc-300m", ESMC)

    @property
    def esmc_600m(self) -> ESMC:
        """``esmc-600m``."""
        return self._handle("esmc-600m", ESMC)

    @property
    def esmc_6b(self) -> ESMC:
        """``esmc-6b``."""
        return self._handle("esmc-6b", ESMC)

    @property
    def prott5_xl(self) -> ProtT5:
        """``prott5-xl``."""
        return self._handle("prott5-xl", ProtT5)

    @property
    def alphafold2(self) -> AlphaFold2:
        """``alphafold2``."""
        return self._handle("alphafold2", AlphaFold2)

    @property
    def af2(self) -> AlphaFold2:
        """``alphafold2``."""
        return self.alphafold2

    @property
    def boltz_1(self) -> Boltz1:
        """``boltz-1``."""
        return self._handle("boltz-1", Boltz1)

    @property
    def boltz1(self) -> Boltz1:
        """``boltz-1``."""
        return self.boltz_1

    @property
    def boltz_1x(self) -> Boltz1:
        """``boltz-1x``."""
        return self._handle("boltz-1x", Boltz1)

    @property
    def boltz1x(self) -> Boltz1:
        """``boltz-1x``."""
        return self.boltz_1x

    @property
    def boltz_2(self) -> Boltz2:
        """``boltz-2``."""
        return self._handle("boltz-2", Boltz2)

    @property
    def boltz2(self) -> Boltz2:
        """``boltz-2``."""
        return self.boltz_2

    @property
    def esmfold(self) -> ESMFold:
        """``esmfold``."""
        return self._handle("esmfold", ESMFold)

    @property
    def esmfold2(self) -> ESMFold2:
        """``esmfold2``."""
        return self._handle("esmfold2", ESMFold2)

    @property
    def esmfold2_fast(self) -> ESMFold2:
        """``esmfold2-fast``."""
        return self._handle("esmfold2-fast", ESMFold2)

    @property
    def minifold(self) -> MiniFold:
        """``minifold``."""
        return self._handle("minifold", MiniFold)

    @property
    def opendde(self) -> OpenDDE:
        """``opendde``."""
        return self._handle("opendde", OpenDDE)

    @property
    def opendde_abag(self) -> OpenDDE:
        """``opendde-abag``."""
        return self._handle("opendde-abag", OpenDDE)

    @property
    def protenix(self) -> Protenix:
        """``protenix``."""
        return self._handle("protenix", Protenix)

    @property
    def protenix_v2(self) -> ProtenixV2:
        """``protenix-v2``."""
        return self._handle("protenix-v2", ProtenixV2)

    @property
    def rosettafold_3(self) -> RosettaFold3:
        """``rosettafold-3``."""
        return self._handle("rosettafold-3", RosettaFold3)

    @property
    def rf3(self) -> RosettaFold3:
        """``rosettafold-3``."""
        return self.rosettafold_3

    @property
    def bioemu_v1_1(self) -> BioEmu:
        """``bioemu-v1-1``."""
        return self._handle("bioemu-v1-1", BioEmu)

    @property
    def bioemu(self) -> BioEmu:
        """``bioemu-v1-1``."""
        return self.bioemu_v1_1

    @property
    def alphaflow_md_base_v1(self) -> AlphaFlow:
        """``alphaflow-md-base-v1``."""
        return self._handle("alphaflow-md-base-v1", AlphaFlow)

    @property
    def alphaflow(self) -> AlphaFlow:
        """``alphaflow-md-base-v1``."""
        return self.alphaflow_md_base_v1

    @property
    def alphaflow_md_distilled_v1(self) -> AlphaFlow:
        """``alphaflow-md-distilled-v1``."""
        return self._handle("alphaflow-md-distilled-v1", AlphaFlow)

    @property
    def esmflow_md_base_v1(self) -> ESMFlow:
        """``esmflow-md-base-v1``."""
        return self._handle("esmflow-md-base-v1", ESMFlow)

    @property
    def esmflow(self) -> ESMFlow:
        """``esmflow-md-base-v1``."""
        return self.esmflow_md_base_v1

    @property
    def esmflow_md_distilled_v1(self) -> ESMFlow:
        """``esmflow-md-distilled-v1``."""
        return self._handle("esmflow-md-distilled-v1", ESMFlow)

    def list_models(
        self,
        verbose: bool = False,
        *,
        path: list[str] | None = None,
    ) -> list[str] | list[ModelMetadata]:
        """List the system models served by the models API.

        Parameters
        ----------
        verbose : bool, optional
            Return each model's metadata rather than its id.
        path : list of str, optional
            Keep models with a method whose path contains one of these values,
            ignoring case. Method paths are listed in each model's
            ``metadata.methods``, e.g. ``batch/embeddings`` or
            ``generate/sequences``, so ``["embeddings"]`` keeps embedding
            models and ``["generate"]`` keeps models that generate sequences
            or structures.

        Returns
        -------
        list of str or list of ModelMetadata
            Model ids, or their metadata with ``verbose``.
        """
        return _api.list_models(self.session, verbose=verbose, path=path)

    def get_model(self, model_id: str) -> Model:
        """Get a handle on a model by id.

        A model with an accessor, named as its id with ``-`` as ``_``, gets the
        accessor's handle. Other ids get their family's handle by prefix, or a
        plain ``Model``.

        Raises
        ------
        ValueError
            If no system model has the id. A fitted model's id is checked when
            the handle is first used, since its fit may still be running.
        """
        if model_id.startswith(f"{umap._FAMILY}/"):
            return UMAPModel(self.session, model_id)
        if model_id.startswith(f"{gp._FAMILY}/"):
            return GPModel(self.session, model_id)
        if model_id.startswith(f"{svd._FAMILY}/"):
            return SVDModel(self.session, model_id)
        handle = self._accessor(model_id) or Model(self.session, model_id)
        try:
            _ = handle.metadata
        except HTTPError as error:
            if error.status_code != 404:
                raise
            raise ValueError(
                f"model {model_id!r} not found; list_models() lists the "
                "available models"
            ) from error
        return handle

    def _accessor(self, model_id: str) -> Model | None:
        name = model_id.replace("-", "_")
        if not isinstance(getattr(type(self), name, None), property):
            return None
        handle = getattr(self, name)
        return handle if isinstance(handle, Model) and handle.id == model_id else None
