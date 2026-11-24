from typing_extensions import assert_type

from openprotein.jobs.futures import (
    StructureDesignFuture,
    StructurePredictionFuture,
)
from openprotein.models import (
    ESM2,
    AlphaFlow,
    AlphaFlowInput,
    BioEmu,
    BioEmuInput,
    ESMFlow,
    ESMFlowInput,
    ESMFold,
    ModelsAPI,
    StructureGenerativeModel,
)
from openprotein.molecules import Protein, Structure


def accessor_types(models: ModelsAPI) -> None:
    assert_type(models.esm2, ESM2)
    assert_type(models.esmfold, ESMFold)
    assert_type(models.bioemu_v1_1, BioEmu)
    assert_type(models.bioemu, BioEmu)
    assert_type(models.alphaflow, AlphaFlow)
    assert_type(models.esmflow, ESMFlow)
    assert_type(models.alphaflow_md_base_v1, AlphaFlow)
    assert_type(models.alphaflow_md_distilled_v1, AlphaFlow)
    assert_type(models.esmflow_md_base_v1, ESMFlow)
    assert_type(models.esmflow_md_distilled_v1, ESMFlow)


def input_types(bioemu: BioEmu, alphaflow: AlphaFlow, esmflow: ESMFlow) -> None:
    assert_type(bioemu.Input, type[BioEmuInput])
    assert_type(alphaflow.Input, type[AlphaFlowInput])
    assert_type(esmflow.Input, type[ESMFlowInput])


def prediction_types(esmflow: ESMFlow, reference: Protein) -> None:
    assert_type(
        esmflow.sample([ESMFlowInput("MKT", reference_structure=reference)]),
        StructurePredictionFuture,
    )
    assert_type(esmflow.sample([ESMFlowInput("MKT")]).get(), list[Structure])


def generative_types(model: StructureGenerativeModel) -> None:
    assert_type(model.generate(), StructureDesignFuture)
