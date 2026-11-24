import pytest

from openprotein.models import (
    BoltzGen,
    Model,
    PoET,
    PoET2,
    ProteinLanguageModel,
    RFdiffusion,
    StructureEnsemblePredictionModel,
    StructureGenerativeModel,
    StructurePredictionModel,
    ensemble,
    folding,
    foundational,
    generative,
)
from openprotein.models.foundational._inverse_folding import _InverseFoldingModel

BASES = [
    (foundational, (ProteinLanguageModel, _InverseFoldingModel)),
    (folding, (StructurePredictionModel,)),
    (ensemble, (StructureEnsemblePredictionModel,)),
    (generative, (StructureGenerativeModel,)),
]


@pytest.mark.parametrize(("package", "bases"), BASES)
def test_each_model_class_extends_its_packages_base(package, bases):
    classes = [
        value
        for value in map(package.__dict__.get, package.__all__)
        if isinstance(value, type) and issubclass(value, Model)
    ]
    assert classes
    for cls in classes:
        assert issubclass(cls, bases), cls


def test_the_model_bases_are_siblings():
    bases = [
        ProteinLanguageModel,
        StructurePredictionModel,
        StructureEnsemblePredictionModel,
        StructureGenerativeModel,
    ]
    for base in bases:
        assert base.__bases__ == (Model,)


def test_poet2_extends_poet():
    assert issubclass(PoET2, PoET)
    assert PoET2.indel is PoET.indel


def test_design_handles_share_query_submission():
    assert issubclass(RFdiffusion, StructureGenerativeModel)
    assert issubclass(BoltzGen, StructureGenerativeModel)
    assert RFdiffusion.generate is StructureGenerativeModel.generate
    assert BoltzGen.generate is StructureGenerativeModel.generate
