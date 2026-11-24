from unittest.mock import patch

import openprotein.fold as fold_package
from openprotein.fold.esmfold import ESMFoldModel
from openprotein.fold.fold import FoldAPI
from openprotein.fold.models import FoldModel, _subclasses


def test_fold_package_classes_predict_rather_than_sample():
    exported = [getattr(fold_package, name) for name in fold_package.__all__]
    assert not any(
        isinstance(item, type) and hasattr(item, "sample") for item in exported
    )


def test_every_fold_model_class_folds():
    classes = [FoldModel, *_subclasses(FoldModel)]
    assert all(callable(getattr(cls, "fold", None)) for cls in classes)
    assert all(
        issubclass(item, FoldModel)
        for name in fold_package.__all__
        if isinstance(item := getattr(fold_package, name), type)
        and name.endswith("Model")
    )


def test_a_listed_id_without_its_own_class_gets_the_generic_fold_model(
    mock_session,
):
    listed = ["esmfold", "example-fold-v1"]
    with patch("openprotein.fold.fold.api.fold_models_list_get", return_value=listed):
        fold = FoldAPI(mock_session)
        models = fold.list_models()
    assert type(vars(fold)["example_fold_v1"]) is FoldModel
    assert type(fold.get_model("example-fold-v1")) is FoldModel
    assert [model.id for model in models] == listed
    assert [type(model) for model in models] == [ESMFoldModel, FoldModel]
