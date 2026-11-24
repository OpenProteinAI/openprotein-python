import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openprotein.align import MSAFuture
from openprotein.jobs.futures import StructurePredictionFuture
from openprotein.models import AlphaFlow, BioEmu, ESMFlow, ModelsAPI

MSA_ID = "3f1c2a7e-9b4d-4e2a-8c1f-0d5e6a7b8c9d"


def _example(method) -> str:
    """The method's doctest lines as one script."""
    lines = (inspect.getdoc(method) or "").splitlines()
    return "\n".join(
        line.strip()[4:]
        for line in lines
        if line.strip().startswith((">>> ", "... ")) or line.strip() in (">>>", "...")
    )


@pytest.mark.parametrize(
    ("handle", "paths"),
    [
        (BioEmu, ["bioemu-v1-1"] * 2),
        (AlphaFlow, ["alphaflow-md-base-v1", "alphaflow-md-distilled-v1"]),
        (ESMFlow, ["esmflow-md-base-v1", "esmflow-md-distilled-v1"]),
    ],
)
def test_sample_examples_submit_valid_requests(
    models_session, ensemble_reference, monkeypatch, handle, paths
):
    monkeypatch.setattr(StructurePredictionFuture, "get", lambda self, **kwargs: [])
    msa = MagicMock(spec=MSAFuture)
    msa.id = MSA_ID
    sequence = ensemble_reference.sequence.decode()
    namespace = {
        "session": SimpleNamespace(models=ModelsAPI(models_session.session)),
        "msa": msa,
        "sequence": sequence,
        "sequences": [sequence, sequence[:40]],
        "reference": ensemble_reference,
    }
    exec(compile(_example(handle.sample), handle.__name__, "exec"), namespace)  # noqa: S102
    assert [path for path, _ in models_session.submissions] == [
        f"v1/models/{model_id}/batch/structures" for model_id in paths
    ]
