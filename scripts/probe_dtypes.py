"""One-off probe: capture the wire dtype of model outputs from the live backend,
for the models the differential tests cover. Not a committed test — used to record
the ground-truth dtype matrix. Run: `uv run python scripts/probe_dtypes.py`.
"""

from __future__ import annotations

import numpy as np
from openprotein import connect
from openprotein.common.reduction import ReductionType
from tests.e2e.correctness import well_known
from tests.e2e.correctness.support import fresh_prompt, require_embedding_model

TIMEOUT = 600
ENCODERS = ["esm2_t33_650M_UR50D", "esmc-300m"]
POETS = ["poet", "poet-2"]


def _report(model_id: str, output: str, arr) -> None:
    a = np.asarray(arr)
    print(f"  {model_id:28s} {output:18s} dtype={a.dtype.name:10s} shape={a.shape}")


def _err(model_id: str, output: str, exc: BaseException) -> None:
    print(f"  {model_id:28s} {output:18s} ERROR: {type(exc).__name__}: {exc}")


def _model(session, model_id: str):
    try:
        return require_embedding_model(session, model_id)
    except BaseException as exc:  # noqa: BLE001 - pytest.skip raises BaseException
        print(f"  {model_id:28s} unavailable: {type(exc).__name__}: {exc}")
        return None


def probe_encoder(session, model_id: str) -> None:
    model = _model(session, model_id)
    if model is None:
        return
    seq = well_known.UBIQUITIN.encode()
    for label, reduction in (
        ("embed[residue]", None),
        ("embed[MEAN]", ReductionType.MEAN),
    ):
        try:
            ((_, arr),) = model.embed(sequences=[seq], reduction=reduction).wait(
                timeout=TIMEOUT
            )
            _report(model_id, label, arr)
        except BaseException as exc:  # noqa: BLE001
            _err(model_id, label, exc)
    try:
        ((_, arr),) = model.logits(sequences=[seq]).wait(timeout=TIMEOUT)
        _report(model_id, "logits", arr)
    except BaseException as exc:  # noqa: BLE001
        _err(model_id, "logits", exc)
    try:
        ((_, arr),) = model.attn(sequences=[seq]).wait(timeout=TIMEOUT)
        _report(model_id, "attn", arr)
    except BaseException as exc:  # noqa: BLE001
        _err(model_id, "attn", exc)


def probe_poet(session, model_id: str) -> None:
    model = _model(session, model_id)
    if model is None:
        return
    query = well_known.POET_QUERY.encode()
    try:
        prompt = fresh_prompt(session, well_known.POET_CONTEXT, timeout=TIMEOUT)
    except BaseException as exc:  # noqa: BLE001
        _err(model_id, "prompt", exc)
        return
    for label, reduction in (
        ("embed[residue]", None),
        ("embed[MEAN]", ReductionType.MEAN),
    ):
        try:
            arr = model.embed(
                sequences=[query], prompt=prompt, reduction=reduction
            ).wait(timeout=TIMEOUT)[0][1]
            _report(model_id, label, arr)
        except BaseException as exc:  # noqa: BLE001
            _err(model_id, label, exc)
    try:
        arr = model.logits(sequences=[query], prompt=prompt).wait(timeout=TIMEOUT)[0][1]
        _report(model_id, "logits", arr)
    except BaseException as exc:  # noqa: BLE001
        _err(model_id, "logits", exc)
    try:
        rows = model.score(sequences=[query], prompt=prompt).wait(timeout=TIMEOUT)
        _report(model_id, "score", np.asarray(rows[0].score))
    except BaseException as exc:  # noqa: BLE001
        _err(model_id, "score", exc)


def main() -> None:
    session = connect()
    print(f"backend: {session.backend}")
    print("== encoders ==")
    for m in ENCODERS:
        probe_encoder(session, m)
    print("== poet ==")
    for m in POETS:
        probe_poet(session, m)


if __name__ == "__main__":
    main()
