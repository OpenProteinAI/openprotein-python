import json

import pytest
from pydantic import TypeAdapter, ValidationError

from openprotein.codecs import json_utf8_v1
from openprotein.fold import FoldConfidence

NULL_SCORES = {
    "complex_iplddt": None,
    "complex_pde": None,
    "complex_ipde": None,
    "protein_iptm": None,
    "ligand_iptm": None,
    "chain_iptm": None,
    "has_clash": None,
}


def two_chain(**overrides):
    payload = {
        "chain_ids": ["A", "B"],
        "ranking_score": 0.81,
        "ptm": 0.72,
        "iptm": 0.64,
        "complex_plddt": 83.5,
        "chain_plddt": [85.0, 80.25],
        "chain_ptm": [0.8, None],
        "chain_pair_iptm": [[0.8, 0.6], [0.6, None]],
        "ipae": 7.5,
        **NULL_SCORES,
    }
    payload.update(overrides)
    return payload


def one_chain(**overrides):
    payload = {
        "chain_ids": ["A"],
        "ranking_score": None,
        "ptm": None,
        "iptm": None,
        "complex_plddt": 91.0,
        "chain_plddt": [91.0],
        "chain_ptm": [None],
        "chain_pair_iptm": [[None]],
        "ipae": None,
        **NULL_SCORES,
    }
    payload.update(overrides)
    return payload


def test_null_scores_validate():
    confidence = FoldConfidence.model_validate(two_chain())
    assert confidence.chain_ids == ["A", "B"]
    assert confidence.chain_pair_iptm == [[0.8, 0.6], [0.6, None]]
    assert confidence.complex_pde is None


def test_single_chain_with_null_scores_validates():
    assert FoldConfidence.model_validate(one_chain()).ipae is None


def test_every_score_validates():
    confidence = FoldConfidence.model_validate(
        two_chain(
            complex_iplddt=70.0,
            complex_pde=1.5,
            complex_ipde=4.0,
            protein_iptm=0.6,
            ligand_iptm=0.3,
            chain_iptm=[0.6, 0.6],
            has_clash=False,
        )
    )
    assert confidence.has_clash is False
    assert confidence.chain_iptm == [0.6, 0.6]


def test_list_from_json_utf8_v1_validates():
    data = json_utf8_v1.encode([two_chain(), two_chain(ranking_score=-99.2)])
    samples = [FoldConfidence.model_validate(s) for s in json_utf8_v1.decode(data)]
    assert [s.ranking_score for s in samples] == [0.81, -99.2]


SCORES = {
    "complex_iplddt": 70.0,
    "complex_pde": 1.5,
    "complex_ipde": 4.0,
    "protein_iptm": 0.6,
    "ligand_iptm": 0.3,
    "chain_iptm": [0.6, 0.6],
    "has_clash": False,
}


@pytest.mark.parametrize(
    "payload",
    [
        two_chain(),
        two_chain(complex_pde=1.5, has_clash=True),
        two_chain(**SCORES),
        one_chain(),
    ],
    ids=["null-scores", "some-scores", "every-score", "one-chain"],
)
def test_dump_re_validates_and_keeps_every_key(payload):
    confidence = FoldConfidence.model_validate(payload)
    assert confidence.model_dump() == payload
    assert json.loads(confidence.model_dump_json()) == payload
    assert FoldConfidence.model_validate(confidence.model_dump()) == confidence
    assert (
        FoldConfidence.model_validate_json(confidence.model_dump_json()) == confidence
    )


def test_dump_keeps_null_keys():
    dumped = FoldConfidence.model_validate(one_chain()).model_dump()
    for key in ("ranking_score", "ptm", "iptm", "ipae", *NULL_SCORES):
        assert dumped[key] is None


def test_list_dump_re_validates():
    samples = [
        FoldConfidence.model_validate(two_chain()),
        FoldConfidence.model_validate(two_chain(complex_pde=1.5)),
    ]
    adapter = TypeAdapter(list[FoldConfidence])
    for dumped in (
        adapter.dump_python(samples),
        json.loads(adapter.dump_json(samples)),
    ):
        assert dumped == [two_chain(), two_chain(complex_pde=1.5)]
        assert adapter.validate_python(dumped) == samples


@pytest.mark.parametrize("key", list(FoldConfidence.model_fields))
def test_every_key_is_required(key):
    payload = two_chain()
    del payload[key]
    with pytest.raises(ValidationError, match=key):
        FoldConfidence.model_validate(payload)


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError, match="chains_ptm"):
        FoldConfidence.model_validate(two_chain(chains_ptm={"0": 0.8}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"ptm": float("nan")},
        {"ranking_score": float("inf")},
        {"chain_plddt": [85.0, float("nan")]},
        {"chain_pair_iptm": [[0.8, float("nan")], [0.6, None]]},
        {"ipae": float("nan")},
    ],
)
def test_non_finite_numbers_are_rejected(overrides):
    with pytest.raises(ValidationError):
        FoldConfidence.model_validate(two_chain(**overrides))


def test_nan_in_json_is_rejected():
    with pytest.raises(ValidationError):
        FoldConfidence.model_validate_json(json.dumps(two_chain(ptm=float("nan"))))


@pytest.mark.parametrize(
    "overrides",
    [
        {"ptm": 1.2},
        {"iptm": -0.1},
        {"complex_plddt": -0.5},
        {"complex_plddt": 100.5},
        {"chain_plddt": [85.0, 101.0]},
        {"chain_ptm": [1.5, None]},
        {"chain_pair_iptm": [[0.8, 1.1], [0.6, None]]},
        {"ipae": -1.0},
        {"complex_iplddt": 120.0},
        {"complex_pde": -0.5},
        {"ligand_iptm": 2.0},
        {"chain_iptm": [0.5, -0.5]},
    ],
)
def test_out_of_scale_values_are_rejected(overrides):
    with pytest.raises(ValidationError):
        FoldConfidence.model_validate(two_chain(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"chain_plddt": [85.0]},
        {"chain_ptm": [0.8, None, 0.7]},
        {"chain_iptm": [0.6]},
        {"chain_pair_iptm": [[0.8, 0.6]]},
        {"chain_pair_iptm": [[0.8, 0.6], [0.6]]},
    ],
)
def test_chain_lists_must_match_chain_ids(overrides):
    with pytest.raises(ValidationError, match="chain_ids"):
        FoldConfidence.model_validate(two_chain(**overrides))


@pytest.mark.parametrize("chain_ids", [[], ["A", "A"]], ids=["empty", "duplicate"])
def test_chain_ids_are_non_empty_and_unique(chain_ids):
    with pytest.raises(ValidationError, match="chain_ids"):
        FoldConfidence.model_validate(two_chain(chain_ids=chain_ids))


@pytest.mark.parametrize("key, value", [("iptm", 0.5), ("ipae", 3.0)])
def test_interface_scores_are_null_for_one_chain(key, value):
    with pytest.raises(ValidationError, match=key):
        FoldConfidence.model_validate(one_chain(**{key: value}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"has_clash": 0},
        {"chain_ids": ["A", 2]},
        {"ptm": "0.5"},
    ],
)
def test_values_are_not_coerced(overrides):
    with pytest.raises(ValidationError):
        FoldConfidence.model_validate(two_chain(**overrides))
