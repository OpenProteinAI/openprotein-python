"""Test the schemas for the design domain."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from openprotein.design.schemas import (
    Criteria,
    Criterion,
    DesignConstraint,
    ModelCriterion,
    NMutationCriterion,
    n_mutations,
)


def test_model_criterion_operators():
    """Test the overloaded operators for ModelCriterion."""
    mc = ModelCriterion(model_id="m1", measurement_name="p1")

    # Test weighting
    mc = mc * 2.0
    assert mc.criterion.weight == 2.0
    mc = 3.0 * mc
    assert mc.criterion.weight == 3.0

    # Test comparisons
    mc_lt = mc < 5.0
    assert mc_lt.criterion.target == 5.0
    assert mc_lt.criterion.direction == ModelCriterion.Criterion.DirectionEnum.lt

    mc_gt = mc > 10.0
    assert mc_gt.criterion.target == 10.0
    assert mc_gt.criterion.direction == ModelCriterion.Criterion.DirectionEnum.gt

    mc_eq = mc == 15.0
    assert mc_eq.criterion.target == 15.0
    assert mc_eq.criterion.direction == ModelCriterion.Criterion.DirectionEnum.eq


def test_criterion_and_operator():
    """Test the AND (&) operator for combining criteria."""
    mc1 = ModelCriterion(model_id="m1", measurement_name="p1") > 1
    mc2 = ModelCriterion(model_id="m2", measurement_name="p2") < 2
    nmc = n_mutations()

    combined = mc1 & mc2 & nmc
    assert isinstance(combined, Criterion)
    assert len(combined.root) == 3


def test_criteria_or_operator():
    """Test the OR (|) operator for combining criteria."""
    c1 = ModelCriterion(model_id="m1", measurement_name="p1") > 1
    c2 = ModelCriterion(model_id="m2", measurement_name="p2") < 2
    c3 = NMutationCriterion()

    combined = c1 | c2 | c3
    assert isinstance(combined, Criteria)
    assert len(combined.root) == 3
    assert all(isinstance(c, Criterion) for c in combined.root)


def test_design_constraint():
    """Test the DesignConstraint class."""
    sequence = "ACGT"
    dc = DesignConstraint(sequence)

    # Test initialization
    assert dc.as_dict() == {1: ["A"], 2: ["C"], 3: ["G"], 4: ["T"]}

    # Test allow
    dc.allow(amino_acids="V", positions=1)
    dc.allow(amino_acids=["I", "L"], positions=[2, 3])
    # Sort the lists for comparison to handle set ordering
    result_allow = {k: sorted(v) for k, v in dc.as_dict().items()}
    assert result_allow == {
        1: sorted(["A", "V"]),
        2: sorted(["C", "I", "L"]),
        3: sorted(["G", "I", "L"]),
        4: ["T"],
    }

    # Test remove
    dc.remove(amino_acids="A", positions=1)
    dc.remove(amino_acids=["L"], positions=[2, 3])
    result_remove = {k: sorted(v) for k, v in dc.as_dict().items()}
    assert result_remove == {
        1: ["V"],
        2: sorted(["C", "I"]),
        3: sorted(["G", "I"]),
        4: ["T"],
    }
