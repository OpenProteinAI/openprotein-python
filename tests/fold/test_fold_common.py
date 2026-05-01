"""Tests for fold request-time helpers in openprotein.fold.common."""

from unittest.mock import MagicMock

import numpy as np

import openprotein.common.residue_contants as RC
from openprotein.fold.common import resolve_templates
from openprotein.molecules.protein import Protein
from openprotein.molecules.template import Template


def _protein_with_structure() -> Protein:
    """Build a Protein carrying coordinates so Template's validation accepts it."""
    amino_acids = list(RC.restype_1to3.keys()) + ["X"]
    length = 6
    p = Protein(sequence="".join(amino_acids[i] for i in range(length)))
    p = p._set_coordinates(np.random.random(p.coordinates.shape).astype(np.float32))
    return p


def _mock_session(resolved_id: str = "query-uuid") -> MagicMock:
    session = MagicMock()
    session.prompt = MagicMock()
    session.prompt._resolve_query.return_value = resolved_id
    # Make isinstance(session.prompt, PromptAPI) succeed inside resolve_templates.
    from openprotein.prompt.prompt import PromptAPI

    session.prompt.__class__ = PromptAPI
    return session


def test_resolve_templates_omits_index_when_unset():
    """Default Template omits `index` on the wire (server treats as applies-to-all)."""
    session = _mock_session()
    tmpl = Template(template=_protein_with_structure())
    [out] = resolve_templates(session, [tmpl])
    assert "index" not in out


def test_resolve_templates_emits_index_list_when_set():
    """Explicit index is serialized as a plain list of ints."""
    session = _mock_session()
    tmpl = Template(template=_protein_with_structure(), index=[0, 1, 2])
    [out] = resolve_templates(session, [tmpl])
    assert out["index"] == [0, 1, 2]


def test_resolve_templates_index_from_tuple_normalizes_to_list():
    """Non-list Sequence inputs (e.g. tuple) land as a JSON-friendly list."""
    session = _mock_session()
    tmpl = Template(template=_protein_with_structure(), index=(3, 4, 5))
    [out] = resolve_templates(session, [tmpl])
    assert out["index"] == [3, 4, 5]


def test_resolve_templates_omits_index_intervals_when_unset():
    """Default Template omits `index_intervals` on the wire."""
    session = _mock_session()
    tmpl = Template(template=_protein_with_structure())
    [out] = resolve_templates(session, [tmpl])
    assert "index_intervals" not in out


def test_resolve_templates_emits_index_intervals_as_lists():
    """Each interval tuple is serialized as a 2-element list of ints."""
    session = _mock_session()
    tmpl = Template(
        template=_protein_with_structure(),
        index_intervals=[(0, 10), (20, 30)],
    )
    [out] = resolve_templates(session, [tmpl])
    assert out["index_intervals"] == [[0, 10], [20, 30]]


def test_resolve_templates_index_and_index_intervals_coexist():
    """`index` and `index_intervals` are independent fields and may both ship."""
    session = _mock_session()
    tmpl = Template(
        template=_protein_with_structure(),
        index=[5, 7],
        index_intervals=[(100, 200)],
    )
    [out] = resolve_templates(session, [tmpl])
    assert out["index"] == [5, 7]
    assert out["index_intervals"] == [[100, 200]]
