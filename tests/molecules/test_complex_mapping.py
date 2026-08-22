from openprotein.molecules import Complex, Protein, Ligand

import gemmi

import pytest


def test_chains_keep_insertion_order():
    complex = Complex({"C": Protein.from_expr(6), "A": Protein.from_expr(7)})
    assert list(complex.keys()) == ["C", "A"]
    complex.set_chain("B", Protein.from_expr(8))
    assert list(complex.keys()) == ["C", "A", "B"]


def test_serialized_chain_order_follows_insertion_order():
    complex = Complex({"C": Protein.from_expr(6), "A": Protein.from_expr(7)})
    block = gemmi.cif.read_string(complex.to_string()).sole_block()
    assert list(block.find_loop("_struct_asym.id")) == ["C", "A"]


def test_mapping_protocol_over_chains():
    protein = Protein.from_expr(6)
    ligand = Ligand(ccd="HEM")
    complex = Complex({"A": protein, "B": ligand})
    assert len(complex) == 2
    assert "A" in complex and "Z" not in complex
    assert complex["A"] is protein
    assert list(complex) == ["A", "B"]
    assert list(complex.get_chain_ids()) == list(complex.keys()) == ["A", "B"]
    assert list(complex.values()) == [protein, ligand]
    assert list(complex.items()) == [("A", protein), ("B", ligand)]
    replacement = Protein.from_expr(9)
    complex["A"] = replacement
    assert complex["A"] is replacement
    del complex["B"]
    assert list(complex.keys()) == ["A"]
    with pytest.raises(KeyError):
        complex["B"]


def test_complex_is_not_hashable():
    with pytest.raises(TypeError, match="unhashable"):
        hash(Complex({"A": Protein.from_expr(6)}))


def test_insert_chain_takes_the_lowest_unused_id():
    complex = Complex({"B": Protein.from_expr(6), "C": Protein.from_expr(7)})
    complex.insert_chain(Protein.from_expr(8))
    assert list(complex.keys()) == ["B", "C", "A"]


def test_update_rejects_overlapping_chain_ids():
    complex = Complex({"A": Protein.from_expr(6)})
    with pytest.raises(ValueError, match="overlapping chain ids"):
        complex.update(A=Protein.from_expr(7))


def test_clear_drops_chains_and_templates():
    complex = Complex({"A": Protein.from_expr(6)}).set_templates(
        [Protein.from_expr(8)]
    )
    complex.clear()
    assert len(complex) == 0
    assert complex.templates == ()


def test_equality_compares_chains():
    protein = Protein.from_expr(6)
    assert Complex({"A": protein}) == Complex({"A": protein})
    assert Complex({"A": protein}) != Complex({"B": protein})
    assert Complex({"A": protein}) != {"A": protein}


def test_or_returns_a_new_complex_leaving_operands_unchanged():
    left = Complex({"A": Protein.from_expr(6)})
    right = Complex({"B": Protein.from_expr(7)})
    combined = left | right
    assert list(combined.keys()) == ["A", "B"]
    assert list((left & right).keys()) == ["A", "B"]
    assert list(left.keys()) == ["A"]
    assert list(right.keys()) == ["B"]


def test_ior_adds_a_chain_in_place():
    complex = Complex({"A": Protein.from_expr(6)})
    complex |= Protein.from_expr(7)
    assert list(complex.keys()) == ["A", "B"]


def test_protein_operators_build_a_two_chain_complex():
    combined = Protein.from_expr(6) | Protein.from_expr(7)
    assert list(combined.keys()) == ["A", "B"]
    combined = "MKV" & Protein.from_expr(6)
    assert isinstance(combined["A"], Protein)
    assert combined["A"].sequence == b"MKV"


@pytest.mark.xfail(
    reason="Complex.copy() returns a Complex with no templates, so combining drops them"
)
def test_templates_survive_combining():
    complex = Complex({"A": Protein.from_expr(6)}).set_templates(
        [Protein.from_expr(8)]
    )
    assert len((complex | Protein.from_expr(7)).templates) == 1
