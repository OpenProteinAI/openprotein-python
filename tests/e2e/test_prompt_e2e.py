import pytest

from openprotein import OpenProtein
from openprotein.errors import InvalidParameterError
from openprotein.molecules import Complex, Protein
from openprotein.molecules.chains import DNA
from tests.utils.strings import random_string


@pytest.mark.e2e
def test_prompt_e2e(session: OpenProtein) -> None:
    """End-to-end test for the prompt feature."""
    # 1. Create a prompt
    prompt_name = f"test-prompt-{random_string()}"
    context = ["ACDEFGHIKLMNPQRSTVWY"]
    prompt = session.prompt.create_prompt(context, name=prompt_name)
    assert prompt.name == prompt_name
    assert prompt.id is not None

    # 2. Get the prompt
    retrieved_prompt = session.prompt.get_prompt(prompt.id)
    assert retrieved_prompt.id == prompt.id

    # 3. List prompts and verify the new prompt is in the list
    prompts = session.prompt.list_prompts()
    assert any(p.id == prompt.id for p in prompts)

    # 4. Create a query
    query_sequence = "MVLSEGEWQLVLHVWAKVEADVAGHGQ"
    query = session.prompt.create_query(query_sequence)
    assert query.id is not None

    # 5. Get the query
    retrieved_query = session.prompt.get_query(query.id)
    assert retrieved_query.id == query.id

    # 6. Check the content of the retrieved query
    retrieved_protein = retrieved_query.get()
    assert isinstance(retrieved_protein, Protein)
    assert retrieved_protein.sequence.decode() == query_sequence


@pytest.mark.e2e
def test_prompt_editing_e2e(session: OpenProtein) -> None:
    """End-to-end test for editing prompts and queries."""
    # 1. Create an initial prompt (context)
    prompt_name_1 = f"test-prompt-edit-1-{random_string()}"
    context_1 = ["ACDEFGHIKLMNPQRSTVWY"]
    prompt_1 = session.prompt.create_prompt(context_1, name=prompt_name_1)
    assert prompt_1.name == prompt_name_1

    # 2. "Edit" the prompt by creating a new one with additional context
    prompt_name_2 = f"test-prompt-edit-2-{random_string()}"
    context_2 = prompt_1.get()  # Get existing context
    context_2.append(
        [Protein(sequence="MVLSEGEWQLVLHVWAKVEADVAGHGQ")]
    )  # Add new context
    prompt_2 = session.prompt.create_prompt(context_2, name=prompt_name_2)
    assert prompt_2.name == prompt_name_2
    assert prompt_2.num_replicates == 2

    # 3. Create a query from a sequence
    query_sequence_1 = "GSHSMRYFFTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDGETRKVKAHSQTHRVDLGTLRGYYNQSEAGSHTVQRMYGCDVGSDWRFLRGYHQYAYDGKDYIALKEDLRSWTAADMAAQTTKHKWEAAKVCSAAKS"
    query_1 = session.prompt.create_query(query_sequence_1)
    assert query_1.id is not None

    # 4. "Edit" the query by masking part of the sequence
    protein_to_edit = query_1.get()
    assert isinstance(protein_to_edit, Protein)
    masked_protein = protein_to_edit.mask_sequence_except_at([1, 2, 3, 4, 5])
    query_2 = session.prompt.create_query(masked_protein)
    assert query_2.id is not None
    assert query_1.id != query_2.id

    # 5. Retrieve the edited query and verify the mask
    retrieved_edited_protein = query_2.get()
    assert isinstance(retrieved_edited_protein, Protein)
    assert retrieved_edited_protein.sequence.startswith(b"GSHSM")
    assert retrieved_edited_protein.sequence.count(ord("X")) > 10


@pytest.mark.e2e
def test_multichain_prompt_and_query_e2e(session: OpenProtein) -> None:
    """End-to-end roundtrip for multichain prompts and queries."""
    chain_a, chain_b = "ACDEFGHIKLMNPQRSTVWY", "MVLSEGEWQLVLHVWAKVEADVAGHGQ"

    # 1. Multichain context as a raw ":" sequence
    prompt_name = f"test-multichain-{random_string()}"
    prompt = session.prompt.create_prompt([f"{chain_a}:{chain_b}"], name=prompt_name)
    retrieved = prompt.get()
    entry = retrieved[0][0]
    assert isinstance(entry, Complex)
    seqs = sorted(p.sequence.decode() for p in entry.get_proteins().values())
    assert seqs == sorted([chain_a, chain_b])

    # 2. Multichain context as a Complex
    complex_in = Complex(
        {"A": Protein(sequence=chain_a), "B": Protein(sequence=chain_b)},
        name=f"complex-{random_string()}",
    )
    prompt_2 = session.prompt.create_prompt([complex_in])
    entry_2 = prompt_2.get()[0][0]
    assert isinstance(entry_2, Complex)

    # 3. Multichain query roundtrip
    query = session.prompt.create_query(f"{chain_a}:{chain_b}")
    retrieved_query = query.get()
    assert isinstance(retrieved_query, Complex)
    assert len(retrieved_query.get_proteins()) == 2

    # 4. get_as_complex still returns a Complex for a single-chain query
    single_query = session.prompt.create_query(chain_a)
    as_complex = single_query.get_as_complex()
    assert isinstance(as_complex, Complex)
    assert len(as_complex.get_proteins()) == 1


@pytest.mark.e2e
def test_non_protein_complex_rejected_clientside(session: OpenProtein) -> None:
    """Complex with DNA/RNA/Ligand chains is rejected before any HTTP call."""
    bad = Complex({"A": Protein(sequence="ACDE"), "B": DNA(sequence="ACGT")})
    with pytest.raises(InvalidParameterError, match="protein chains"):
        session.prompt.create_prompt([bad])
    with pytest.raises(InvalidParameterError, match="protein chains"):
        session.prompt.create_query(bad)
