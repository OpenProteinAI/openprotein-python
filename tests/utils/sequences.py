"""Sequence generation utilities for testing."""

import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_sequence_fake(length: int) -> str:
    """
    Generate a random protein sequence of a given length.
    Uses the 20 standard amino acids.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(random.choice(amino_acids) for _ in range(length))


def random_sequence_real(length: int, real_proteins: list[str] | None = None) -> str:
    """
    Generate a realistic protein sequence by combining snippets from real proteins
    and filling gaps with biologically probable amino acids.
    """
    if length < 0:
        raise ValueError("length must be non-negative")

    # Default set of real protein sequences (you can expand this)
    if real_proteins is None:
        real_proteins = [
            "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
            "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
            "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
            "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL",
        ]

    # Amino acid frequencies based on natural proteins (approximate)
    aa_weights = {
        "A": 8.25,
        "R": 5.53,
        "N": 4.06,
        "D": 5.45,
        "C": 1.37,
        "Q": 3.93,
        "E": 6.75,
        "G": 7.07,
        "H": 2.27,
        "I": 5.96,
        "L": 9.66,
        "K": 5.84,
        "M": 2.42,
        "F": 3.86,
        "P": 4.70,
        "S": 6.56,
        "T": 5.34,
        "W": 1.08,
        "Y": 2.92,
        "V": 6.87,
    }

    amino_acids = list(aa_weights.keys())
    weights = list(aa_weights.values())

    def weighted_random_aa():
        return random.choices(amino_acids, weights=weights)[0]

    if length == 0:
        return ""

    sequence = ""

    while len(sequence) < length:
        remaining = length - len(sequence)

        # Choose a random protein and extract a snippet
        protein = random.choice(real_proteins)

        # Random snippet length (3-15 residues, but not more than remaining)
        snippet_length = min(random.randint(3, 15), remaining, len(protein))

        if snippet_length > 0:
            # Random starting position
            start_pos = random.randint(0, len(protein) - snippet_length)
            snippet = protein[start_pos : start_pos + snippet_length]
            sequence += snippet

        # Fill remaining space with weighted random amino acids
        remaining = length - len(sequence)
        if remaining > 0:
            # Add 1-5 random amino acids (or whatever fits)
            fill_length = min(random.randint(1, 5), remaining)
            for _ in range(fill_length):
                sequence += weighted_random_aa()

    return sequence[:length]  # Ensure exact length


def random_mutated_sequences(
    base_seq: str, num_sequences: int = 3, mutation_rate: float = 0.05
) -> list[str]:
    """
    Generates a list of sequences by introducing random mutations (substitutions,
    insertions, and deletions) to a base sequence. Includes the original base
    sequence in the returned list.
    """
    sequences = {base_seq}
    while len(sequences) < num_sequences:
        mutated_seq_list = []
        i = 0
        while i < len(base_seq):
            # Check for mutation
            if random.random() < mutation_rate:
                mutation_type = random.choice(["substitute", "insert", "delete"])

                if mutation_type == "substitute":
                    original_aa = base_seq[i]
                    new_aa = random.choice(
                        [aa for aa in AMINO_ACIDS if aa != original_aa]
                    )
                    mutated_seq_list.append(new_aa)

                elif mutation_type == "insert":
                    # insert before current residue
                    mutated_seq_list.append(random.choice(AMINO_ACIDS))
                    mutated_seq_list.append(base_seq[i])  # keep the original

                # For 'delete', we simply do not append the current residue,
                # effectively deleting it.

            else:  # No mutation
                mutated_seq_list.append(base_seq[i])

            i += 1

        # Add the new unique sequence
        sequences.add("".join(mutated_seq_list))

    return list(sequences)
