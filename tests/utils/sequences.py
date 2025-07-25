"""Sequence generation utilities for testing."""

import random

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_mutated_sequences(
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
