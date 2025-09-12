import re
import string

valid_id_pattern = re.compile(r"^[A-Z]{1,5}$|^\d{1,5}$")


def is_valid_id(id_str: str) -> bool:
    """
    Check if the id_str matches the valid pattern for IDs (1-5 uppercase or 1-5 digits).
    """
    if not id_str or len(id_str) > 5:
        return False
    return bool(valid_id_pattern.fullmatch(id_str))


def id_generator(used_ids: list[str] | None = None, max_alpha_len=5, max_numeric=99999):
    """
    Yields new chain IDs, skipping any in 'used_ids'.
    First A..Z, AA..ZZ, … up to max_alpha_len, then '1','2',… up to max_numeric.
    """
    used = set(tuple(used_ids or []))
    letters = list(string.ascii_uppercase)

    # --- Alphabetic IDs ---
    curr_len = 1
    curr_indices = [0] * curr_len  # start at 'A'

    def bump_indices():
        # lexicographically increment curr_indices; return False on overflow
        for i in reversed(range(len(curr_indices))):
            if curr_indices[i] < len(letters) - 1:
                curr_indices[i] += 1
                for j in range(i + 1, len(curr_indices)):
                    curr_indices[j] = 0
                return True
        return False

    while curr_len <= max_alpha_len:
        candidate = "".join(letters[i] for i in curr_indices)
        if candidate not in used:
            used.add(candidate)
            yield candidate
        # bump
        if not bump_indices():
            curr_len += 1
            if curr_len > max_alpha_len:
                break
            curr_indices = [0] * curr_len

    # --- Numeric IDs ---
    num = 1
    while num <= max_numeric:
        candidate = str(num)
        num += 1
        if candidate not in used:
            used.add(candidate)
            yield candidate

    # exhausted
    raise RuntimeError("exhausted all possible IDs")
