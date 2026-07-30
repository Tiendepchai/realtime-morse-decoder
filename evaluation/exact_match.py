from __future__ import annotations

from src.utils.text import normalize_text


def compute_exact_match(
    prediction: str,
    reference: str,
    already_normalized: bool = False,
) -> int:
    if already_normalized:
        return int(prediction == reference)
    return int(normalize_text(prediction) == normalize_text(reference))
