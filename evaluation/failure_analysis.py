from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from baselines.morse_rules import (
    FAILURE_SYMBOL_MAPPING,
    FAILURE_TIMING_PARSE,
    FAILURE_TONE_DETECTION,
)

KNOWN_FAILURE_TYPES = (
    FAILURE_TONE_DETECTION,
    FAILURE_TIMING_PARSE,
    FAILURE_SYMBOL_MAPPING,
)


def summarize_failures(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    total = len(records)
    counter: Counter[str] = Counter()
    num_failures = 0

    for record in records:
        if not bool(record.get("is_failure", False)):
            continue

        num_failures += 1
        failure_type = str(record.get("failure_type") or "unknown_failure")
        counter[failure_type] += 1

    breakdown: dict[str, dict[str, float | int]] = {}
    for failure_type in KNOWN_FAILURE_TYPES:
        count = int(counter.get(failure_type, 0))
        breakdown[failure_type] = {
            "count": count,
            "rate": (count / total) if total else 0.0,
        }

    if counter.get("unknown_failure"):
        count = int(counter["unknown_failure"])
        breakdown["unknown_failure"] = {
            "count": count,
            "rate": (count / total) if total else 0.0,
        }

    return {
        "num_failures": num_failures,
        "decode_failure_rate": (num_failures / total) if total else 0.0,
        "failure_breakdown": breakdown,
    }
