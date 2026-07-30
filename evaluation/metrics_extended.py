from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .exact_match import compute_exact_match
from .failure_analysis import summarize_failures
from src.utils.text import normalize_text


def levenshtein_distance(reference: Sequence[Any], prediction: Sequence[Any]) -> int:
    rows = len(reference)
    cols = len(prediction)
    if rows == 0:
        return cols
    if cols == 0:
        return rows

    previous = list(range(cols + 1))
    current = [0] * (cols + 1)

    for row in range(1, rows + 1):
        current[0] = row
        for col in range(1, cols + 1):
            substitution_cost = 0 if reference[row - 1] == prediction[col - 1] else 1
            current[col] = min(
                previous[col] + 1,
                current[col - 1] + 1,
                previous[col - 1] + substitution_cost,
            )
        previous, current = current, previous

    return previous[cols]


def calculate_cer(
    prediction: str,
    reference: str,
    already_normalized: bool = False,
) -> float:
    if already_normalized:
        prediction_normalized = prediction
        reference_normalized = reference
    else:
        prediction_normalized = normalize_text(prediction)
        reference_normalized = normalize_text(reference)

    distance = levenshtein_distance(list(reference_normalized), list(prediction_normalized))
    denominator = max(1, len(reference_normalized))
    if len(reference_normalized) == 0 and len(prediction_normalized) == 0:
        return 0.0
    return distance / denominator


def calculate_wer(
    prediction: str,
    reference: str,
    already_normalized: bool = False,
) -> float:
    if already_normalized:
        prediction_normalized = prediction
        reference_normalized = reference
    else:
        prediction_normalized = normalize_text(prediction)
        reference_normalized = normalize_text(reference)

    prediction_words = prediction_normalized.split()
    reference_words = reference_normalized.split()
    distance = levenshtein_distance(reference_words, prediction_words)
    denominator = max(1, len(reference_words))
    if len(reference_words) == 0 and len(prediction_words) == 0:
        return 0.0
    return distance / denominator


def enrich_prediction_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for record in records:
        reference_normalized = normalize_text(str(record.get("reference", "")))
        prediction_normalized = normalize_text(str(record.get("prediction", "")))

        output = dict(record)
        output["reference_normalized"] = reference_normalized
        output["prediction_normalized"] = prediction_normalized
        output["exact_match"] = compute_exact_match(
            prediction_normalized,
            reference_normalized,
            already_normalized=True,
        )
        output["cer_sample"] = calculate_cer(
            prediction_normalized,
            reference_normalized,
            already_normalized=True,
        )
        output["wer_sample"] = calculate_wer(
            prediction_normalized,
            reference_normalized,
            already_normalized=True,
        )

        if reference_normalized and not prediction_normalized:
            output["is_failure"] = bool(output.get("is_failure", False) or True)
            output["failure_type"] = str(output.get("failure_type") or "timing_parse_failure")
        else:
            output["is_failure"] = bool(output.get("is_failure", False))
            output["failure_type"] = str(output.get("failure_type") or "")

        enriched.append(output)
    return enriched


def aggregate_metrics(
    records: Iterable[Mapping[str, Any]],
    method: str | None = None,
) -> dict[str, Any]:
    enriched = enrich_prediction_records(records)
    total_samples = len(enriched)

    total_char_edits = 0
    total_chars = 0
    total_word_edits = 0
    total_words = 0
    exact_matches = 0

    for record in enriched:
        reference_normalized = str(record["reference_normalized"])
        prediction_normalized = str(record["prediction_normalized"])
        total_char_edits += levenshtein_distance(list(reference_normalized), list(prediction_normalized))
        total_chars += len(reference_normalized)
        total_word_edits += levenshtein_distance(reference_normalized.split(), prediction_normalized.split())
        total_words += len(reference_normalized.split())
        exact_matches += int(record["exact_match"])

    if total_chars == 0:
        cer = 0.0 if total_char_edits == 0 else 1.0
    else:
        cer = total_char_edits / total_chars

    if total_words == 0:
        wer = 0.0 if total_word_edits == 0 else 1.0
    else:
        wer = total_word_edits / total_words

    failure_summary = summarize_failures(enriched)
    resolved_method = method or (str(enriched[0].get("method", "")) if enriched else "")
    return {
        "method": resolved_method,
        "num_samples": total_samples,
        "cer": cer,
        "wer": wer,
        "exact_match_rate": (exact_matches / total_samples) if total_samples else 0.0,
        "num_failures": failure_summary["num_failures"],
        "decode_failure_rate": failure_summary["decode_failure_rate"],
        "failure_breakdown": failure_summary["failure_breakdown"],
    }


def summarize_by_field(
    records: Iterable[Mapping[str, Any]],
    field_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        value = record.get(field_name)
        if value is None or str(value) == "":
            continue
        grouped.setdefault(str(value), []).append(record)

    summaries: list[dict[str, Any]] = []
    for field_value in sorted(grouped.keys()):
        summary = aggregate_metrics(grouped[field_value])
        summary[field_name] = field_value
        summaries.append(summary)
    return summaries
