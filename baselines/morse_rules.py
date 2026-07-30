from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.utils.text import normalize_text

MORSE_TO_CHAR = {
    ".-": "A",
    "-...": "B",
    "-.-.": "C",
    "-..": "D",
    ".": "E",
    "..-.": "F",
    "--.": "G",
    "....": "H",
    "..": "I",
    ".---": "J",
    "-.-": "K",
    ".-..": "L",
    "--": "M",
    "-.": "N",
    "---": "O",
    ".--.": "P",
    "--.-": "Q",
    ".-.": "R",
    "...": "S",
    "-": "T",
    "..-": "U",
    "...-": "V",
    ".--": "W",
    "-..-": "X",
    "-.--": "Y",
    "--..": "Z",
    ".----": "1",
    "..---": "2",
    "...--": "3",
    "....-": "4",
    ".....": "5",
    "-....": "6",
    "--...": "7",
    "---..": "8",
    "----.": "9",
    "-----": "0",
}
CHAR_TO_MORSE = {char: code for code, char in MORSE_TO_CHAR.items()}

FAILURE_TONE_DETECTION = "tone_detection_failure"
FAILURE_TIMING_PARSE = "timing_parse_failure"
FAILURE_SYMBOL_MAPPING = "symbol_mapping_failure"


@dataclass(frozen=True)
class DurationRules:
    """
    Morse timing rules in dot units.

    Standard Morse timing:
    - dot: 1 unit
    - dash: 3 units
    - intra-element gap: 1 unit
    - inter-letter gap: 3 units
    - inter-word gap: 7 units
    """

    dot_dash_split_units: float = 2.0
    letter_gap_split_units: float = 2.5
    word_gap_split_units: float = 6.0
    max_tone_units: float = 8.0
    max_gap_units: float = 32.0


@dataclass(frozen=True)
class MorseRun:
    is_tone: bool
    frames: int
    duration_s: float


@dataclass
class MorseDecodeOutcome:
    prediction: str
    is_failure: bool
    failure_type: Optional[str]
    failure_details: list[str] = field(default_factory=list)
    dot_unit_s: Optional[float] = None
    morse_tokens: list[str] = field(default_factory=list)


def _flush_current_symbol(
    current_symbol: list[str],
    decoded_chars: list[str],
    invalid_symbols: list[str],
    morse_tokens: list[str],
) -> None:
    if not current_symbol:
        return

    code = "".join(current_symbol)
    morse_tokens.append(code)
    mapped = MORSE_TO_CHAR.get(code)
    if mapped is None:
        invalid_symbols.append(code)
    else:
        decoded_chars.append(mapped)
    current_symbol.clear()


def decode_runs_to_text(
    runs: Sequence[MorseRun],
    dot_unit_s: Optional[float],
    rules: DurationRules,
) -> MorseDecodeOutcome:
    tone_count = sum(1 for run in runs if run.is_tone and run.duration_s > 0.0)
    if tone_count == 0:
        return MorseDecodeOutcome(
            prediction="",
            is_failure=True,
            failure_type=FAILURE_TONE_DETECTION,
            failure_details=["no_tone_runs_detected"],
            dot_unit_s=dot_unit_s,
        )

    if dot_unit_s is None or dot_unit_s <= 0.0:
        return MorseDecodeOutcome(
            prediction="",
            is_failure=True,
            failure_type=FAILURE_TIMING_PARSE,
            failure_details=["dot_unit_estimation_failed"],
            dot_unit_s=dot_unit_s,
        )

    decoded_chars: list[str] = []
    current_symbol: list[str] = []
    invalid_symbols: list[str] = []
    timing_errors: list[str] = []
    morse_tokens: list[str] = []

    for run in runs:
        if run.duration_s <= 0.0:
            continue

        units = run.duration_s / dot_unit_s

        if run.is_tone:
            if units > rules.max_tone_units:
                timing_errors.append(f"tone_run_too_long:{run.duration_s:.5f}s")
                continue

            symbol = "." if units < rules.dot_dash_split_units else "-"
            current_symbol.append(symbol)
            continue

        if not current_symbol and not decoded_chars:
            continue

        if units >= rules.max_gap_units:
            units = rules.max_gap_units

        if units < rules.letter_gap_split_units:
            continue

        _flush_current_symbol(current_symbol, decoded_chars, invalid_symbols, morse_tokens)

        if units >= rules.word_gap_split_units and decoded_chars and decoded_chars[-1] != " ":
            decoded_chars.append(" ")

    _flush_current_symbol(current_symbol, decoded_chars, invalid_symbols, morse_tokens)

    prediction = normalize_text("".join(decoded_chars))
    failure_details = [*timing_errors, *[f"invalid_symbol:{symbol}" for symbol in invalid_symbols]]
    failure_type: Optional[str] = None

    if invalid_symbols:
        failure_type = FAILURE_SYMBOL_MAPPING
    elif timing_errors:
        failure_type = FAILURE_TIMING_PARSE
    elif not prediction:
        failure_type = FAILURE_TIMING_PARSE
        failure_details.append("empty_prediction_after_decoding")

    return MorseDecodeOutcome(
        prediction=prediction,
        is_failure=failure_type is not None,
        failure_type=failure_type,
        failure_details=failure_details,
        dot_unit_s=dot_unit_s,
        morse_tokens=morse_tokens,
    )
