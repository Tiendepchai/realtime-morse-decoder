from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import scipy.signal
import soundfile as sf

from baselines.morse_rules import CHAR_TO_MORSE, MorseRun
from baselines.signal_utils import (
    bandpass_filter,
    build_frequency_candidates,
    estimate_dominant_frequency,
    finalize_detection_scores,
    frame_signal,
    goertzel_power,
    load_audio_mono,
)
from baselines.timing_estimation import DotEstimationConfig, estimate_dot_unit
from src.utils.text import normalize_text


def ensure_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.reshape(-1)


def canonicalize_label_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""

    tokens = normalized.split()
    if tokens and all(len(token) == 1 for token in tokens):
        normalized = "".join(tokens)

    supported_chars = set(CHAR_TO_MORSE) | {" "}
    return "".join(char for char in normalized if char in supported_chars)


def format_label_text(text: str, label_format: str) -> str:
    canonical = canonicalize_label_text(text)
    if label_format == "compact":
        return canonical
    if label_format != "char_spaced":
        raise ValueError(f"Unsupported label format: {label_format}")
    if not canonical or " " in canonical:
        return canonical
    return " ".join(list(canonical))


@dataclass(frozen=True)
class ExpectedUnit:
    index: int
    kind: str
    is_tone: bool
    nominal_units: float
    char: str
    symbol: str
    word_index: int
    char_index: int
    symbol_index: int


@dataclass(frozen=True)
class DetectedRun:
    index: int
    is_tone: bool
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int
    duration_s: float
    units: float | None


@dataclass(frozen=True)
class DetectionConfig:
    sample_rate: int = 16000
    target_frequency_hz: float | None = None
    auto_frequency: bool = True
    frequency_search_low_hz: float = 400.0
    frequency_search_high_hz: float = 1200.0
    frequency_tolerance_hz: float = 80.0
    neighboring_bins: int = 1
    bandpass_half_width_hz: float = 150.0
    frame_length_ms: float = 20.0
    hop_length_ms: float = 10.0
    smoothing_window: int = 3
    threshold_mode: str = "adaptive_percentile"
    hysteresis_high: float = 0.55
    hysteresis_low: float = 0.35
    min_tone_duration_ms: float = 30.0
    fill_gap_duration_ms: float = 20.0
    dot_estimation: DotEstimationConfig = field(default_factory=DotEstimationConfig)


@dataclass(frozen=True)
class DetectionArtifacts:
    sample_rate: int
    target_frequency_hz: float | None
    dot_unit_s: float | None
    frame_length: int
    hop_length: int
    high_threshold: float
    low_threshold: float
    runs: list[DetectedRun]


@dataclass(frozen=True)
class AlignmentConfig:
    tone_weight: float = 1.0
    gap_weight: float = 0.75
    skip_expected_base: float = 1.0
    skip_expected_scale: float = 0.2
    skip_observed_base: float = 1.0
    skip_observed_scale: float = 0.15
    mismatch_penalty: float = 2.5


@dataclass(frozen=True)
class AlignmentMatch:
    expected: ExpectedUnit
    observed: DetectedRun
    cost: float


@dataclass(frozen=True)
class AlignmentResult:
    total_cost: float
    matched_count: int
    expected_count: int
    observed_count: int
    coverage: float
    matches: list[AlignmentMatch]


@dataclass(frozen=True)
class SynthesisConfig:
    jitter_ratio: float = 0.05
    crossfade_ms: float = 4.0
    peak_normalize: float = 0.98


@dataclass(frozen=True)
class GapFormattingResult:
    raw_text: str
    canonical_text: str
    formatted_text: str
    boundary_units: list[float]
    word_boundary_flags: list[bool]
    threshold_units: float | None
    guided_dot_unit_s: float | None


def build_expected_units(text: str) -> list[ExpectedUnit]:
    canonical = canonicalize_label_text(text)
    if not canonical:
        return []

    expected: list[ExpectedUnit] = []
    words = canonical.split()
    global_char_index = 0
    global_unit_index = 0

    for word_index, word in enumerate(words):
        for word_char_index, char in enumerate(word):
            code = CHAR_TO_MORSE.get(char)
            if not code:
                continue

            for symbol_index, symbol in enumerate(code):
                expected.append(
                    ExpectedUnit(
                        index=global_unit_index,
                        kind="dot" if symbol == "." else "dash",
                        is_tone=True,
                        nominal_units=1.0 if symbol == "." else 3.0,
                        char=char,
                        symbol=symbol,
                        word_index=word_index,
                        char_index=global_char_index,
                        symbol_index=symbol_index,
                    )
                )
                global_unit_index += 1

                if symbol_index < len(code) - 1:
                    expected.append(
                        ExpectedUnit(
                            index=global_unit_index,
                            kind="intra_gap",
                            is_tone=False,
                            nominal_units=1.0,
                            char=char,
                            symbol="",
                            word_index=word_index,
                            char_index=global_char_index,
                            symbol_index=symbol_index,
                        )
                    )
                    global_unit_index += 1

            if word_char_index < len(word) - 1:
                expected.append(
                    ExpectedUnit(
                        index=global_unit_index,
                        kind="letter_gap",
                        is_tone=False,
                        nominal_units=3.0,
                        char=char,
                        symbol="",
                        word_index=word_index,
                        char_index=global_char_index,
                        symbol_index=len(code),
                    )
                )
                global_unit_index += 1

            global_char_index += 1

        if word_index < len(words) - 1:
            expected.append(
                ExpectedUnit(
                    index=global_unit_index,
                    kind="word_gap",
                    is_tone=False,
                    nominal_units=7.0,
                    char=" ",
                    symbol="",
                    word_index=word_index,
                    char_index=global_char_index,
                    symbol_index=0,
                )
            )
            global_unit_index += 1

    return expected


def _select_relative_word_gap_threshold(
    boundary_units: Sequence[float],
    min_cluster_ratio: float = 1.8,
    min_cluster_delta_units: float = 2.0,
) -> float | None:
    values = np.asarray([float(value) for value in boundary_units if np.isfinite(value) and value > 0.0], dtype=np.float64)
    if values.size < 2:
        return None

    values.sort()
    best_ratio = 0.0
    best_threshold = None
    for index in range(values.size - 1):
        left = float(values[index])
        right = float(values[index + 1])
        ratio = (right + 1e-6) / (left + 1e-6)
        delta = right - left
        if ratio >= min_cluster_ratio and delta >= min_cluster_delta_units and ratio > best_ratio:
            best_ratio = ratio
            best_threshold = (left + right) / 2.0
    return best_threshold


def infer_gap_formatted_text(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
    detection_config: DetectionConfig | None = None,
    min_cluster_ratio: float = 1.5,
    min_cluster_delta_units: float = 2.0,
) -> GapFormattingResult:
    raw_text = normalize_text(text)
    canonical_text = canonicalize_label_text(raw_text).replace(" ", "")
    if len(canonical_text) <= 1:
        fallback_text = canonical_text or raw_text
        return GapFormattingResult(
            raw_text=raw_text,
            canonical_text=canonical_text,
            formatted_text=fallback_text,
            boundary_units=[],
            word_boundary_flags=[],
            threshold_units=None,
            guided_dot_unit_s=None,
        )

    config = detection_config or DetectionConfig(sample_rate=sample_rate)
    if int(config.sample_rate) != int(sample_rate):
        config = replace(config, sample_rate=int(sample_rate))

    detection = detect_morse_runs(audio, config=config)
    expected = build_expected_units(canonical_text)
    if not detection.runs or not expected:
        return GapFormattingResult(
            raw_text=raw_text,
            canonical_text=canonical_text,
            formatted_text=canonical_text,
            boundary_units=[],
            word_boundary_flags=[],
            threshold_units=None,
            guided_dot_unit_s=None,
        )

    alignment = align_expected_to_observed(expected, detection.runs, config=AlignmentConfig())
    matched_boundary_units: dict[int, float] = {}
    for match in alignment.matches:
        if match.expected.kind != "letter_gap" or match.observed.is_tone:
            continue
        if match.observed.units is None or match.observed.units <= 0.0:
            continue
        matched_boundary_units[int(match.expected.char_index)] = float(match.observed.units)

    boundary_units = [
        matched_boundary_units[index]
        for index in range(len(canonical_text) - 1)
        if index in matched_boundary_units
    ]
    guided_dot_s = detection.dot_unit_s
    if len(boundary_units) != max(0, len(canonical_text) - 1):
        boundaries, guided_dot_s, _ = _build_guided_boundaries(
            expected=expected,
            runs=detection.runs,
            audio_length=len(ensure_mono_float32(audio)),
            sample_rate=int(sample_rate),
        )
        boundary_units = []
        for unit_index, expected_unit in enumerate(expected):
            if expected_unit.kind != "letter_gap":
                continue
            start_sample = int(boundaries[unit_index])
            end_sample = int(boundaries[unit_index + 1])
            duration_s = max(0.0, float(end_sample - start_sample) / float(sample_rate))
            if guided_dot_s is None or guided_dot_s <= 0.0:
                continue
            boundary_units.append(duration_s / float(guided_dot_s))

    threshold_units = _select_relative_word_gap_threshold(
        boundary_units,
        min_cluster_ratio=min_cluster_ratio,
        min_cluster_delta_units=min_cluster_delta_units,
    )
    word_boundary_flags = [
        bool(threshold_units is not None and units >= threshold_units)
        for units in boundary_units
    ]

    formatted_parts: list[str] = []
    for index, char in enumerate(canonical_text):
        formatted_parts.append(char)
        if index < len(word_boundary_flags) and word_boundary_flags[index]:
            formatted_parts.append(" ")
    formatted_text = normalize_text("".join(formatted_parts), restrict_charset=False)

    return GapFormattingResult(
        raw_text=raw_text,
        canonical_text=canonical_text,
        formatted_text=formatted_text or canonical_text,
        boundary_units=[float(units) for units in boundary_units],
        word_boundary_flags=word_boundary_flags,
        threshold_units=None if threshold_units is None else float(threshold_units),
        guided_dot_unit_s=None if guided_dot_s is None else float(guided_dot_s),
    )


def _mask_to_detected_runs(
    mask: np.ndarray,
    frame_length: int,
    hop_length: int,
    sample_rate: int,
    audio_length: int,
    dot_unit_s: float | None,
) -> list[DetectedRun]:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    runs: list[DetectedRun] = []
    start = 0
    run_index = 0
    while start < mask.size:
        state = bool(mask[start])
        end = start + 1
        while end < mask.size and bool(mask[end]) == state:
            end += 1

        start_sample = max(0, start * hop_length)
        end_sample = min(audio_length, max(start_sample + 1, (end - 1) * hop_length + frame_length))
        duration_s = float((end - start) * hop_length) / float(sample_rate)
        units = None if dot_unit_s is None or dot_unit_s <= 0.0 else duration_s / dot_unit_s
        runs.append(
            DetectedRun(
                index=run_index,
                is_tone=state,
                start_frame=start,
                end_frame=end,
                start_sample=start_sample,
                end_sample=end_sample,
                duration_s=duration_s,
                units=units,
            )
        )
        run_index += 1
        start = end
    return runs


def detect_morse_runs(audio: np.ndarray, config: DetectionConfig) -> DetectionArtifacts:
    audio = ensure_mono_float32(audio)
    if audio.size == 0:
        return DetectionArtifacts(
            sample_rate=config.sample_rate,
            target_frequency_hz=None,
            dot_unit_s=None,
            frame_length=0,
            hop_length=0,
            high_threshold=1.0,
            low_threshold=0.0,
            runs=[],
        )

    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    if peak > 0.0:
        audio = audio / peak

    target_frequency_hz = config.target_frequency_hz
    if target_frequency_hz is None and config.auto_frequency:
        target_frequency_hz = estimate_dominant_frequency(
            audio,
            sample_rate=config.sample_rate,
            search_low_hz=config.frequency_search_low_hz,
            search_high_hz=config.frequency_search_high_hz,
        )

    working = audio
    if target_frequency_hz is not None and config.bandpass_half_width_hz > 0.0:
        lowcut = max(config.frequency_search_low_hz, float(target_frequency_hz) - config.bandpass_half_width_hz)
        highcut = min(config.frequency_search_high_hz, float(target_frequency_hz) + config.bandpass_half_width_hz)
        working = bandpass_filter(audio, config.sample_rate, lowcut, highcut, order=4)

    frame_length = max(1, int(round(config.sample_rate * config.frame_length_ms / 1000.0)))
    hop_length = max(1, int(round(config.sample_rate * config.hop_length_ms / 1000.0)))
    min_tone_frames = max(1, int(round(config.min_tone_duration_ms / config.hop_length_ms)))
    max_gap_frames = max(1, int(round(config.fill_gap_duration_ms / config.hop_length_ms)))

    frames = frame_signal(working, frame_length=frame_length, hop_length=hop_length)
    if frames.size == 0:
        return DetectionArtifacts(
            sample_rate=config.sample_rate,
            target_frequency_hz=target_frequency_hz,
            dot_unit_s=None,
            frame_length=frame_length,
            hop_length=hop_length,
            high_threshold=1.0,
            low_threshold=0.0,
            runs=[],
        )

    if target_frequency_hz is None:
        scores = np.mean(np.square(frames), axis=1).astype(np.float32)
    else:
        frequency_candidates = build_frequency_candidates(
            target_frequency_hz=float(target_frequency_hz),
            tolerance_hz=config.frequency_tolerance_hz,
            neighboring_bins=config.neighboring_bins,
            sample_rate=config.sample_rate,
            frame_length=frame_length,
        )
        if not frequency_candidates:
            scores = np.mean(np.square(frames), axis=1).astype(np.float32)
        else:
            narrowband_power = goertzel_power(
                frames,
                sample_rate=config.sample_rate,
                frequencies_hz=frequency_candidates,
            )
            frame_power = np.mean(np.square(frames), axis=1).astype(np.float64)
            frame_power = np.maximum(frame_power, 1e-8)
            scores = np.log1p(np.max(narrowband_power, axis=1) / frame_power).astype(np.float32)

    detection = finalize_detection_scores(
        raw_scores=scores,
        threshold_mode=config.threshold_mode,
        hysteresis_high=config.hysteresis_high,
        hysteresis_low=config.hysteresis_low,
        smoothing_window=config.smoothing_window,
        min_tone_frames=min_tone_frames,
        max_gap_frames=max_gap_frames,
    )

    timing_runs = [
        run
        for run in _mask_to_detected_runs(
            detection.mask,
            frame_length=frame_length,
            hop_length=hop_length,
            sample_rate=config.sample_rate,
            audio_length=len(audio),
            dot_unit_s=None,
        )
    ]
    dot_unit_s, _ = estimate_dot_unit(
        [
            MorseRun(is_tone=run.is_tone, frames=max(1, run.end_frame - run.start_frame), duration_s=run.duration_s)
            for run in timing_runs
        ],
        config.dot_estimation,
    )

    runs = _mask_to_detected_runs(
        detection.mask,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=config.sample_rate,
        audio_length=len(audio),
        dot_unit_s=dot_unit_s,
    )
    return DetectionArtifacts(
        sample_rate=config.sample_rate,
        target_frequency_hz=target_frequency_hz,
        dot_unit_s=dot_unit_s,
        frame_length=frame_length,
        hop_length=hop_length,
        high_threshold=detection.high_threshold,
        low_threshold=detection.low_threshold,
        runs=runs,
    )


def _match_cost(expected: ExpectedUnit, observed: DetectedRun, config: AlignmentConfig) -> float:
    observed_units = observed.units if observed.units is not None and observed.units > 0.0 else 1.0
    duration_cost = abs(np.log((observed_units + 1e-6) / (expected.nominal_units + 1e-6)))
    duration_cost *= config.tone_weight if expected.is_tone else config.gap_weight
    if expected.is_tone != observed.is_tone:
        duration_cost += config.mismatch_penalty
    return float(duration_cost)


def _skip_expected_cost(expected: ExpectedUnit, config: AlignmentConfig) -> float:
    return float(config.skip_expected_base + config.skip_expected_scale * expected.nominal_units)


def _skip_observed_cost(observed: DetectedRun, config: AlignmentConfig) -> float:
    observed_units = observed.units if observed.units is not None and observed.units > 0.0 else 1.0
    return float(config.skip_observed_base + config.skip_observed_scale * observed_units)


def align_expected_to_observed(
    expected: Sequence[ExpectedUnit],
    observed: Sequence[DetectedRun],
    config: AlignmentConfig,
) -> AlignmentResult:
    expected = list(expected)
    observed = list(observed)
    rows = len(expected)
    cols = len(observed)

    dp = np.full((rows + 1, cols + 1), np.inf, dtype=np.float64)
    back: list[list[tuple[str, float] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    dp[0, 0] = 0.0

    for row in range(1, rows + 1):
        step_cost = _skip_expected_cost(expected[row - 1], config)
        dp[row, 0] = dp[row - 1, 0] + step_cost
        back[row][0] = ("skip_expected", step_cost)

    for col in range(1, cols + 1):
        step_cost = _skip_observed_cost(observed[col - 1], config)
        dp[0, col] = dp[0, col - 1] + step_cost
        back[0][col] = ("skip_observed", step_cost)

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            best_op = ("skip_expected", _skip_expected_cost(expected[row - 1], config))
            best_cost = dp[row - 1, col] + best_op[1]

            skip_observed_cost = _skip_observed_cost(observed[col - 1], config)
            if dp[row, col - 1] + skip_observed_cost < best_cost:
                best_op = ("skip_observed", skip_observed_cost)
                best_cost = dp[row, col - 1] + skip_observed_cost

            match_cost = _match_cost(expected[row - 1], observed[col - 1], config)
            if dp[row - 1, col - 1] + match_cost < best_cost:
                best_op = ("match", match_cost)
                best_cost = dp[row - 1, col - 1] + match_cost

            dp[row, col] = best_cost
            back[row][col] = best_op

    row = rows
    col = cols
    matches: list[AlignmentMatch] = []
    while row > 0 or col > 0:
        op = back[row][col]
        if op is None:
            break
        action, step_cost = op
        if action == "match":
            expected_unit = expected[row - 1]
            observed_run = observed[col - 1]
            if expected_unit.is_tone == observed_run.is_tone:
                matches.append(AlignmentMatch(expected=expected_unit, observed=observed_run, cost=float(step_cost)))
            row -= 1
            col -= 1
        elif action == "skip_expected":
            row -= 1
        else:
            col -= 1

    matches.reverse()
    coverage = 0.0 if not expected else float(len(matches)) / float(len(expected))
    return AlignmentResult(
        total_cost=float(dp[rows, cols]),
        matched_count=len(matches),
        expected_count=len(expected),
        observed_count=len(observed),
        coverage=coverage,
        matches=matches,
    )


def _active_tone_window(runs: Sequence[DetectedRun], audio_length: int) -> tuple[int, int]:
    tones = [run for run in runs if run.is_tone]
    if not tones:
        return 0, int(audio_length)
    return int(tones[0].start_sample), int(tones[-1].end_sample)


def _build_guided_boundaries(
    expected: Sequence[ExpectedUnit],
    runs: Sequence[DetectedRun],
    audio_length: int,
    sample_rate: int,
) -> tuple[list[int], float | None, tuple[int, int]]:
    active_start, active_end = _active_tone_window(runs, audio_length)
    active_start = max(0, min(active_start, audio_length))
    active_end = max(active_start + 1, min(active_end, audio_length))

    total_nominal_units = float(sum(unit.nominal_units for unit in expected))
    if total_nominal_units <= 0.0:
        return [active_start, active_end], None, (active_start, active_end)

    guided_dot_s = float(active_end - active_start) / float(sample_rate) / total_nominal_units
    candidate_boundaries = sorted(
        {
            active_start,
            active_end,
            *[
                boundary
                for run in runs
                for boundary in (run.start_sample, run.end_sample)
                if active_start <= boundary <= active_end
            ],
        }
    )
    tolerance_samples = max(
        int(round(sample_rate * 0.02)),
        int(round(sample_rate * guided_dot_s * 0.75)),
    )
    min_segment_samples = max(1, int(round(sample_rate * guided_dot_s * 0.12)))

    boundaries = [active_start]
    internal_targets: list[int] = []
    cumulative_units = 0.0
    for unit in expected[:-1]:
        cumulative_units += unit.nominal_units
        internal_targets.append(active_start + int(round(cumulative_units * guided_dot_s * sample_rate)))

    for target_index, target in enumerate(internal_targets):
        previous_boundary = boundaries[-1]
        remaining_segments = len(internal_targets) - target_index
        latest_boundary = active_end - remaining_segments * min_segment_samples
        latest_boundary = max(previous_boundary + min_segment_samples, latest_boundary)
        chosen = int(np.clip(target, previous_boundary + min_segment_samples, latest_boundary))

        valid_candidates = [
            candidate
            for candidate in candidate_boundaries
            if previous_boundary + min_segment_samples <= candidate <= latest_boundary
        ]
        if valid_candidates:
            nearest = min(valid_candidates, key=lambda candidate: abs(candidate - target))
            if abs(nearest - target) <= tolerance_samples:
                chosen = int(nearest)
        boundaries.append(chosen)

    boundaries.append(active_end)
    for index in range(1, len(boundaries)):
        boundaries[index] = max(boundaries[index], boundaries[index - 1] + 1)
    boundaries[-1] = min(boundaries[-1], audio_length)
    return boundaries, guided_dot_s, (active_start, active_end)


def load_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resample_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    audio = ensure_mono_float32(audio)
    if audio.size == 0:
        return np.zeros(max(0, target_length), dtype=np.float32)
    target_length = max(1, int(target_length))
    if target_length == len(audio):
        return audio.astype(np.float32)
    return scipy.signal.resample(audio, target_length).astype(np.float32)


def load_audio_for_detection(path: str | Path, sample_rate: int) -> np.ndarray:
    return load_audio_mono(str(Path(path).expanduser().resolve()), sample_rate=sample_rate)


def _read_audio_preserve_scale(path: str | Path, sample_rate: int) -> np.ndarray:
    path = Path(path).expanduser().resolve()
    audio, source_sample_rate = sf.read(path)
    audio = ensure_mono_float32(audio)
    if audio.size == 0:
        return audio

    if int(source_sample_rate) != int(sample_rate):
        gcd = int(np.gcd(int(source_sample_rate), int(sample_rate)))
        up = int(sample_rate // gcd)
        down = int(source_sample_rate // gcd)
        audio = scipy.signal.resample_poly(audio, up=up, down=down).astype(np.float32)
    return audio.astype(np.float32)


def _fallback_specs(kind: str) -> list[tuple[str, float]]:
    return {
        "dot": [("dash", 1.0 / 3.0)],
        "dash": [("dot", 3.0)],
        "intra_gap": [("letter_gap", 1.0 / 3.0), ("word_gap", 1.0 / 7.0)],
        "letter_gap": [("word_gap", 3.0 / 7.0), ("intra_gap", 3.0)],
        "word_gap": [("letter_gap", 7.0 / 3.0), ("intra_gap", 7.0)],
    }.get(kind, [])


def load_unit_bank(bank_dir: str | Path) -> dict[str, Any]:
    bank_dir = Path(bank_dir).expanduser().resolve()
    payload = json.loads((bank_dir / "bank.json").read_text(encoding="utf-8"))
    payload["bank_dir"] = str(bank_dir)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for entry in payload.get("entries", []):
        by_kind.setdefault(str(entry["kind"]), []).append(entry)
    payload["by_kind"] = by_kind
    return payload


def _sample_bank_audio(
    kind: str,
    bank: dict[str, Any],
    rng: np.random.Generator,
    audio_cache: dict[str, np.ndarray],
    visited: set[str] | None = None,
) -> np.ndarray:
    visited = set() if visited is None else set(visited)
    if kind in visited:
        raise ValueError(f"Recursive fallback while resolving unit kind={kind}")
    visited.add(kind)

    entries = bank["by_kind"].get(kind, [])
    if entries:
        item = entries[int(rng.integers(0, len(entries)))]
        path = str(item["path"])
        if path not in audio_cache:
            audio_cache[path] = _read_audio_preserve_scale(path, sample_rate=int(bank["sample_rate"]))
        return audio_cache[path].copy()

    for fallback_kind, scale in _fallback_specs(kind):
        if bank["by_kind"].get(fallback_kind):
            audio = _sample_bank_audio(fallback_kind, bank, rng, audio_cache, visited=visited)
            target_length = max(1, int(round(len(audio) * scale)))
            return _resample_audio(audio, target_length)

    raise ValueError(f"No audio available for unit kind={kind}")


def _concatenate_with_crossfade(clips: Sequence[np.ndarray], crossfade_samples: int) -> np.ndarray:
    if not clips:
        return np.zeros(0, dtype=np.float32)

    output = ensure_mono_float32(clips[0]).astype(np.float32)
    for clip in clips[1:]:
        clip = ensure_mono_float32(clip).astype(np.float32)
        if output.size == 0:
            output = clip
            continue

        fade = min(int(crossfade_samples), len(output), len(clip))
        if fade <= 0:
            output = np.concatenate([output, clip]).astype(np.float32)
            continue

        fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        blended = output[-fade:] * fade_out + clip[:fade] * fade_in
        output = np.concatenate([output[:-fade], blended, clip[fade:]]).astype(np.float32)
    return output


def synthesize_from_unit_bank(
    text: str,
    bank: dict[str, Any],
    config: SynthesisConfig | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, int]:
    config = config or SynthesisConfig()
    rng = np.random.default_rng(seed)
    expected = build_expected_units(text)
    if not expected:
        return np.zeros(0, dtype=np.float32), int(bank["sample_rate"])

    audio_cache: dict[str, np.ndarray] = {}
    clips: list[np.ndarray] = []
    for unit in expected:
        clip = _sample_bank_audio(unit.kind, bank, rng, audio_cache)
        if config.jitter_ratio > 0.0 and clip.size > 8:
            scale = float(rng.uniform(1.0 - config.jitter_ratio, 1.0 + config.jitter_ratio))
            target_length = max(1, int(round(len(clip) * scale)))
            clip = _resample_audio(clip, target_length)
        clips.append(clip)

    sample_rate = int(bank["sample_rate"])
    crossfade_samples = int(round(sample_rate * config.crossfade_ms / 1000.0))
    audio = _concatenate_with_crossfade(clips, crossfade_samples=crossfade_samples)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = (audio / peak * float(config.peak_normalize)).astype(np.float32)
    return audio.astype(np.float32), sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, ensure_mono_float32(audio), sample_rate)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def extract_unit_bank_entries(
    audio: np.ndarray,
    text: str,
    source_id: str,
    out_dir: str | Path,
    detection_config: DetectionConfig | None = None,
    alignment_config: AlignmentConfig | None = None,
    extract_pad_ms: float = 5.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detection_config = detection_config or DetectionConfig()
    alignment_config = alignment_config or AlignmentConfig()
    out_dir = Path(out_dir).expanduser().resolve()
    audio = ensure_mono_float32(audio)

    expected = build_expected_units(text)
    detection = detect_morse_runs(audio, config=detection_config)
    boundaries, guided_dot_s, active_window = _build_guided_boundaries(
        expected=expected,
        runs=detection.runs,
        audio_length=len(audio),
        sample_rate=detection.sample_rate,
    )
    aligned_runs = [
        DetectedRun(
            index=run.index,
            is_tone=run.is_tone,
            start_frame=run.start_frame,
            end_frame=run.end_frame,
            start_sample=run.start_sample,
            end_sample=run.end_sample,
            duration_s=run.duration_s,
            units=None if guided_dot_s is None or guided_dot_s <= 0.0 else run.duration_s / guided_dot_s,
        )
        for run in detection.runs
    ]
    alignment = align_expected_to_observed(expected, aligned_runs, config=alignment_config)

    entries: list[dict[str, Any]] = []
    for unit_index, expected_unit in enumerate(expected):
        start_sample = max(0, boundaries[unit_index])
        end_sample = min(len(audio), boundaries[unit_index + 1])
        if end_sample <= start_sample:
            continue

        clip = ensure_mono_float32(audio[start_sample:end_sample])
        rel_path = Path("units") / expected_unit.kind / f"{source_id}_{unit_index:04d}.wav"
        abs_path = out_dir / rel_path
        write_wav(abs_path, clip, sample_rate=detection.sample_rate)
        entries.append(
            {
                "kind": expected_unit.kind,
                "path": str(abs_path),
                "relative_path": str(rel_path),
                "source_id": source_id,
                "text": format_label_text(text, label_format="char_spaced"),
                "canonical_text": canonicalize_label_text(text),
                "char": expected_unit.char,
                "expected_index": expected_unit.index,
                "observed_index": None,
                "start_s": float(start_sample) / float(detection.sample_rate),
                "end_s": float(end_sample) / float(detection.sample_rate),
                "duration_s": float(end_sample - start_sample) / float(detection.sample_rate),
                "expected_units": float(expected_unit.nominal_units),
                "observed_units": None if guided_dot_s is None or guided_dot_s <= 0.0 else float((end_sample - start_sample) / detection.sample_rate / guided_dot_s),
                "match_cost": None,
                "target_frequency_hz": detection.target_frequency_hz,
                "dot_unit_s": guided_dot_s,
                "segmentation_mode": "guided_boundaries",
            }
        )

    record_summary = {
        "source_id": source_id,
        "canonical_text": canonicalize_label_text(text),
        "formatted_text": format_label_text(text, label_format="char_spaced"),
        "matched_count": alignment.matched_count,
        "expected_count": alignment.expected_count,
        "observed_count": alignment.observed_count,
        "coverage": alignment.coverage,
        "total_cost": alignment.total_cost,
        "guided_boundary_count": len(boundaries) - 1,
        "active_start_s": float(active_window[0]) / float(detection.sample_rate),
        "active_end_s": float(active_window[1]) / float(detection.sample_rate),
        "target_frequency_hz": detection.target_frequency_hz,
        "dot_unit_s": detection.dot_unit_s,
        "guided_dot_unit_s": guided_dot_s,
    }
    return entries, record_summary


def summarize_entries(entries: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[str(entry["kind"])] = counts.get(str(entry["kind"]), 0) + 1
    return counts


def save_unit_bank(
    out_dir: str | Path,
    sample_rate: int,
    entries: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir).expanduser().resolve()
    payload = {
        "sample_rate": int(sample_rate),
        "entry_count": len(entries),
        "counts_by_kind": summarize_entries(entries),
        "records": list(records),
        "entries": list(entries),
        "meta": meta or {},
    }
    write_json(out_dir / "bank.json", payload)
    return payload


def build_manifest_rows(
    generated_dir: str | Path,
    records: Sequence[dict[str, Any]],
    file_name: str,
) -> Path:
    generated_dir = Path(generated_dir).expanduser().resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = generated_dir / file_name
    fieldnames = list(records[0].keys()) if records else ["path", "text", "canonical_text", "split"]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    return manifest_path
